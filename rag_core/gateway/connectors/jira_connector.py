"""Live Jira retrieval — full-content diagnostics edition."""

from __future__ import annotations

import re
from typing import Any

import httpx

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.models import Evidence, EvidenceOrigin, SyncBatch

# ── Tunables ──────────────────────────────────────────────────────
_MAX_COMMENTS = 50       # per page    → Jira default page size
_MAX_COMMENT_PAGES = 10  # max pages   → 500 comments total
_MAX_LINKED = 5          # linked issues whose content we fetch
# ───────────────────────────────────────────────────────────────────


class JiraConnector:
    retrieval_kind = "live"

    def __init__(self, base_url: str, token: str, source: str = "jira") -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self.source = source

    # ── search_live ────────────────────────────────────────────────

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        issue_key = _extract_issue_key(request.query)
        jql_queries: list[str] = []
        if issue_key:
            jql_queries.append(f"issueKey={issue_key}")
        jql_queries.append(f'text~"{_escape_query(request.query)}"')

        issues: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for jql in jql_queries:
            payload = await self._get(
                "/rest/api/2/search",
                params={
                    "jql": jql,
                    "maxResults": request.topk,
                    "fields": "summary,description,updated,issuelinks",
                },
            )
            for issue in payload.get("issues", []):
                key = str(issue["key"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    issues.append(issue)

        evidence_list: list[Evidence] = []
        for issue in issues[: request.topk]:
            key = str(issue["key"])

            enrichment: dict[str, Any] = {
                "comments_total": 0,
                "comments_loaded": 0,
                "comments_status": "ok",
                "linked_issues_loaded": 0,
                "linked_issues_status": "ok",
            }

            comments_text = ""
            linked_meta: list[dict[str, str]] = []
            linked_content: list[dict[str, str]] = []

            if issue_key and key == issue_key:
                comments_text, c_total, c_loaded, c_err = await self._fetch_comments(key)
                enrichment["comments_total"] = c_total
                enrichment["comments_loaded"] = c_loaded
                if c_err:
                    enrichment["comments_status"] = "failed"
                    enrichment["comments_error"] = c_err

                linked_meta, linked_content, lk_loaded, lk_err = await self._fetch_linked_issues(key)
                enrichment["linked_issues_loaded"] = lk_loaded
                if lk_err:
                    enrichment["linked_issues_status"] = "failed"
                    enrichment["linked_issues_error"] = lk_err

            ev = _evidence(
                issue, self._base, self.source,
                comments_text, linked_meta, linked_content, enrichment,
            )
            evidence_list.append(ev)

        return evidence_list

    # ── comments (paginated) ───────────────────────────────────────

    async def _fetch_comments(self, key: str) -> tuple[str, int, int, str]:
        parts: list[str] = []
        total = 0
        loaded = 0
        error = ""
        for page in range(_MAX_COMMENT_PAGES):
            try:
                data = await self._get(
                    f"/rest/api/2/issue/{key}/comment",
                    params={"startAt": page * _MAX_COMMENTS, "maxResults": _MAX_COMMENTS},
                )
            except Exception as exc:
                error = str(exc)[:200]
                break
            comments = data.get("comments", [])
            total = data.get("total", len(comments))
            if not comments:
                break
            for c in comments:
                author = (c.get("author") or {}).get("displayName", "unknown")
                created = c.get("created", "")
                body = c.get("body", "")
                if isinstance(body, dict):
                    body = body.get("content", str(body))
                parts.append(f"[{created}] {author}: {body}")
                loaded += 1
            if len(comments) < _MAX_COMMENTS:
                break
        return "\n\n".join(parts), total, loaded, error

    # ── linked issues (keys + content of first N) ──────────────────

    async def _fetch_linked_issues(
        self, key: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], int, str]:
        meta: list[dict[str, str]] = []
        content: list[dict[str, str]] = []
        error = ""
        try:
            data = await self._get(f"/rest/api/2/issue/{key}")
            fields = data.get("fields") or {}
            links = fields.get("issuelinks") or []
            attempted = 0
            for link in links:
                link_type = (link.get("type") or {}).get("name", "relates to")
                target = link.get("outwardIssue") or link.get("inwardIssue")
                if target is None:
                    continue
                lk = str(target.get("key", ""))
                ls = str((target.get("fields") or {}).get("summary", ""))
                meta.append({"key": lk, "summary": ls, "type": str(link_type)})
                if attempted < _MAX_LINKED:
                    attempted += 1
                    try:
                        lk_data = await self._get(
                            f"/rest/api/2/issue/{lk}",
                            params={"fields": "summary,description"},
                        )
                        lk_fields = lk_data.get("fields") or {}
                        lk_desc = lk_fields.get("description") or ""
                        if isinstance(lk_desc, dict):
                            lk_desc = lk_desc.get("content", str(lk_desc))
                    except Exception as exc:
                        error = error or str(exc)[:200]
                        continue
                    content.append({
                        "key": lk,
                        "summary": ls,
                        "description": str(lk_desc)[:2000],
                    })
        except Exception as exc:
            error = str(exc)[:200]
        return meta, content, len(content), error

    # ── health / sync / fetch ──────────────────────────────────────

    async def health(self) -> dict[str, object]:
        try:
            await self._get("/rest/api/2/myself")
        except Exception as exc:
            return {"source": self.source, "available": False, "reason": str(exc)}
        return {"source": self.source, "available": True}

    async def sync_changes(self, cursor: str | None) -> SyncBatch:
        return SyncBatch(added=[])

    async def fetch(self, ref: object) -> object:
        raise NotImplementedError("Jira fetch is not implemented")

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=30.0,
            trust_env=False,
        ) as client:
            response = await client.get(f"{self._base}{path}", params=params)
            response.raise_for_status()
            return response.json()


# ── helpers ────────────────────────────────────────────────────────

def _escape_query(query: str) -> str:
    return query.replace("\\", "\\\\").replace('"', '\\"')


def _extract_issue_key(query: str) -> str | None:
    match = re.search(r"\b([A-Z]+-\d+)\b", query)
    return match.group(1) if match else None


def _evidence(
    issue: dict[str, Any],
    base_url: str,
    source: str,
    comments: str = "",
    linked_meta: list[dict[str, str]] | None = None,
    linked_content: list[dict[str, str]] | None = None,
    enrichment: dict[str, Any] | None = None,
) -> Evidence:
    fields = issue.get("fields") or {}
    key = str(issue["key"])
    summary = str(fields.get("summary") or "")
    description = fields.get("description") or ""
    if isinstance(description, dict):
        description = description.get("content") or ""

    text = f"{summary}\n{description}"
    if comments:
        text += f"\n\n--- COMMENTS ---\n{comments}"
    if linked_content:
        parts = []
        for lc in linked_content:
            parts.append(f"[LINKED] {lc['key']}: {lc['summary']}\n{lc['description']}")
        text += "\n\n--- LINKED ISSUES ---\n" + "\n\n".join(parts)

    metadata: dict[str, Any] = {
        "updated": fields.get("updated"),
        "comments": comments[:500] if comments else "",
    }
    if linked_meta:
        metadata["linked_issues"] = linked_meta
    if enrichment:
        metadata["enrichment"] = enrichment

    return Evidence(
        id=f"{source}:{key}",
        document_id=key,
        title=summary,
        text=text,
        source=source,
        uri=f"{base_url}/browse/{key}",
        origin=EvidenceOrigin.LIVE_CORPORATE,
        metadata=metadata,
    )
