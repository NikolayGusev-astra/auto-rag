"""Live Jira retrieval through the Jira REST API — full-content diagnostics edition.

Exactly what changed (per the SIRIUS-195479 regression analysis):

* Before: summary + description only. Comments, linked issues, and attachments were
  missing from the indexed evidence.  Result: Auto-RAG could not answer questions
  about OS versions, resolutions, or related tickets (all in comments).
* After: exact-key hits fetch comments AND linked issues alongside the issue body.
  The evidence text now contains the full comment thread; linked-issue keys and
  summaries appear in metadata.  This closes the biggest gap found in the ЦБ РФ
  investigation.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.models import Evidence, EvidenceOrigin, SyncBatch


class JiraConnector:
    retrieval_kind = "live"

    def __init__(self, base_url: str, token: str, source: str = "jira") -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self.source = source

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
            comments_text = ""
            linked: list[dict[str, str]] = []

            # ── Exact-key enrichment: pull comments + linked issues ──
            if issue_key and key == issue_key:
                comments_text = await self._fetch_comments(key)
                linked = await self._fetch_linked_issues(key)

            ev = _evidence(issue, self._base, self.source, comments_text, linked)
            evidence_list.append(ev)

        return evidence_list

    async def _fetch_comments(self, key: str) -> str:
        """Pull the full comment thread for a single issue."""
        try:
            data = await self._get(f"/rest/api/2/issue/{key}/comment")
            comments = data.get("comments", [])
            if not comments:
                return ""
            parts: list[str] = []
            for c in comments[:50]:  # guard against gigantic threads
                author = (c.get("author") or {}).get("displayName", "unknown")
                created = c.get("created", "")
                body = c.get("body", "")
                if isinstance(body, dict):
                    body = body.get("content", str(body))
                parts.append(f"[{created}] {author}: {body}")
            return "\n\n".join(parts)
        except Exception:
            return ""

    async def _fetch_linked_issues(self, key: str) -> list[dict[str, str]]:
        """Pull linked-issue references (blocked by, relates to, etc.)."""
        try:
            data = await self._get(f"/rest/api/2/issue/{key}")
            fields = data.get("fields") or {}
            links = fields.get("issuelinks") or []
            result: list[dict[str, str]] = []
            for link in links[:20]:
                link_type = (link.get("type") or {}).get("name", "relates to")
                target = link.get("outwardIssue") or link.get("inwardIssue")
                if target is None:
                    continue
                result.append({
                    "key": str(target.get("key", "")),
                    "summary": str((target.get("fields") or {}).get("summary", "")),
                    "type": str(link_type),
                })
            return result
        except Exception:
            return []

    async def health(self) -> dict[str, object]:
        try:
            await self._get("/rest/api/2/myself")
        except Exception as exc:
            return {"source": self.source, "available": False, "reason": str(exc)}
        return {"source": self.source, "available": True}

    async def sync_changes(self, cursor: str | None) -> SyncBatch:
        del cursor
        return SyncBatch(added=[])

    async def fetch(self, ref: object) -> object:
        del ref
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
    linked: list[dict[str, str]] | None = None,
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

    metadata: dict[str, Any] = {"updated": fields.get("updated")}
    if linked:
        metadata["linked_issues"] = linked

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
