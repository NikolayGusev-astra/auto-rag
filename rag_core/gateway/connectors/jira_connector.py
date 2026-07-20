"""Live Jira retrieval through the Jira REST API."""
from __future__ import annotations

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
        payload = await self._get(
            "/rest/api/2/search",
            params={
                "jql": f'text~"{_escape_query(request.query)}"',
                "maxResults": request.topk,
                "fields": "summary,description,updated",
            },
        )
        return [_evidence(issue, self._base, self.source) for issue in payload.get("issues", [])]

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


def _evidence(issue: dict[str, Any], base_url: str, source: str) -> Evidence:
    fields = issue.get("fields") or {}
    key = str(issue["key"])
    summary = str(fields.get("summary") or "")
    description = fields.get("description") or ""
    if isinstance(description, dict):
        description = description.get("content") or ""
    return Evidence(
        id=f"{source}:{key}",
        document_id=key,
        title=summary,
        text=f"{summary}\n{description}",
        source=source,
        uri=f"{base_url}/browse/{key}",
        origin=EvidenceOrigin.LIVE_CORPORATE,
        metadata={"updated": fields.get("updated")},
    )
