"""Live Confluence retrieval through the Confluence REST API."""
from __future__ import annotations

from html.parser import HTMLParser
import re
from typing import Any

import httpx

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.models import Evidence, EvidenceOrigin, SyncBatch


class ConfluenceConnector:
    retrieval_kind = "live"

    def __init__(self, base_url: str, token: str, source: str = "confluence") -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self.source = source

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        payload = await self._get(
            "/rest/api/content/search",
            params={
                "cql": f'text~"{_escape_query(request.query)}"',
                "limit": request.topk,
                "expand": "body.storage",
            },
        )
        return [_evidence(page, self._base, self.source) for page in payload.get("results", [])]

    async def health(self) -> dict[str, object]:
        try:
            await self._get("/rest/api/content", params={"limit": 1})
        except Exception as exc:
            return {"source": self.source, "available": False, "reason": str(exc)}
        return {"source": self.source, "available": True}

    async def sync_changes(self, cursor: str | None) -> SyncBatch:
        del cursor
        return SyncBatch(added=[])

    async def fetch(self, ref: object) -> object:
        del ref
        raise NotImplementedError("Confluence fetch is not implemented")

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=30.0,
            trust_env=False,
        ) as client:
            response = await client.get(f"{self._base}{path}", params=params)
            response.raise_for_status()
            return response.json()


def extract_storage_text(page: dict[str, Any]) -> str:
    storage = (page.get("body") or {}).get("storage") or {}
    parser = _StorageTextParser()
    parser.feed(str(storage.get("value") or ""))
    return parser.text


class _StorageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._parts.append(data.strip())

    @property
    def text(self) -> str:
        return re.sub(r"\s+([,.;:!?])", r"\1", " ".join(self._parts))


def _escape_query(query: str) -> str:
    return query.replace("\\", "\\\\").replace('"', '\\"')


def _evidence(page: dict[str, Any], base_url: str, source: str) -> Evidence:
    page_id = str(page["id"])
    return Evidence(
        id=f"{source}:{page_id}",
        document_id=page_id,
        title=str(page.get("title") or ""),
        text=extract_storage_text(page),
        source=source,
        uri=f"{base_url}/pages/viewpage.action?pageId={page_id}",
        origin=EvidenceOrigin.LIVE_CORPORATE,
    )
