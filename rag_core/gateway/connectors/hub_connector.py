"""Live Automation Hub retrieval through the Galaxy v3 API."""
from __future__ import annotations

from typing import Any

import httpx

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.models import Evidence, EvidenceOrigin, SyncBatch


class HubConnector:
    retrieval_kind = "live"

    def __init__(self, base_url: str, token: str, source: str = "hub") -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self.source = source

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        collections_payload = await self._get(
            "/api/galaxy/v3/collections/",
            params={"namespace": "astra", "limit": 50},
        )
        index_payload = await self._get(
            "/api/galaxy/v3/plugin/ansible/content/published/collections/index/",
            params={"namespace": "astra", "keywords": request.query},
        )
        collections = _merge_collections(
            _matching_collections(_items(collections_payload), request.query),
            _items(index_payload),
        )
        results: list[Evidence] = []
        for collection in collections[: request.topk]:
            namespace = _namespace(collection)
            name = str(collection.get("name") or "")
            if not namespace or not name:
                continue
            versions = await self._get(
                f"/api/galaxy/v3/plugin/ansible/content/published/{namespace}/{name}/"
            )
            latest = _latest_version(versions)
            results.append(_evidence(namespace, name, latest, self._base, self.source))
        return results

    async def health(self) -> dict[str, object]:
        try:
            await self._get("/api/galaxy/v3/collections/", params={"limit": 1})
        except Exception as exc:
            return {"source": self.source, "available": False, "reason": str(exc)}
        return {"source": self.source, "available": True}

    async def sync_changes(self, cursor: str | None) -> SyncBatch:
        del cursor
        return SyncBatch(added=[])

    async def fetch(self, ref: object) -> object:
        del ref
        raise NotImplementedError("Hub fetch is not implemented")

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        async with httpx.AsyncClient(
            headers={"Authorization": f"Token {self._token}"},
            timeout=30.0,
            trust_env=False,
            follow_redirects=True,
        ) as client:
            response = await client.get(f"{self._base}{path}", params=params)
            response.raise_for_status()
            return response.json()


def _items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    values = payload.get("data", payload.get("results", []))
    return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []


def _namespace(collection: dict[str, Any]) -> str:
    namespace = collection.get("namespace")
    if isinstance(namespace, dict):
        return str(namespace.get("name") or "")
    return str(namespace or "")


def _matching_collections(collections: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    keywords = query.casefold().split()
    return [
        collection
        for collection in collections
        if all(keyword in str(collection.get("name") or "").casefold() for keyword in keywords)
    ]


def _merge_collections(*collection_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for collections in collection_lists:
        for collection in collections:
            key = (_namespace(collection).casefold(), str(collection.get("name") or "").casefold())
            if not all(key) or key in seen:
                continue
            seen.add(key)
            merged.append(collection)
    return merged


def _latest_version(payload: Any) -> str:
    versions = _items(payload)
    if not versions:
        return "unknown"
    return str(versions[0].get("version") or "unknown")


def _evidence(namespace: str, name: str, version: str, base_url: str, source: str) -> Evidence:
    return Evidence(
        id=f"{source}:{namespace}.{name}",
        document_id=f"{namespace}.{name}",
        title=name,
        text=f"collection: {name} latest: {version}",
        source=source,
        uri=f"{base_url}/ui/repo/published/{namespace}/{name}",
        origin=EvidenceOrigin.LIVE_CORPORATE,
    )
