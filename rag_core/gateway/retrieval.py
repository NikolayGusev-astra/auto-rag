"""Reusable retrieval entrypoint for gateway and legacy adapters."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.coordinator import RetrievalCoordinator
from rag_core.gateway.models import Evidence, EvidenceOrigin


class _EvidenceConnector:
    """Adapt lightweight connectors to the coordinator contract."""

    def __init__(self, connector: SourceConnector) -> None:
        self._connector = connector
        self.source = getattr(connector, "source", "unknown")

    async def health(self) -> object:
        health = getattr(self._connector, "health", None)
        return await health() if health is not None else {"available": True}

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        raw_results = await self._connector.search_live(request)
        return [
            item if isinstance(item, Evidence) else _evidence_from_mapping(item, self.source)
            for item in raw_results
        ]


def _evidence_from_mapping(item: Mapping[str, Any], source: str) -> Evidence:
    document_id = str(item.get("document_id") or item.get("id") or item.get("uri") or "unknown")
    return Evidence(
        id=str(item.get("id") or document_id),
        document_id=document_id,
        title=str(item.get("title") or ""),
        text=str(item.get("text") or item.get("content") or ""),
        source=str(item.get("source") or source),
        uri=item.get("uri") or item.get("url"),
        origin=EvidenceOrigin.LOCAL_SNAPSHOT,
        retrieval_score=float(item.get("retrieval_score", item.get("score", 0.0))),
        metadata=dict(item.get("metadata") or {}),
    )


async def retrieve(
    request: SearchRequest,
    connectors: Mapping[str, SourceConnector],
    reranker: object | None = None,
) -> list[Evidence]:
    """Search connectors, normalize their output, and fuse it into Evidence."""
    adapted = {name: _EvidenceConnector(connector) for name, connector in connectors.items()}
    return await RetrievalCoordinator(adapted, reranker=reranker).search(request)
