"""Local, lexical retrieval over persisted Memvid enrichment episodes."""
from __future__ import annotations

from rag_core.gateway.adaptive.enrichment import MemvidEnricher
from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.models import Evidence


class MemvidConnector(SourceConnector):
    source = "memvid"
    retrieval_kind = "local"

    def __init__(self, enricher: MemvidEnricher) -> None:
        self._enricher = enricher

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        return self._enricher.search_episodes(request.query, request.topk)

    async def fetch(self, ref: object) -> object:
        raise NotImplementedError("fetch is not implemented for Memvid episodes")

    async def sync_changes(self, cursor: str | None) -> object:
        raise NotImplementedError("Memvid episodes are updated by enrichment")

    async def health(self) -> dict[str, bool]:
        return {"available": bool(self._enricher.episodes)}
