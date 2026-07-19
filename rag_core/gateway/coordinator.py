from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.models import Evidence


EvidenceFilter = Callable[[Evidence], bool]


class RetrievalCoordinator:
    def __init__(
        self,
        connectors: Mapping[str, SourceConnector] | None = None,
        filters: Iterable[EvidenceFilter] = (),
        reranker: object | None = None,
    ) -> None:
        self._connectors = dict(connectors or {})
        self._filters = tuple(filters)
        self._reranker = reranker

    async def search(self, request: SearchRequest) -> list[Evidence]:
        evidence = []
        for connector in self._connectors.values():
            if self._is_web_connector(connector) and not request.include_web:
                continue
            if not await self._is_available(connector):
                continue
            evidence.extend(await connector.search_live(request))
        fused = self.fuse(evidence)
        if self._reranker is not None:
            fused = await self._reranker.rerank(request.query, fused, request.topk)
        return fused[:request.topk]

    @staticmethod
    def _is_web_connector(connector: SourceConnector) -> bool:
        return getattr(connector, "source", "").lower() in {"web", "public_web"}

    @staticmethod
    async def _is_available(connector: SourceConnector) -> bool:
        try:
            health = await connector.health()
        except Exception:
            return False
        if isinstance(health, Mapping):
            return bool(health.get("available", False))
        return bool(getattr(health, "available", False))

    def fuse(self, evidence: Iterable[Evidence]) -> list[Evidence]:
        best_by_document: dict[str, Evidence] = {}
        for item in evidence:
            if item.metadata.get("deprecated") or any(
                not evidence_filter(item) for evidence_filter in self._filters
            ):
                continue
            final_score = item.retrieval_score
            if item.reranker_score is not None:
                final_score = 0.4 * item.retrieval_score + 0.6 * item.reranker_score
            scored = replace(item, final_score=round(final_score, 4))
            current = best_by_document.get(scored.document_id)
            if current is None or scored.final_score > current.final_score:
                best_by_document[scored.document_id] = scored
        by_source: dict[str, list[Evidence]] = {}
        for item in best_by_document.values():
            by_source.setdefault(item.source, []).append(item)
        for items in by_source.values():
            items.sort(key=lambda item: item.final_score, reverse=True)

        ordered_sources = sorted(
            by_source, key=lambda source: by_source[source][0].final_score, reverse=True
        )
        balanced: list[Evidence] = []
        while any(by_source.values()):
            for source in ordered_sources:
                if by_source[source]:
                    balanced.append(by_source[source].pop(0))
        return balanced
