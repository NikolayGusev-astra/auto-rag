from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
import logging

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.models import Evidence


EvidenceFilter = Callable[[Evidence], bool]
logger = logging.getLogger(__name__)


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
        self.last_failed_sources: list[str] = []
        self.last_successful_sources: list[str] = []
        self.last_timed_out_sources: list[str] = []

    async def search(self, request: SearchRequest) -> list[Evidence]:
        self.last_failed_sources = []
        self.last_successful_sources = []
        self.last_timed_out_sources = []
        health = await self.health_map()
        evidence = []
        for name, connector in self._connectors.items():
            if self._is_web_connector(connector) and not request.include_web:
                continue
            if not health.get(name, False):
                continue
            try:
                results = await connector.search_live(request)
            except asyncio.CancelledError:
                self.last_timed_out_sources.append(name)
                raise
            except Exception as error:
                self._record_failure(name, error)
                continue
            if results:
                self.last_successful_sources.append(name)
                evidence.extend(results)
        fused = self.fuse(evidence)
        if self._reranker is not None:
            fused = await self._reranker.rerank(request.query, fused, request.topk)
        return fused[:request.topk]

    async def health_map(self) -> dict[str, bool]:
        availability: dict[str, bool] = {}
        for name, connector in self._connectors.items():
            try:
                availability[name] = await self._is_available(connector)
            except Exception as error:
                self._record_failure(name, error)
                availability[name] = False
        return availability

    @staticmethod
    def _is_web_connector(connector: SourceConnector) -> bool:
        return (
            getattr(connector, "retrieval_kind", None) == "web"
            or getattr(connector, "source", "").lower() in {"web", "public_web"}
        )

    @staticmethod
    async def _is_available(connector: SourceConnector) -> bool:
        health = await connector.health()
        if isinstance(health, Mapping):
            return bool(health.get("available", False))
        return bool(getattr(health, "available", False))

    def _record_failure(self, source: str, error: Exception) -> None:
        if source not in self.last_failed_sources:
            self.last_failed_sources.append(source)
        logger.warning(
            "retrieval connector failed",
            extra={"source": source, "error_type": type(error).__name__, "error_message": str(error)},
        )

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
