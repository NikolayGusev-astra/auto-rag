from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
import logging

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.boosting import apply_exact_match_boost
from rag_core.gateway.deduplication import deduplicate_evidence
from rag_core.gateway.models import Evidence
from rag_core.gateway.stage_timer import StageTimer


EvidenceFilter = Callable[[Evidence], bool]
logger = logging.getLogger(__name__)


class RetrievalCoordinator:
    def __init__(
        self,
        connectors: Mapping[str, SourceConnector] | None = None,
        filters: Iterable[EvidenceFilter] = (),
        reranker: object | None = None,
        exact_id_boost: float = 1.0,
        exact_slug_title_boost: float = 0.7,
    ) -> None:
        self._connectors = dict(connectors or {})
        self._filters = tuple(filters)
        self._reranker = reranker
        self._exact_id_boost = exact_id_boost
        self._exact_slug_title_boost = exact_slug_title_boost
        self.last_failed_sources: list[str] = []
        self.last_successful_sources: list[str] = []
        self.last_timed_out_sources: list[str] = []
        self.last_latency: dict[str, dict[str, object]] = {}
        self.last_deduplication: dict[str, int] = {"input": 0, "output": 0, "removed": 0}

    async def search(self, request: SearchRequest) -> list[Evidence]:
        self.last_failed_sources = []
        self.last_successful_sources = []
        self.last_timed_out_sources = []
        timer = StageTimer()
        timer.start("search")
        evidence: list[Evidence] = []
        try:
            for name, connector in self._connectors.items():
                stage = self._latency_stage(name, connector)
                if self._is_web_connector(connector) and not request.include_web:
                    timer.skip(stage, reason="web_disabled")
                    continue
                try:
                    available = await self._is_available(connector)
                except Exception as error:
                    self._record_failure(name, error)
                    timer.skip(stage, reason="health_check_failed")
                    continue
                if not available:
                    timer.skip(stage, reason="unavailable")
                    continue
                timer.start(stage)
                try:
                    results = await connector.search_live(request)
                except asyncio.CancelledError:
                    timer.stop(stage, status="cancelled")
                    self.last_timed_out_sources.append(name)
                    raise
                except Exception as error:
                    timer.stop(stage, status="failed")
                    self._record_failure(name, error)
                    continue
                timer.stop(stage)
                if results:
                    self.last_successful_sources.append(name)
                    evidence.extend(results)
            return await self._finalize(request, evidence)
        finally:
            timer.stop("search")
            self.last_latency = timer.summary()

    async def _finalize(self, request: SearchRequest, evidence: list[Evidence]) -> list[Evidence]:
        fused = self.fuse(
            apply_exact_match_boost(
                request.query,
                item,
                exact_id_boost=self._exact_id_boost,
                exact_slug_title_boost=self._exact_slug_title_boost,
            )
            for item in evidence
        )
        if self._reranker is not None:
            try:
                fused = await self._reranker.rerank(request.query, fused, request.topk)
            except Exception as error:
                self._record_failure("reranker", error)
                fused = fused[:request.topk]
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

    @classmethod
    def _latency_stage(cls, name: str, connector: SourceConnector) -> str:
        label = f"{name} {getattr(connector, 'source', '')}".lower()
        if "browser" in label or "camoufox" in label:
            return f"browser_fallback:{name}"
        if cls._is_web_connector(connector):
            return f"web:{name}"
        return name

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
        deduplicated = deduplicate_evidence(balanced)
        self.last_deduplication = {
            "input": len(balanced),
            "output": len(deduplicated),
            "removed": len(balanced) - len(deduplicated),
        }
        return deduplicated
