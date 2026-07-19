from __future__ import annotations

import asyncio
import inspect
import time

from rag_core.gateway.adaptive.contracts import RoutingFeedback
from rag_core.gateway.adaptive.enrichment import MemvidEnricher
from rag_core.gateway.adaptive.feedback_store import FeedbackStore
from rag_core.gateway.coordinator import RetrievalCoordinator
from rag_core.gateway.connector import SearchRequest


class AdaptiveLoop:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._coordinator = RetrievalCoordinator()

    async def run(
        self, request, connectors: dict, *, memory=None, planner=None, feedback=None, enricher=None,
        active_revision_path=None, embedding_profile_id=None, feedback_path=None, enrichment_path=None,
    ) -> dict:
        if feedback is None and feedback_path is not None:
            feedback = FeedbackStore(feedback_path)
        if enricher is None and enrichment_path is not None:
            enricher = MemvidEnricher(enrichment_path)
        started_at = time.perf_counter()
        available_connectors = dict(connectors)
        memory_key = getattr(memory, "source", "agent_memory") if memory is not None else None
        if self.enabled and memory is not None:
            available_connectors[memory_key] = memory
        plan = None
        coordinator = RetrievalCoordinator(available_connectors)
        availability = await coordinator.health_map()
        failed_sources: set[str] = set(coordinator.last_failed_sources)
        kind_availability = {
            kind: any(
                available for name, available in availability.items()
                if getattr(available_connectors[name], "retrieval_kind", "live") == kind
            )
            for kind in {
                getattr(connector, "retrieval_kind", "live")
                for connector in available_connectors.values()
            }
        }
        if self.enabled and planner is not None:
            plan = planner.plan(request.query, kind_availability, {})

        selected_connectors = self._selected_connectors(connectors, plan)
        if self.enabled and memory is not None:
            memory_selected = plan is not None and (
                plan.include_memory
                or "memory" in plan.sources
                or getattr(memory, "source", None) in plan.sources
            )
            if availability.get(memory_key, False) and memory_selected:
                selected_connectors[memory_key] = memory
        include_web = plan.include_web if plan is not None else request.include_web
        top_k = plan.top_k if plan is not None else request.topk
        queries = plan.queries if plan is not None else (request.query,)
        coordinator = RetrievalCoordinator(selected_connectors)
        evidence = []
        successful_sources: set[str] = set()
        timed_out_sources: set[str] = set()
        timed_out_queries: list[str] = []
        skipped_queries: list[str] = []
        budget_seconds = (
            plan.retrieval_budget_ms / 1000
            if plan is not None and plan.retrieval_budget_ms is not None
            else None
        )
        query_index = 0
        try:
            async with asyncio.timeout(
                None if budget_seconds is None else budget_seconds + 0.02
            ):
                for query_index, query in enumerate(queries):
                    search_request = SearchRequest(
                        query=query,
                        topk=top_k,
                        domain=request.domain,
                        collection=request.collection,
                        include_web=include_web,
                        continuation_token=request.continuation_token,
                    )
                    results = await coordinator.search(search_request)
                    evidence.extend(results)
                    failed_sources.update(coordinator.last_failed_sources)
                    successful_sources.update(coordinator.last_successful_sources)
        except TimeoutError:
            timed_out_queries.append(queries[query_index])
            skipped_queries.extend(queries[query_index + 1:])
            timed_out_sources.update(coordinator.last_timed_out_sources or selected_connectors)
            failed_sources.update(timed_out_sources)

        fused = self._coordinator.fuse(evidence)[:top_k]
        if self.enabled and feedback is not None:
            feedback.record(RoutingFeedback(
                query=request.query,
                plan_id=request.query if plan is None else plan.original_query,
                selected_sources=tuple(selected_connectors),
                successful_sources=tuple(sorted(successful_sources)),
                useful_document_ids=tuple(item.document_id for item in fused), result_count=len(fused),
                latency_ms=round((time.perf_counter() - started_at) * 1000),
            ))
        if self.enabled and enricher is not None:
            episode = enricher.build_episode(
                request.query, fused, successful=bool(fused), index_revision=active_revision_path,
                embedding_profile_id=embedding_profile_id,
            )
            persist = getattr(enricher, "persist_episode", None)
            if persist is not None:
                persisted = persist(episode)
                if inspect.isawaitable(persisted):
                    await persisted
        return {
            "results": [item.__dict__ for item in fused],
            "mode": "adaptive" if self.enabled else "reference",
            "metadata": {
                "failed_sources": sorted(failed_sources),
                "timed_out_sources": sorted(timed_out_sources),
                "timed_out_queries": timed_out_queries,
                "skipped_queries": skipped_queries,
            },
        }

    @staticmethod
    def _selected_connectors(connectors: dict, plan) -> dict:
        if plan is None:
            return dict(connectors)
        selected_sources = set(plan.sources)
        return {
            name: connector for name, connector in connectors.items()
            if (
                name in selected_sources
                or getattr(connector, "source", None) in selected_sources
                or (
                    getattr(connector, "retrieval_kind", "live") == "local"
                    and plan.include_local
                )
                or (
                    getattr(connector, "retrieval_kind", "live") == "live"
                    and plan.include_live
                )
                or (
                    getattr(connector, "retrieval_kind", "live") == "web"
                    and plan.include_web
                )
            )
        }
