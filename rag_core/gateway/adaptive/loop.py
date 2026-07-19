from __future__ import annotations

import inspect
import time

from rag_core.gateway.adaptive.contracts import RoutingFeedback
from rag_core.gateway.coordinator import RetrievalCoordinator
from rag_core.gateway.connector import SearchRequest


class AdaptiveLoop:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._coordinator = RetrievalCoordinator()

    async def run(
        self, request, connectors: dict, *, memory=None, planner=None, feedback=None, enricher=None,
        active_revision_path=None, embedding_profile_id=None,
    ) -> dict:
        started_at = time.perf_counter()
        available_connectors = dict(connectors)
        if self.enabled and memory is not None:
            available_connectors.setdefault(getattr(memory, "source", "agent_memory"), memory)
        plan = None
        coordinator = RetrievalCoordinator(available_connectors)
        availability = await coordinator.health_map()
        failed_sources: set[str] = set(coordinator.last_failed_sources)
        if self.enabled and planner is not None:
            plan = planner.plan(request.query, availability, {})

        selected_connectors = self._selected_connectors(available_connectors, plan)
        if self.enabled and memory is not None:
            selected_connectors.setdefault(getattr(memory, "source", "agent_memory"), memory)
        include_web = plan.include_web if plan is not None else request.include_web
        top_k = plan.top_k if plan is not None else request.topk
        queries = plan.queries if plan is not None else (request.query,)
        coordinator = RetrievalCoordinator(selected_connectors)
        evidence = []
        successful_sources: set[str] = set()
        for query in queries:
            evidence.extend(await coordinator.search(SearchRequest(
                query=query,
                topk=top_k,
                domain=request.domain,
                collection=request.collection,
                include_web=include_web,
                continuation_token=request.continuation_token,
            )))
            failed_sources.update(coordinator.last_failed_sources)
            successful_sources.update(coordinator.last_successful_sources)

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
            "metadata": {"failed_sources": sorted(failed_sources)},
        }

    @staticmethod
    def _selected_connectors(connectors: dict, plan) -> dict:
        if plan is None:
            return dict(connectors)
        selected_sources = set(plan.sources)
        return {
            name: connector for name, connector in connectors.items()
            if name in selected_sources or getattr(connector, "source", None) in selected_sources
        }
