from __future__ import annotations

import inspect

from rag_core.gateway.adaptive.contracts import RoutingFeedback
from rag_core.gateway.coordinator import RetrievalCoordinator
from rag_core.gateway.models import Evidence, EvidenceOrigin


class AdaptiveLoop:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._coordinator = RetrievalCoordinator()

    async def run(
        self, request, connectors: dict, *, memory=None, planner=None, feedback=None, enricher=None
    ) -> dict:
        plan = None
        if self.enabled and planner is not None:
            availability = {name: True for name in connectors}
            plan = planner.plan(request.query, availability, {})

        evidence: list[Evidence] = []
        for connector in connectors.values():
            try:
                evidence.extend(self._as_evidence(await connector.search_live(request), connector))
            except Exception:
                continue

        if self.enabled and memory is not None:
            try:
                hits = memory.search_live(request)
                if inspect.isawaitable(hits):
                    hits = await hits
                evidence.extend(self._as_evidence(hits, memory))
            except Exception:
                pass

        fused = self._coordinator.fuse(evidence)
        if self.enabled and feedback is not None:
            feedback.record(RoutingFeedback(
                query=request.query,
                plan_id=request.query if plan is None else plan.original_query,
                selected_sources=tuple(connectors),
                successful_sources=tuple(sorted({item.source for item in fused})),
                useful_document_ids=(), result_count=len(fused), latency_ms=0,
            ))
        if self.enabled and enricher is not None:
            enricher.build_episode(request.query, fused)
        return {
            "results": [item.__dict__ for item in fused],
            "mode": "adaptive" if self.enabled else "reference",
        }

    @staticmethod
    def _as_evidence(items, connector) -> list[Evidence]:
        converted = []
        for index, item in enumerate(items):
            if isinstance(item, Evidence):
                converted.append(item)
                continue
            source = getattr(connector, "source", "local")
            converted.append(Evidence(
                id=item.get("id", f"{source}:{index}"),
                document_id=item.get("document_id", f"{source}:{index}"),
                title=item.get("title", ""), text=item.get("text", ""), source=source,
                uri=item.get("uri"),
                origin=EvidenceOrigin.AGENT_MEMORY if source == "agent_memory" else EvidenceOrigin.LOCAL_SNAPSHOT,
                retrieval_score=float(item.get("score", item.get("retrieval_score", 0.0))),
                metadata=item.get("metadata", {}),
            ))
        return converted
