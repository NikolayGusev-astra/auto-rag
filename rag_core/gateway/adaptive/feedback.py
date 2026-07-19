from __future__ import annotations

from rag_core.gateway.adaptive.contracts import RoutingFeedback


class FeedbackStore:
    def __init__(self) -> None:
        self._events: list[RoutingFeedback] = []

    def record(self, feedback: RoutingFeedback) -> None:
        self._events.append(feedback)

    def aggregate(self) -> dict[str, dict[str, int]]:
        stats: dict[str, dict[str, int]] = {}
        for feedback in self._events:
            for source in feedback.selected_sources:
                stats.setdefault(source, {"selected": 0, "useful": 0})["selected"] += 1
            for source in feedback.successful_sources:
                stats.setdefault(source, {"selected": 0, "useful": 0})["useful"] += 1
        return stats

    def evaluate(self, golden: list) -> dict[str, int]:
        return {"events": len(self._events), "golden_size": len(golden)}
