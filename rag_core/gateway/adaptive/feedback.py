from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rag_core.gateway.adaptive.contracts import RoutingFeedback


class FeedbackStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self._events: list[RoutingFeedback] = []
        self._path = Path(path) if path is not None else None

    @property
    def events(self) -> tuple[RoutingFeedback, ...]:
        return tuple(self._events)

    def record(self, feedback: RoutingFeedback) -> None:
        self._events.append(feedback)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(feedback), ensure_ascii=False) + "\n")

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
