from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rag_core.gateway.adaptive.contracts import RoutingFeedback


class FeedbackStore:
    def __init__(self, persist_path: Path | None = None) -> None:
        self._path = Path(persist_path) if persist_path is not None else None
        self._events: list[RoutingFeedback] = []
        if self._path is not None:
            self._load()

    @property
    def events(self) -> tuple[RoutingFeedback, ...]:
        return tuple(self._events)

    @property
    def path(self) -> Path | None:
        return self._path

    def record(self, event: RoutingFeedback) -> None:
        self._events.append(event)
        if self._path is not None:
            self._append_jsonl(event)

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                for field in (
                    "selected_sources",
                    "successful_sources",
                    "useful_document_ids",
                ):
                    data[field] = tuple(data[field])
                self._events.append(RoutingFeedback(**data))

    def _append_jsonl(self, event: RoutingFeedback) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def aggregate(self) -> dict:
        selected: dict[str, int] = {}
        successful: dict[str, int] = {}
        legacy_stats: dict[str, dict[str, int]] = {}
        for event in self._events:
            for source in event.selected_sources:
                selected[source] = selected.get(source, 0) + 1
                legacy_stats.setdefault(source, {"selected": 0, "useful": 0})["selected"] += 1
            for source in event.successful_sources:
                successful[source] = successful.get(source, 0) + 1
                legacy_stats.setdefault(source, {"selected": 0, "useful": 0})["useful"] += 1
        total = len(self._events)
        successes = sum(
            event.explicit_success is True
            or (event.explicit_success is None and bool(event.successful_sources))
            for event in self._events
        )
        return {
            "total": total,
            "by_source": selected,
            "selected_sources": selected,
            "successful_sources": successful,
            "success_rate": successes / total if total else 0.0,
            "avg_latency_ms": sum(event.latency_ms for event in self._events) / total if total else 0.0,
            **legacy_stats,
        }

    def evaluate(self, golden: dict | list | None = None) -> dict:
        if isinstance(golden, list):
            return {"events": len(self._events), "golden_size": len(golden)}
        aggregate = self.aggregate()
        golden = golden or {}
        forbidden = set(golden.get("forbidden_sources", ()))
        canary = bool(forbidden.intersection(aggregate["selected_sources"]))
        expected = golden.get("expected_sources", golden.get("selected_sources"))
        matches_golden = expected is None or aggregate["selected_sources"] == expected
        return {
            "candidate_policy": (
                "canary" if canary else "eligible" if matches_golden else "review"
            ),
            "canary": canary,
            "events": len(self._events),
        }
