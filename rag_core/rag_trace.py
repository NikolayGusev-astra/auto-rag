"""
RAG Tracing — structured telemetry for every pipeline stage.

Usage:
    trace = RagTrace(query, domain, collection)
    with trace.stage("zvec_search"):
        result = search(query)
    trace.stage("zvec_search", status="ok", chunks=len(result), score=max_score)
    print(trace.json())  # full trace as JSON
"""
from __future__ import annotations

import json
import time
from typing import Any


class RagTrace:
    """Accumulates structured events across a single RAG query."""

    def __init__(self, query: str, domain: str = "", collection: str = ""):
        self.query = query[:200]
        self.domain = domain
        self.collection = collection
        self.stages: list[dict] = []
        self._stack: list[dict] = []  # active timers

    def begin(self, name: str, **meta) -> None:
        """Start a stage timer."""
        self._stack.append({"name": name, "ts": time.time(), "meta": meta})

    def end(self, name: str, status: str = "ok", **extra) -> dict:
        """End a stage timer and record it."""
        for i in range(len(self._stack) - 1, -1, -1):
            entry = self._stack[i]
            if entry["name"] == name:
                elapsed = time.time() - entry["ts"]
                event = {
                    "stage": name,
                    "duration_ms": round(elapsed * 1000),
                    "status": status,
                    **entry["meta"],
                    **extra,
                }
                self.stages.append(event)
                self._stack.pop(i)
                return event
        # If no matching timer, record with unknown duration
        event = {"stage": name, "duration_ms": -1, "status": status, **extra}
        self.stages.append(event)
        return event

    def event(self, name: str, **data) -> dict:
        """Record a point event (no duration)."""
        event = {"stage": name, "duration_ms": 0, "status": "info", **data}
        self.stages.append(event)
        return event

    def decision(self, name: str, choice: str, reason: str, **extra) -> dict:
        """Record a routing decision."""
        return self.event(name, type="decision", choice=choice, reason=reason, **extra)

    def error(self, name: str, message: str) -> dict:
        """Record an error."""
        return self.event(name, type="error", status="error", message=str(message)[:500])

    class _StageCtx:
        def __init__(self, trace: "RagTrace", name: str, **meta):
            self.trace = trace
            self.name = name
            self.meta = meta

        def __enter__(self):
            self.trace.begin(self.name, **self.meta)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                self.trace.end(self.name, status="error", error=str(exc_val)[:300])
            else:
                self.trace.end(self.name)

    def stage(self, name: str, **meta) -> _StageCtx:
        """Context manager: with trace.stage('zvec'): ..."""
        return self._StageCtx(self, name, **meta)

    @property
    def total_ms(self) -> int:
        """Total duration of all stages (ms)."""
        if not self.stages:
            return 0
        return sum(s.get("duration_ms", 0) for s in self.stages if s.get("duration_ms", 0) > 0)

    def json(self, indent: int = 2) -> str:
        """Export as JSON."""
        return json.dumps({
            "query": self.query[:200],
            "domain": self.domain,
            "collection": self.collection,
            "total_ms": self.total_ms,
            "stages": self.stages,
        }, ensure_ascii=False, indent=indent)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        parts = [f"{s['stage']}={s['duration_ms']}ms({s['status']})" for s in self.stages]
        return " → ".join(parts)