"""Small, in-process latency recorder for gateway request stages."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class StageTimer:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started: dict[str, float] = {}
        self._stages: dict[str, dict[str, Any]] = {}

    def start(self, stage: str) -> None:
        self._started[stage] = self._clock()

    def stop(self, stage: str, *, status: str = "completed") -> None:
        started = self._started.pop(stage)
        self._stages[stage] = {
            "status": status,
            "duration_ms": round((self._clock() - started) * 1000, 3),
        }

    def skip(self, stage: str, *, reason: str) -> None:
        self._started.pop(stage, None)
        self._stages[stage] = {"status": "skipped", "reason": reason}

    def summary(self) -> dict[str, dict[str, Any]]:
        return dict(self._stages)
