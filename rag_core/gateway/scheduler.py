"""Scheduling primitives for keeping interactive gateway work responsive."""
from __future__ import annotations

import heapq
import itertools
import os
import sys
from typing import Any


class PriorityQueue:
    """Stable lower-number-first work queue (search before sync)."""

    def __init__(self) -> None:
        self._items: list[tuple[int, int, Any]] = []
        self._sequence = itertools.count()

    def put(self, item: Any, priority: int) -> None:
        heapq.heappush(self._items, (priority, next(self._sequence), item))

    def get(self) -> Any:
        return heapq.heappop(self._items)[2]


def apply_low_cpu_priority(enabled: bool = True) -> bool:
    """Best-effort idle/low priority for background gateway processes.

    Thread priority has no portable Python API, so the gateway lowers its
    process priority instead.  Permission failures and unsupported systems are
    intentionally harmless and reported as ``False``.
    """
    if not enabled:
        return False
    try:
        if sys.platform == "win32":
            import psutil

            psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
        elif hasattr(os, "nice"):
            os.nice(19)
        else:
            return False
    except (ImportError, OSError, PermissionError):
        return False
    return True
