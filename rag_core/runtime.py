"""RagRuntime — dependency injection container replacing module globals.

P1 fix (audit): rag_async.py held global mutable state (_EXECUTOR, _CACHE,
_ZVEC_COLLECTION, _MEMVID singleton, config). That blocks test isolation,
hot reload, multi-tenant and graceful shutdown. RagRuntime owns these and is
passed explicitly where needed; a process-wide default is provided for the
existing call sites so behaviour is unchanged in single-tenant local mode.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class RagRuntime:
    """Owns per-process RAG resources. Build one per configuration/tenant."""

    def __init__(self, *,
                 executor_workers: int = 6,
                 cache_max: int = 100,
                 tenant_id: str = "default",
                 index_revision: str = "unknown",
                 config_revision: str = "unknown",
                 model_revision: str = "unknown") -> None:
        self.executor = ThreadPoolExecutor(max_workers=executor_workers)
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self.cache_max = cache_max
        self.tenant_id = tenant_id
        self.index_revision = index_revision
        self.config_revision = config_revision
        self.model_revision = model_revision
        self._zvec = None
        self._zvec_lock = threading.Lock()
        self._memory = None

    # ── cache ──
    def cache_get(self, key: str) -> dict | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def cache_set(self, key: str, value: dict) -> None:
        self._cache[key] = value
        while len(self._cache) > self.cache_max:
            self._cache.popitem(last=False)

    # ── zvec ──
    def get_zvec(self):
        if self._zvec is not None:
            return self._zvec
        with self._zvec_lock:
            if self._zvec is not None:
                return self._zvec
            import os
            import zvec  # lazy
            from rag_core.rag_config import ZVEC_PATH, ZVEC_COLLECTION
            zpath = os.path.join(ZVEC_PATH, ZVEC_COLLECTION)
            lock_path = zpath + "/LOCK"
            try:
                with open(lock_path, "w") as f:
                    f.write(str(os.getpid()))
            except OSError:
                pass
            self._zvec = zvec.open(zpath)
            return self._zvec

    # ── memory ──
    def get_memory(self):
        return self._memory

    def set_memory(self, mem) -> None:
        self._memory = mem

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
        if self._zvec is not None:
            try:
                self._zvec.close()
            except Exception:
                pass
            self._zvec = None


# Process-wide default (single-tenant local mode). Tests may build their own.
_DEFAULT_RUNTIME: RagRuntime | None = None
_RT_LOCK = threading.Lock()


def default_runtime() -> RagRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        with _RT_LOCK:
            if _DEFAULT_RUNTIME is None:
                _DEFAULT_RUNTIME = RagRuntime()
    return _DEFAULT_RUNTIME


def set_default_runtime(rt: RagRuntime) -> None:
    global _DEFAULT_RUNTIME
    _DEFAULT_RUNTIME = rt