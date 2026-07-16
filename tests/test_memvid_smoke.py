"""
test_memvid_smoke.py — Noop-contract smoke tests for memvid memory layer.

Verifies that memory NEVER breaks RAG when disabled or when the SDK is
missing. These tests enforce the noop-contract:
  - recall() always returns []
  - record() never raises
  - MemvidTraced.active correctly reflects disabled state
"""

import os
import pytest

# ---------------------------------------------------------------------------
# Test A — Noop when disabled (RAG_MEMVID_ENABLED=false)
# ---------------------------------------------------------------------------
def test_noop_when_disabled(monkeypatch):
    """Memory must never raise when RAG_MEMVID_ENABLED=false.
    recall returns [], record returns a bool without exception.
    """
    monkeypatch.setenv("RAG_MEMVID_ENABLED", "false")

    from memvid_memory import Episode, MemvidMemory
    MemvidMemory.reset()

    m = MemvidMemory.for_tenant("smoke")
    assert m.recall("anything") == []
    # record must not raise; value depends on backend (noop -> False)
    assert m.record(Episode(query="q", answer="a")) in (True, False)


# ---------------------------------------------------------------------------
# Test B — Graceful degradation when SDK missing + enabled
# ---------------------------------------------------------------------------
def test_graceful_when_sdk_missing(monkeypatch):
    """When memvid-sdk is not installed but RAG_MEMVID_ENABLED=true,
    backend must fall back to _NoopMemvidBackend without crashing.
    """
    import importlib

    sdk_available = False
    try:
        import memvid_sdk  # noqa: F401
        sdk_available = True
    except ImportError:
        try:
            import memvid  # noqa: F401  (legacy fallback)
            sdk_available = True
        except ImportError:
            sdk_available = False

    if sdk_available:
        pytest.skip("memvid-sdk is installed — cannot test graceful-missing path")

    monkeypatch.setenv("RAG_MEMVID_ENABLED", "true")

    from memvid_memory import MemvidMemory, _NoopMemvidBackend
    MemvidMemory.reset()

    m = MemvidMemory.for_tenant("smoke2")
    # Backend should be the noop fallback because import failed
    assert isinstance(m._backend, _NoopMemvidBackend)
    # Double-check via recall returning []
    assert m.recall("anything") == []


# ---------------------------------------------------------------------------
# Test C — MemvidTraced importable and active reflects disabled state
# ---------------------------------------------------------------------------
def test_memvid_traced_active_disabled(monkeypatch):
    """MemvidTraced wraps MemvidMemory; its `active` property must
    correctly reflect that memory is disabled.
    """
    monkeypatch.setenv("RAG_MEMVID_ENABLED", "false")

    from memvid_memory import MemvidMemory
    from memvid_trace import MemvidTraced
    MemvidMemory.reset()

    m = MemvidTraced(MemvidMemory.for_tenant("smoke3"))
    assert m.active is False