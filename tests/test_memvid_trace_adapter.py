"""Tests for RagTrace ↔ memvid_trace adapter."""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import pytest

# conftest adds rag_core to sys.path
from rag_core.memvid_memory import Episode
from rag_core.rag_trace import RagTrace


# ---------------------------------------------------------------------------
# Stub — in-memory MemvidMemory replacement, no real memvid needed
# ---------------------------------------------------------------------------
class StubMemvidMemory:
    """Fake MemvidMemory back-end for isolated adapter tests."""

    active: bool = True
    recall_threshold: float = 0.75

    def recall(
        self,
        query: str,
        *,
        domain: Optional[str] = None,
        top_k: Optional[int] = None,
        when: Optional[str] = None,
    ) -> List[Episode]:
        return [
            Episode(
                query=query,
                answer="test answer",
                domain=domain or "test",
                score=0.9,
                episode_id="test-ep-1",
            )
        ]

    def record(self, episode: Episode) -> bool:
        return True

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def stub_memory():
    return StubMemvidMemory()


@pytest.fixture
def traced(stub_memory):
    """Build a MemvidTraced wrapping the stub, then an adapter around it."""
    # Import here so test collection doesn't crash before memvid_trace_adapter exists
    from memvid_trace import MemvidTraced
    from memvid_trace_adapter import MemvidTracedAdapter

    inner = MemvidTraced(stub_memory)
    return MemvidTracedAdapter(inner)


# ===========================================================================
# Test A — recall with RagTrace object
# ===========================================================================
def test_recall_with_ragtrace(traced):
    """Adapter pushes a memvid.recall event with duration_ms onto a RagTrace."""
    trace = RagTrace("test query", domain="astra")
    eps = traced.recall("test query", domain="astra", trace=trace)

    assert len(eps) == 1
    assert eps[0].score == 0.9
    assert len(trace.stages) == 1
    assert trace.stages[0]["stage"] == "memvid.recall"
    assert "duration_ms" in trace.stages[0]
    assert trace.stages[0]["duration_ms"] >= 0
    assert "hits" in trace.stages[0]
    assert trace.stages[0]["hits"] == 1
    assert "top_score" in trace.stages[0]


# ===========================================================================
# Test B — record with RagTrace object
# ===========================================================================
def test_record_with_ragtrace(traced):
    """Adapter pushes a memvid.record event with duration_ms onto a RagTrace."""
    trace = RagTrace("test query", domain="astra")
    ep = Episode(query="q", answer="a", domain="astra")
    ok = traced.record(ep, trace=trace)

    assert ok is True
    assert len(trace.stages) == 1
    assert trace.stages[0]["stage"] == "memvid.record"
    assert "duration_ms" in trace.stages[0]
    assert trace.stages[0]["duration_ms"] >= 0
    assert "episode_id" in trace.stages[0]
    assert "commit" in trace.stages[0]
    assert trace.stages[0]["commit"] is True


# ===========================================================================
# Test C — trace=None: no exception, stages unchanged
# ===========================================================================
def test_trace_none(traced):
    """trace=None must not raise and must leave stages untouched."""
    trace = RagTrace("test query", domain="astra")
    eps = traced.recall("test query", trace=None)
    assert len(eps) == 1
    assert len(trace.stages) == 0  # untouched because we passed None

    ep = Episode(query="q", answer="a", domain="astra")
    ok = traced.record(ep, trace=None)
    assert ok is True
    assert len(trace.stages) == 0  # still untouched


# ===========================================================================
# Test D — dict path still works (legacy)
# ===========================================================================
def test_dict_path_preserved(traced):
    """Passing a plain dict uses the existing _push_stage (latency_ms) path."""
    trace: Dict[str, Any] = {"stages": []}
    eps = traced.recall("test query", domain="astra", trace=trace)

    assert len(eps) == 1
    assert len(trace["stages"]) == 1
    assert trace["stages"][0]["stage"] == "memvid.recall"
    assert "latency_ms" in trace["stages"][0]
    # dict path also writes total_latency_ms
    assert "total_latency_ms" in trace
    assert trace["total_latency_ms"] >= 0


# ===========================================================================
# Test — recall_as_context works
# ===========================================================================
def test_recall_as_context(traced):
    """recall_as_context returns non-empty string and still works."""
    ctx = traced.recall_as_context("test query", domain="astra", max_chars=500)
    assert isinstance(ctx, str)
    assert "test answer" in ctx
    assert "[PRIOR EPISODES" in ctx