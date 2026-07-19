"""Regression tests for the repeat-audit P0/P1 findings.

These specifically exercise the runtime paths that the static audit found
broken: MCP fallback NameError, compound tenant/ACL propagation, compound
ranking by calibrated score, and tenant-isolated memory.
"""
import asyncio
from unittest import mock

from rag_core import rag_async
from rag_trace import RagTrace


def _fake_loop():
    return asyncio.new_event_loop()


async def test_mcp_fallback_receives_tenant_no_nameerror():
    """P1: _fallback_to_mcp_web must accept tenant_id and not raise NameError."""
    loop = _fake_loop()
    trace = RagTrace("q", "d", "c")
    rag_async.MCP_SERVERS["context7"] = {}
    # run_in_executor spawns a thread bound to a different loop in tests;
    # patch it to call the blocking func synchronously in the test loop.
    async def fake_run_in_executor(executor, fn, *args):
        return fn(*args)
    with mock.patch.object(loop, "run_in_executor", fake_run_in_executor), \
         mock.patch.object(rag_async, "_blocking_mcp_single",
                           return_value=[{"text": "x", "score": 0.5, "source": "context7"}]):
        res = await rag_async._fallback_to_mcp_web(
            "q", "d", "c", loop, trace, tenant_id="tenant-a")
    assert res["source"] == "context7"
    assert res["chunks"][0]["_trust"] == "trusted_internal"
    # top-level score now derived from calibrated chunks, not a raw constant
    assert 0 < res["score"] <= 1


async def test_compound_preserves_tenant_and_acl():
    """P1: compound subrequests must carry the original tenant_id/acl_hash."""
    trace = RagTrace("q", "devops", "deployment")
    dcd = {"domain": "devops", "collection": "deployment", "confidence": 0.5,
           "fallback": False, "tenant_id": "tenant-X", "acl_hash": "acl-123"}

    captured = {}

    async def fake_impl(q, dcd_sub, trace=None, federate=True, max_results=5):
        captured.setdefault("seen", []).append(dcd_sub.get("tenant_id"))
        # return a dummy result so fusion runs
        return {"source": "zvec", "chunks": [{"text": "c", "score": 0.5}], "score": 0.5}

    with mock.patch.object(rag_async, "_async_rag_search_impl", side_effect=fake_impl):
        await rag_async.async_rag_search(
            "настройка ald pro postgresql репликация", dcd, trace=trace, federate=False)
    assert captured["seen"], "compound subrequests were not issued"
    assert all(t == "tenant-X" for t in captured["seen"]), captured["seen"]


async def test_compound_ranking_by_calibrated_score():
    """P1: compound fusion must sort chunks by calibrated score, not order."""
    trace = RagTrace("q", "devops", "deployment")
    dcd = {"domain": "devops", "collection": "deployment", "confidence": 0.5,
           "fallback": False, "tenant_id": "t", "acl_hash": "a"}

    async def fake_impl(q, dcd_sub, trace=None, federate=True, max_results=5):
        # First subquery (rusbitech domain) returns weak chunks,
        # second (devops domain) returns strong — but weak comes first in gather.
        if dcd_sub.get("domain") == "rusbitech":
            return {"source": "zvec", "chunks": [
                {"text": "weak1", "score": 0.30},
                {"text": "weak2", "score": 0.31},
            ], "score": 0.31}
        return {"source": "zvec", "chunks": [
            {"text": "strong1", "score": 0.90},
            {"text": "strong2", "score": 0.85},
        ], "score": 0.90}

    with mock.patch.object(rag_async, "_async_rag_search_impl", side_effect=fake_impl):
        res = await rag_async.async_rag_search(
            "настройка ald pro postgresql репликация", dcd, trace=trace, federate=False)
    fused = res["chunks"]
    # strong chunk must outrank weak despite being from the 2nd subquery
    assert fused[0]["text"] == "strong1"
    assert fused[0]["score"] == 0.90


async def test_memory_isolated_by_tenant_registry():
    """P0: _get_memory(tenant) returns tenant-distinct instances."""
    created = {}
    def fake_for_tenant(tid):
        # return a distinct object per tenant so identity checks work
        if tid not in created:
            created[tid] = object()
        return created[tid]
    with mock.patch.object(rag_async, "_MEMVID_AVAILABLE", True), \
         mock.patch.dict("sys.modules", {"memvid_config_bridge": mock.MagicMock()}), \
         mock.patch.object(rag_async, "MemvidTraced", lambda x: x), \
         mock.patch.object(rag_async, "MemvidMemory") as MM:
            MM.for_tenant.side_effect = fake_for_tenant
            a = rag_async._get_memory("tenant-a")
            b = rag_async._get_memory("tenant-b")
            assert a != b, "memory instances must differ per tenant"
            # same tenant returns same instance (registry, not re-created)
            a2 = rag_async._get_memory("tenant-a")
            assert a is a2


async def test_memory_hit_short_circuits_before_compound():
    """NEW regression: memory hit must return before compound decomposition,
    not be overridden by fresh subqueries."""
    trace = RagTrace("q", "devops", "deployment")
    dcd = {"domain": "devops", "collection": "deployment", "confidence": 0.5,
           "fallback": False, "tenant_id": "t", "acl_hash": "a"}
    compound_called = {"flag": False}

    async def fake_impl(q, dcd_sub, trace=None, federate=True, max_results=5):
        compound_called["flag"] = True
        return {"source": "zvec", "chunks": [{"text": "c", "score": 0.5}], "score": 0.5}

    # Force a memory hit: monkeypatch _get_memory to return a fake with active=True
    class _FakeMem:
        active = True
        recall_threshold = 0.0
        def recall(self, *a, **k):
            from types import SimpleNamespace
            ep = SimpleNamespace(answer="recalled answer",
                                 sources=[{"source": "wiki", "trusted": True}],
                                 score=0.95)
            return [ep]
        def record(self, *a, **k):
            pass

    with mock.patch.object(rag_async, "_MEMVID_AVAILABLE", True), \
         mock.patch.object(rag_async, "_get_memory", return_value=_FakeMem()), \
         mock.patch.object(rag_async, "_async_rag_search_impl", side_effect=fake_impl):
        res = await rag_async.async_rag_search(
            "настройка ald pro postgresql репликация", dcd, trace=trace, federate=False)
    assert res.get("from_memory") is True, "memory hit must short-circuit"
    assert compound_called["flag"] is False, "compound must NOT run on memory hit"
    assert res["score"] > 0, "memory result must carry calibrated score"
    assert res["chunks"][0]["_trust"] == "trusted_internal"


async def test_no_ctx_nameerror_in_compound_path():
    """P1.2: full compound path must not raise NameError on ctx (cache write)."""
    trace = RagTrace("q", "devops", "deployment")
    dcd = {"domain": "devops", "collection": "deployment", "confidence": 0.5,
           "fallback": False, "tenant_id": "t", "acl_hash": "a"}

    async def fake_impl(q, dcd_sub, trace=None, federate=True, max_results=5):
        return {"source": "zvec", "chunks": [{"text": "c", "score": 0.5}], "score": 0.5}

    with mock.patch.object(rag_async, "_async_rag_search_impl", side_effect=fake_impl):
        res = await rag_async.async_rag_search(
            "настройка ald pro postgresql репликация", dcd, trace=trace, federate=False)
    assert res["source"] == "compound"
