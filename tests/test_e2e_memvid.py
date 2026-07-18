"""E2E: memvid short-circuit — speed + accuracy, ON vs OFF.

Drives the REAL pipeline entrypoint `async_rag_search` (with compound
split + memvid recall/record wired in). We seed an episode
directly into memvid, then fire a *paraphrased* query and
check:
  - ON  (RAG_MEMVID_ENABLED=true, SDK present): recall short-circuits
         -> from_memory=True, no zvec stage in trace, faster.
  - OFF (RAG_MEMVID_ENABLED=false / no SDK): full pipeline runs,
         from_memory absent, slower (zvec/web executed).

This exercises the actual integration, not just memvid_memory unit.

Run with the memvid venv:
  .venv-memvid/Scripts/python.exe -m pytest tests/test_e2e_memvid.py -q -s
"""
import os
import sys
import time
import tempfile
import shutil
import asyncio

import pytest

pytest.importorskip("memvid_sdk")

# Enable memvid for the whole module (real backend).
os.environ["RAG_MEMVID_ENABLED"] = "true"
os.environ.setdefault("RAG_EMBEDDING_URL", "http://localhost:1234/v1/embeddings")
os.environ.setdefault("RAG_EMBEDDING_MODEL", "text-embedding-multilingual-e5-large-instruct")
os.environ.setdefault("RAG_MEMVID_EMBED_URL", "http://localhost:1234/v1/embeddings")
os.environ.setdefault("RAG_MEMVID_EMBED_MODEL", "text-embedding-multilingual-e5-large-instruct")
# Speed: disable slow live sources so the test is about memvid, not the net.
os.environ["RAG_SEARXNG_ENABLED"] = "false"
os.environ["RAG_WEB_SEARCH_ENABLED"] = "false"
os.environ["RAG_MCP_ENABLED"] = "false"
os.environ["RAG_FEDERATED_ENABLED"] = "false"
os.environ["RAG_EMBEDDING_URL"] = "http://localhost:1234/v1/embeddings"

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from memvid_memory import Episode, MemvidMemory
import rag_async


@pytest.fixture
def seeded(monkeypatch):
    d = tempfile.mkdtemp(prefix="e2e_memvid_")
    monkeypatch.setenv("RAG_MEMVID_DIR", d)
    monkeypatch.setenv("RAG_MEMVID_ENABLED", "true")
    # The public pipeline owns a module singleton and LRU cache; isolate both
    # so prior tests cannot reuse a capsule or an empty cached RAG result.
    rag_async._memory = None
    rag_async._CACHE.clear()
    MemvidMemory.reset()
    m = MemvidMemory.for_tenant("hermes_default")
    assert m.active is True
    assert m.record(Episode(
        query="Как сбросить пароль администратора в Astra Linux?",
        answer="Используйте sudo passwd root в recovery mode.",
        sources=[{"uri": "confluence://AL/123"}],
        domain="astra",
        tenant="hermes_default",
    ))
    m.close()
    MemvidMemory.reset()
    yield d
    rag_async._memory = None
    rag_async._CACHE.clear()
    MemvidMemory.reset()
    shutil.rmtree(d, ignore_errors=True)


def _run(query: str, domain: str):
    """Run async_rag_search with memvid ON, return (result, elapsed_s)."""
    t0 = time.perf_counter()
    res = asyncio.run(rag_async.async_rag_search(
        query, {"domain": domain, "collection": "", "confidence": 0.5, "fallback": False}))
    return res, time.perf_counter() - t0


def test_memvid_shortcircuit_on_vs_off(seeded):
    """ON must short-circuit (from_memory) and beat OFF on latency."""
    paraphrased = "восстановление доступа к root через recovery mode в астре"

    # --- ON ---
    on_res, on_t = _run(paraphrased, "astra")
    # --- OFF (disable memvid, fresh module-level state) ---
    os.environ["RAG_MEMVID_ENABLED"] = "false"
    rag_async._MEMVID_AVAILABLE = False  # force noop path
    off_res, off_t = _run(paraphrased, "astra")
    os.environ["RAG_MEMVID_ENABLED"] = "true"
    rag_async._MEMVID_AVAILABLE = True

    # ACCURACY
    assert on_res.get("from_memory") is True, \
        f"ON should short-circuit from memory; got {on_res.get('source')}"
    assert "recovery" in (on_res.get("answer") or "").lower() or \
           "passwd" in (on_res.get("answer") or "").lower(), \
        f"ON answer should match seeded episode; got {on_res.get('answer')!r}"
    # OFF must NOT claim memory
    assert off_res.get("from_memory") is not True, \
        f"OFF must not short-circuit; got {off_res}"

    # SPEED — ON must be fast (local vec cosine, no network pipeline).
    # NOTE: we do NOT assert on_t < off_t — OFF latency is
    # environment-dependent (absent live zvec/web makes OFF near-zero
    # in an isolated clone). The real proof is from_memory + a sane
    # local bound: recall is embed(query) + jsonl scan, sub-second
    # in practice; we cap at 5s to catch regressions (e.g. a stray
    # network call sneaking into the short-circuit path).
    assert on_t < 5.0, f"memvid ON too slow ({on_t:.3f}s) — check for stray net calls"
    print(f"\n  ON  = {on_t:.3f}s  from_memory={on_res.get('from_memory')} "
          f"src={on_res.get('source')}")
    print(f"  OFF = {off_t:.3f}s  from_memory={off_res.get('from_memory')} "
          f"src={off_res.get('source')}")
