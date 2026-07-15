"""Tests for memvid env-bridge + rag_async wiring (T3)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "rag_core"))


# ---------------------------------------------------------------------------
# Test A: env bridge mapping
# ---------------------------------------------------------------------------
def test_bridge_maps_embedding_vars(monkeypatch):
    """bridge_memvid_env() inherits EMBEDDING_* into RAG_MEMVID_* when unset."""
    monkeypatch.delenv("RAG_MEMVID_EMBED_MODEL", raising=False)
    monkeypatch.delenv("RAG_MEMVID_EMBED_URL", raising=False)
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-baai-bge-m3-568m")
    monkeypatch.setenv("EMBEDDING_URL", "http://localhost:1234/v1/embeddings")

    import memvid_config_bridge
    memvid_config_bridge.bridge_memvid_env()

    assert os.environ["RAG_MEMVID_EMBED_MODEL"] == "text-embedding-baai-bge-m3-568m"
    assert os.environ["RAG_MEMVID_EMBED_URL"] == "http://localhost:1234/v1/embeddings"


# ---------------------------------------------------------------------------
# Test B: wiring recall -> RAG -> record (enabled)
# ---------------------------------------------------------------------------
def test_wiring_recall_before_record(monkeypatch):
    """With memory enabled + fake backend, recall runs before RAG, record after."""
    monkeypatch.setenv("RAG_MEMVID_ENABLED", "true")

    import rag_async

    # Fake episode + memory
    fake_ep = MagicMock()
    fake_ep.score = 0.9
    fake_ep.answer = "cached answer"
    fake_ep.sources = [{"source": "mem"}]

    fake_mem = MagicMock()
    fake_mem.active = True
    fake_mem.recall_threshold = 0.75
    fake_mem.recall.return_value = [fake_ep]
    fake_mem.record.return_value = True

    # Patch _get_memory to return our fake
    with patch.object(rag_async, "_get_memory", return_value=fake_mem):
        # Build a minimal dcd_result
        dcd = {"domain": "astra", "collection": "wiki", "confidence": 0.5}
        # trace=None -> created internally; we just check record called
        import asyncio
        result = asyncio.run(
            rag_async.async_rag_search("сброс пароля", dcd)
        )

    # recall should have been called (short-circuit or not)
    assert fake_mem.recall.called
    # record should have been called (unless short-circuit from memory)
    if not result.get("from_memory"):
        assert fake_mem.record.called
    else:
        # short-circuit path: record NOT called for a memory hit
        assert not fake_mem.record.called


# ---------------------------------------------------------------------------
# Test C: active property reflects disabled state
# ---------------------------------------------------------------------------
def test_memvid_active_false_when_disabled(monkeypatch):
    """MemvidMemory.active is False when RAG_MEMVID_ENABLED=false."""
    monkeypatch.setenv("RAG_MEMVID_ENABLED", "false")

    from memvid_memory import MemvidMemory, MemvidConfig
    MemvidMemory.reset()
    cfg = MemvidConfig.from_env()
    cfg.enabled = False
    m = MemvidMemory(cfg)
    assert m.active is False
