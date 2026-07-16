"""Native single-file vector index contract for memvid-sdk 2.x.

The SDK supports vectors inside .mv2 through put_many(..., embeddings=...).
A .vecidx.jsonl sidecar is a legacy workaround and must not be required.
"""
import os
import shutil
import tempfile

import pytest

pytest.importorskip("memvid_sdk")

from memvid_memory import Episode, MemvidMemory


@pytest.fixture
def native_tenant(monkeypatch):
    directory = tempfile.mkdtemp(prefix="memvid_native_")
    monkeypatch.setenv("RAG_MEMVID_ENABLED", "true")
    monkeypatch.setenv("RAG_MEMVID_MODE", "both")
    monkeypatch.setenv("RAG_MEMVID_DIR", directory)
    monkeypatch.setenv("RAG_MEMVID_EMBED_URL", "http://127.0.0.1:1234/v1/embeddings")
    monkeypatch.setenv("RAG_MEMVID_EMBED_MODEL", "bge-m3")
    MemvidMemory.reset()
    yield directory
    MemvidMemory.reset()
    shutil.rmtree(directory, ignore_errors=True)


def test_native_mv2_semantic_recall_survives_reopen(native_tenant):
    """Native MV2 stores vectors internally; no JSONL sidecar is created."""
    tenant = "native_contract"
    memory = MemvidMemory.for_tenant(tenant)
    assert memory.active is True
    assert memory.record(Episode(
        query="Как настроить репликацию PostgreSQL?",
        answer="Настройте primary и standby с потоковой передачей WAL.",
        domain="database",
        tenant=tenant,
    )) is True
    memory.close()
    MemvidMemory.reset()

    capsule = os.path.join(native_tenant, f"memory_{tenant}.mv2")
    assert os.path.exists(capsule)
    assert not os.path.exists(capsule + ".vecidx.jsonl")

    reopened = MemvidMemory.for_tenant(tenant)
    hits = reopened.recall("как сделать primary и replica базу данных", domain="database")
    assert hits
    assert hits[0].score > 0
    assert "WAL" in hits[0].answer
