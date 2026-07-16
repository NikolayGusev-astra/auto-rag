import json
import os
import shutil
import tempfile

import pytest

pytest.importorskip("memvid_sdk")

from memvid_memory import Episode, MemvidMemory, MemvidConfig, _Embedder


def test_legacy_sidecar_migrates_to_native_mv2(monkeypatch):
    """Legacy JSONL is atomically migrated to a one-file native MV2 index."""
    directory = tempfile.mkdtemp(prefix="memvid_migrate_")
    tenant = "legacy"
    try:
        monkeypatch.setenv("RAG_MEMVID_ENABLED", "true")
        monkeypatch.setenv("RAG_MEMVID_MODE", "both")
        monkeypatch.setenv("RAG_MEMVID_DIR", directory)
        monkeypatch.setenv("RAG_MEMVID_EMBED_URL", "http://127.0.0.1:1234/v1/embeddings")
        monkeypatch.setenv("RAG_MEMVID_EMBED_MODEL", "bge-m3")

        cfg = MemvidConfig.from_env()
        cfg.tenant = tenant
        cfg.dir = __import__("pathlib").Path(directory)
        capsule = cfg.capsule_path
        sidecar = capsule.with_suffix(capsule.suffix + ".vecidx.jsonl")
        capsule.write_bytes(b"legacy-placeholder")

        ep = Episode(
            query="Как настроить PostgreSQL репликацию?",
            answer="Используйте primary и standby с WAL streaming.",
            domain="database",
            tenant=tenant,
        )
        vector = _Embedder(cfg).embed(ep.answer)
        sidecar.write_text(json.dumps({
            "entity": ep.episode_id,
            "vec": vector,
            "payload": json.loads(ep.to_payload()),
        }, ensure_ascii=False) + "\n", encoding="utf-8")

        MemvidMemory.reset()
        memory = MemvidMemory.for_tenant(tenant)
        assert memory.active
        assert capsule.exists()
        assert not sidecar.exists()
        assert capsule.with_suffix(capsule.suffix + ".legacy.bak").exists()
        assert sidecar.with_suffix(sidecar.suffix + ".legacy.bak").exists()

        hits = memory.recall("как развернуть primary и replica postgres", domain="database")
        assert hits
        assert "WAL" in hits[0].answer
    finally:
        MemvidMemory.reset()
        shutil.rmtree(directory, ignore_errors=True)
