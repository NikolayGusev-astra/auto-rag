"""RED: local vec index must enable semantic recall over memvid-sdk 2.0.160.

memvid-sdk 2.0.160 (kind="basic") does NOT build a searchable
index from add_memory_cards without a managed embedding backend,
so find() returns [] — recall() never surfaces past episodes.
This test demands semantic recall (different phrasing, same meaning)
and FAILS on current code.

SKIPPED without memvid_sdk (production noop mode).

Run with the memvid venv:
  .venv-memvid/Scripts/python.exe -m pytest tests/test_memvid_vecidx.py -q
"""
import os
import sys
import tempfile
import shutil

import pytest

pytest.importorskip("memvid_sdk")

os.environ["RAG_MEMVID_ENABLED"] = "true"
os.environ.setdefault("RAG_MEMVID_EMBED_URL", "http://localhost:1234/v1/embeddings")
os.environ.setdefault("RAG_MEMVID_EMBED_MODEL", "text-embedding-baai-bge-m3-568m")

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from memvid_memory import Episode, MemvidMemory


@pytest.fixture
def tenant_capsule():
    d = tempfile.mkdtemp(prefix="memvid_vec_")
    os.environ["RAG_MEMVID_DIR"] = d
    yield d
    MemvidMemory.reset()
    shutil.rmtree(d, ignore_errors=True)


def test_semantic_recall_via_local_vecidx(tenant_capsule):
    """Record an episode, then recall it by MEANING (different wording).

    RED on current code: memvid-sdk find() returns [] (no local
    index in basic mode) -> recall()==[] -> assertion fails.
    GREEN after local vec index: LM Studio embeds the stored answer,
    we cosine-rank at recall time, top hit matches.
    """
    m = MemvidMemory.for_tenant("hermes_vec")
    assert m.active is True

    # stored episode: specific phrasing
    m.record(Episode(
        query="Как сбросить пароль администратора в Astra Linux?",
        answer="Используйте sudo passwd root в recovery mode.",
        domain="astra",
        tenant="hermes_vec",
    ))

    # recall with DIFFERENT phrasing, same meaning
    hits = m.recall("восстановление доступа к root через recovery", domain="astra")
    assert len(hits) >= 1, f"expected >=1 semantic hit, got {len(hits)}"
    assert hits[0].score > 0.0, f"expected score>0, got {hits[0].score}"
    assert "recovery" in hits[0].answer or "passwd" in hits[0].answer
