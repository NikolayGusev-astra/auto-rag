import os

from rag_core.gateway.models import Document, SyncBatch
from rag_core.gateway.sync.engine import SyncEngine


def test_sync_writes_staged_not_active(tmp_path):
    engine = SyncEngine(root=tmp_path)
    docs = [
        Document(
            id="jira:1",
            source="jira",
            source_instance="p",
            title="t",
            text="x",
            content_hash="h1",
        )
    ]
    batch = SyncBatch(added=docs, cursor="c1")

    revision = engine.stage_sync("jira", batch)

    assert os.path.isdir(revision.path)
    assert engine.active_revision("jira") is None


def test_tombstones_written(tmp_path):
    engine = SyncEngine(root=tmp_path)
    revision = engine.stage_sync("jira", SyncBatch(deleted=["jira:0"], cursor="c2"))

    tombstones = revision.path / "tombstones.jsonl"
    assert tombstones.exists()
    assert tombstones.read_text(encoding="utf-8").strip().splitlines() == ["jira:0"]
