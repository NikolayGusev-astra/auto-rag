import os

import pytest

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


def test_publish_is_atomic_and_blocks_corrupt_staged_revision(tmp_path):
    engine = SyncEngine(root=tmp_path)
    good = SyncBatch(added=[_document("jira:1", "h1")], cursor="c1")
    revision = engine.stage_sync("jira", good)

    engine.publish("jira", revision)
    assert engine.active_revision("jira") == str(revision.path)

    bad_revision = engine.stage_sync("jira", SyncBatch(added=[_document("jira:2", "h2")], cursor="c2"))
    (bad_revision.path / "docs.jsonl").write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError):
        engine.publish("jira", bad_revision)

    assert engine.active_revision("jira") == str(revision.path)


def test_published_documents_exclude_tombstones(tmp_path):
    engine = SyncEngine(root=tmp_path)
    revision = engine.stage_sync(
        "jira",
        SyncBatch(added=[_document("jira:1", "h1"), _document("jira:0", "h0")], deleted=["jira:0"]),
    )

    engine.publish("jira", revision)

    assert [document["id"] for document in engine.active_documents("jira")] == ["jira:1"]


def test_incremental_add_carries_forward_active_documents(tmp_path):
    engine = SyncEngine(root=tmp_path)
    initial = engine.stage_sync(
        "jira",
        SyncBatch(added=[_document("jira:a", "ha"), _document("jira:b", "hb"), _document("jira:c", "hc")]),
    )
    engine.publish("jira", initial)

    incremental = engine.stage_sync("jira", SyncBatch(added=[_document("jira:d", "hd")]))
    engine.publish("jira", incremental)

    assert [document["id"] for document in engine.active_documents("jira")] == ["jira:a", "jira:b", "jira:c", "jira:d"]


def test_sync_status_returns_cursor(tmp_path):
    engine = SyncEngine(root=tmp_path)
    revision = engine.stage_sync("jira", SyncBatch(added=[_document("jira:1", "h1")], cursor="c1"))
    engine.publish("jira", revision)

    status = engine.sync_status("jira")

    assert status["cursor"] == "c1"
    assert status["available"] is True


async def test_sync_source_uses_supplied_cursor(tmp_path):
    connector = _Connector()
    engine = SyncEngine(root=tmp_path)

    await engine.sync_source(connector, cursor="resume-from-here")

    assert connector.received_cursor == "resume-from-here"
    assert engine.sync_status("jira")["cursor"] == "c2"


def _document(document_id, content_hash):
    return Document(
        id=document_id,
        source="jira",
        source_instance="p",
        title="t",
        text="x",
        content_hash=content_hash,
    )


class _Connector:
    source = "jira"

    def __init__(self):
        self.received_cursor = None

    async def sync_changes(self, cursor):
        self.received_cursor = cursor
        return SyncBatch(added=[_document("jira:2", "h2")], cursor="c2")
