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


def test_incremental_change_replaces_active_document(tmp_path):
    engine = SyncEngine(root=tmp_path)
    _publish(engine, "jira", added=[_document("jira:a", "v1"), _document("jira:b", "hb")])

    _publish(engine, "jira", changed=[_document("jira:a", "v2")])

    assert _active_hashes(engine, "jira") == {"jira:a": "v2", "jira:b": "hb"}


def test_incremental_delete_removes_active_document(tmp_path):
    engine = SyncEngine(root=tmp_path)
    _publish(
        engine,
        "jira",
        added=[_document("jira:a", "ha"), _document("jira:b", "hb"), _document("jira:c", "hc")],
    )

    _publish(engine, "jira", deleted=["jira:b"])

    assert _active_hashes(engine, "jira") == {"jira:a": "ha", "jira:c": "hc"}


def test_incremental_change_add_and_delete_merge_in_order(tmp_path):
    engine = SyncEngine(root=tmp_path)
    _publish(engine, "jira", added=[_document("jira:a", "old"), _document("jira:b", "hb")])

    _publish(
        engine,
        "jira",
        added=[_document("jira:c", "hc")],
        changed=[_document("jira:a", "new")],
        deleted=["jira:b"],
    )

    assert _active_hashes(engine, "jira") == {"jira:a": "new", "jira:c": "hc"}


def test_incremental_conflicts_are_ordered_and_idempotent(tmp_path):
    engine = SyncEngine(root=tmp_path)
    _publish(engine, "jira", added=[_document("jira:a", "old"), _document("jira:b", "hb")])
    batch = SyncBatch(
        added=[_document("jira:a", "added"), _document("jira:c", "hc")],
        changed=[_document("jira:a", "changed"), _document("jira:b", "changed-b")],
        deleted=["jira:b", "jira:unknown"],
    )

    _publish_batch(engine, "jira", batch)
    _publish_batch(engine, "jira", batch)

    assert _active_hashes(engine, "jira") == {"jira:a": "changed", "jira:c": "hc"}


def test_empty_batch_preserves_active_documents(tmp_path):
    engine = SyncEngine(root=tmp_path)
    _publish(engine, "jira", added=[_document("jira:a", "ha")])

    _publish_batch(engine, "jira", SyncBatch())

    assert _active_hashes(engine, "jira") == {"jira:a": "ha"}


def test_stage_sync_skips_corrupt_previous_revision(tmp_path):
    engine = SyncEngine(root=tmp_path)
    revision = engine.stage_sync("jira", SyncBatch(added=[_document("jira:a", "ha")]))
    engine.publish("jira", revision)
    (revision.path / "docs.jsonl").write_text("{broken", encoding="utf-8")

    replacement = engine.stage_sync("jira", SyncBatch(added=[_document("jira:b", "hb")]))
    engine.publish("jira", replacement)

    assert _active_hashes(engine, "jira") == {"jira:b": "hb"}


def test_sources_have_independent_active_revisions(tmp_path):
    engine = SyncEngine(root=tmp_path)
    _publish(engine, "jira", added=[_document("jira:a", "old")])
    _publish(engine, "wiki", added=[_document("wiki:a", "wiki")])

    _publish(engine, "jira", changed=[_document("jira:a", "new")])

    assert _active_hashes(engine, "jira") == {"jira:a": "new"}
    assert _active_hashes(engine, "wiki") == {"wiki:a": "wiki"}


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


def _publish(engine, source, *, added=(), changed=(), deleted=()):
    _publish_batch(engine, source, SyncBatch(added=list(added), changed=list(changed), deleted=list(deleted)))


def _publish_batch(engine, source, batch):
    engine.publish(source, engine.stage_sync(source, batch))


def _active_hashes(engine, source):
    return {document["id"]: document["content_hash"] for document in engine.active_documents(source)}


class _Connector:
    source = "jira"

    def __init__(self):
        self.received_cursor = None

    async def sync_changes(self, cursor):
        self.received_cursor = cursor
        return SyncBatch(added=[_document("jira:2", "h2")], cursor="c2")
