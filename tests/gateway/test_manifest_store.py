import json

import pytest

from rag_core.gateway.model_providers import EmbeddingProfile
from rag_core.gateway.sync.engine import SyncEngine
from rag_core.gateway.sync.manifest_store import (
    CorruptManifestError,
    MissingRevisionError,
    RevisionManifestStore,
)
from rag_core.gateway.sync.publisher import RevisionPublisher


def test_write_then_read_round_trips_single_versioned_schema(tmp_path):
    store = RevisionManifestStore(tmp_path, "jira")
    profile = {"provider_family": "test", "dimension": 3}

    store.write(profile=profile, active_revision="revision-1", cursor="cursor-1")

    assert store.read() == {
        "schema_version": 1,
        "profile": profile,
        "active_revision": "revision-1",
        "cursor": "cursor-1",
    }


def test_corrupt_manifest_raises_specific_error(tmp_path):
    store = RevisionManifestStore(tmp_path, "jira")
    store.path.parent.mkdir(parents=True)
    store.path.write_text("{broken", encoding="utf-8")

    with pytest.raises(CorruptManifestError):
        store.read()
    with pytest.raises(CorruptManifestError):
        store.active_revision()


def test_missing_manifest_is_distinct_from_corrupt_and_first_sync(tmp_path):
    store = RevisionManifestStore(tmp_path, "jira")

    assert store.read() is None
    assert store.active_revision() is None


def test_active_revision_with_missing_or_unreadable_documents_raises_specific_error(tmp_path):
    store = RevisionManifestStore(tmp_path, "jira")
    store.write(profile={}, active_revision=str(tmp_path / "missing"), cursor=None)

    with pytest.raises(MissingRevisionError):
        store.active_revision()

    revision = tmp_path / "revision"
    revision.mkdir()
    (revision / "docs.jsonl").write_text("{broken", encoding="utf-8")
    store.write(profile={}, active_revision=str(revision), cursor=None)

    with pytest.raises(MissingRevisionError):
        store.active_revision()


def test_sync_engine_and_revision_publisher_publish_through_same_schema(tmp_path):
    engine = SyncEngine(tmp_path)
    revision = engine.stage_sync("jira", _batch())
    engine.publish("jira", revision)

    sync_manifest = RevisionManifestStore(tmp_path, "jira").read()
    assert sync_manifest == {
        "schema_version": 1,
        "profile": {},
        "active_revision": str(revision.path),
        "cursor": "cursor-1",
    }

    profile = EmbeddingProfile("family", "model", "r1", 3, True, "cosine", "p1")
    published_revision = RevisionPublisher(tmp_path).build_staged(profile, [{"id": "d1", "text": "x"}])
    RevisionPublisher(tmp_path).publish(profile, published_revision)

    publisher_manifest = RevisionManifestStore(tmp_path, None).read()
    assert publisher_manifest == {
        "schema_version": 1,
        "profile": profile.__dict__,
        "active_revision": str(published_revision),
        "cursor": None,
    }


def _batch():
    from rag_core.gateway.models import Document, SyncBatch

    return SyncBatch(
        added=[Document("jira:1", "jira", "p", "title", "text", "hash")],
        cursor="cursor-1",
    )
