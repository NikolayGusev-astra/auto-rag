import pytest

from rag_core.gateway.model_providers import EmbeddingProfile
from rag_core.gateway.model_runtime.manifest import IndexManifest
from rag_core.gateway.model_runtime.reindex import ReindexPlanner
from rag_core.gateway.sync.publisher import RevisionPublisher


def test_publish_verified_staged_revision(tmp_path):
    planner = ReindexPlanner(root=tmp_path)
    profile = _profile()
    revision = planner.build_staged(profile, docs=[{"id": "d1", "text": "x"}])
    assert planner.check_integrity(revision) is True

    publisher = RevisionPublisher(root=tmp_path)
    publisher.publish(profile, revision)

    manifest = IndexManifest(root=tmp_path)
    assert manifest.profile == profile
    assert manifest.active_revision == str(revision)


def test_publish_rejects_unverified_revision(tmp_path):
    planner = ReindexPlanner(root=tmp_path)
    profile = _profile()
    revision = planner.build_staged(profile, docs=[{"id": "d1", "text": "x"}])
    (revision / "docs.jsonl").write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity"):
        RevisionPublisher(root=tmp_path).publish(profile, revision)


def test_build_staged_delegates_to_reindex_planner(tmp_path):
    publisher = RevisionPublisher(root=tmp_path)

    revision = publisher.build_staged(_profile(), docs=[{"id": "d1", "text": "x"}])

    assert (revision / "docs.jsonl").exists()


def _profile():
    return EmbeddingProfile("sentence-transformers", "m/e5", "r2", 768, True, "cosine", "q-p-v1")
