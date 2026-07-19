from rag_core.gateway.model_providers import EmbeddingProfile
from rag_core.gateway.model_runtime.manifest import IndexManifest
from rag_core.gateway.model_runtime.reindex import ReindexPlanner


def test_reindex_builds_staged_and_integrity_ok(tmp_path):
    planner = ReindexPlanner(root=tmp_path)
    new_profile = EmbeddingProfile("sentence-transformers", "m/e5", "r2", 768, True, "cosine", "q-p-v1")

    revision_path = planner.build_staged(new_profile, docs=[{"id": "d1", "text": "x"}])

    assert revision_path.exists()
    assert IndexManifest(root=tmp_path).active_revision is None
    assert planner.check_integrity(revision_path) is True


def test_reindex_integrity_fails_on_corrupt(tmp_path):
    planner = ReindexPlanner(root=tmp_path)
    profile = EmbeddingProfile("sentence-transformers", "m/e5", "r2", 768, True, "cosine", "q-p-v1")
    revision_path = planner.build_staged(profile, docs=[{"id": "d1", "text": "x"}])
    (revision_path / "docs.jsonl").write_text("{broken", encoding="utf-8")

    assert planner.check_integrity(revision_path) is False
