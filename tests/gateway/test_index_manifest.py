from rag_core.gateway.model_providers import EmbeddingProfile
from rag_core.gateway.model_runtime.manifest import IndexManifest


def test_manifest_roundtrip_embedding_profile(tmp_path):
    profile = EmbeddingProfile(
        provider_family="sentence-transformers",
        model_id="intfloat/multilingual-e5-base",
        model_revision="abc123",
        dimension=768,
        normalized=True,
        distance_metric="cosine",
        preprocessing_revision="q-p-v1",
    )
    manifest = IndexManifest(root=tmp_path)
    manifest.write(profile=profile, active_revision="rev-0001")

    loaded = IndexManifest(root=tmp_path)

    assert loaded.profile == profile
    assert loaded.active_revision == "rev-0001"


def test_manifest_equality_by_fields():
    first = EmbeddingProfile("sentence-transformers", "m", "r", 768, True, "cosine", "q-p-v1")
    second = EmbeddingProfile("sentence-transformers", "m", "r", 768, True, "cosine", "q-p-v1")

    assert first == second
