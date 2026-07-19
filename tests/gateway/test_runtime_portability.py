from rag_core.gateway.model_providers import EmbeddingProfile
from rag_core.gateway.model_runtime.compatibility import check_compatible
from rag_core.gateway.model_runtime.manifest import IndexManifest
from rag_core.gateway.model_runtime.providers.cpu import make_cpu_profile


def test_two_runtimes_same_model_open_index():
    index_profile = EmbeddingProfile("sentence-transformers", "m/e5", "r", 768, True, "cosine", "q-p-v1")
    cpu_profile = make_cpu_profile("m/e5", dim=768, revision="r", pre="q-p-v1")

    ok, _ = check_compatible(cpu_profile, index_profile)

    assert ok is True


def test_dim_match_diff_model_rejected():
    index_profile = EmbeddingProfile("family", "model-a", "r", 768, True, "cosine", "q-p-v1")
    other_profile = EmbeddingProfile("family", "model-b", "r", 768, True, "cosine", "q-p-v1")

    ok, reason = check_compatible(other_profile, index_profile)

    assert ok is False
    assert "model" in reason.lower()


def test_manifest_blocks_incompatible_on_load(tmp_path):
    manifest = IndexManifest(root=tmp_path)
    manifest.write(
        profile=EmbeddingProfile("f", "model-a", "r", 768, True, "cosine", "q-p-v1"),
        active_revision="rev1",
    )
    loaded = IndexManifest(root=tmp_path)
    incoming = EmbeddingProfile("f", "model-b", "r", 768, True, "cosine", "q-p-v1")

    ok, _ = check_compatible(incoming, loaded.profile)

    assert ok is False
