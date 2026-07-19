from rag_core.gateway.model_providers import EmbeddingProfile
from rag_core.gateway.model_runtime.compatibility import check_compatible


def _profile(model_id, dim=768, revision="r", norm=True, metric="cosine", pre="q-p-v1"):
    return EmbeddingProfile("family", model_id, revision, dim, norm, metric, pre)


def test_equal_dim_diff_model_blocked():
    ok, reason = check_compatible(_profile("model-a"), _profile("model-b"))
    assert ok is False
    assert "model" in reason.lower()


def test_same_model_revision_compatible():
    ok, _ = check_compatible(_profile("model-a", revision="r1"), _profile("model-a", revision="r1"))
    assert ok is True


def test_diff_normalization_blocked():
    ok, _ = check_compatible(_profile("m", norm=True), _profile("m", norm=False))
    assert ok is False


def test_diff_preprocessing_blocked():
    ok, _ = check_compatible(_profile("m", pre="v1"), _profile("m", pre="v2"))
    assert ok is False
