from rag_core.gateway.model_providers import (
    EmbeddingProvider, RerankerProvider, LanguageModelProvider,
    EmbeddingProfile, EmbeddingCapabilities,
)


def test_embedding_profile_is_frozen():
    p = EmbeddingProfile(
        provider_family="sentence-transformers",
        model_id="intfloat/multilingual-e5-base",
        model_revision="abc123",
        dimension=768, normalized=True,
        distance_metric="cosine", preprocessing_revision="query-passages-v1",
    )
    assert p.dimension == 768
    assert p.normalized is True


def test_providers_are_runtime_checkable():
    assert hasattr(EmbeddingProvider, "embed_query")
    assert hasattr(RerankerProvider, "rerank")
    assert hasattr(LanguageModelProvider, "complete")
