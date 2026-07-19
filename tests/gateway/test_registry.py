from rag_core.gateway.model_runtime.registry import ProviderRegistry, RuntimeCapabilities


def test_negotiate_reports_lexical_without_embeddings():
    registry = ProviderRegistry(
        embeddings=None,
        lexical=True,
        reranking=False,
        query_rewrite=False,
        generation=False,
        offline=True,
    )

    capabilities = registry.negotiate()

    assert capabilities.lexical_search is True
    assert capabilities.embeddings is False
    assert capabilities.generation is False


def test_minimal_profile_allows_retrieval():
    capabilities = RuntimeCapabilities(
        embeddings=False,
        lexical_search=True,
        reranking=False,
        query_rewrite=False,
        generation=False,
        offline=True,
    )

    assert capabilities.lexical_search
