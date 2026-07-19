import pytest

from rag_core.gateway.model_runtime.providers.cpu import (
    SentenceTransformersEmbeddingProvider,
    make_cpu_profile,
)


def test_cpu_profile_dimension_declared():
    profile = make_cpu_profile("intfloat/multilingual-e5-base", dim=768)
    assert profile.dimension == 768
    assert profile.normalized is True


@pytest.mark.asyncio
async def test_cpu_provider_embed_returns_vectors(monkeypatch):
    class FakeModel:
        def encode(self, texts, **kwargs):
            return [[0.1] * 4 for _ in texts]

    provider = SentenceTransformersEmbeddingProvider(model_id="fake/e5", dim=4)
    monkeypatch.setattr(provider, "_model", FakeModel())

    vector = await provider.embed_query("привет")
    assert len(vector) == 4
    documents = await provider.embed_documents(["a", "b"])
    assert len(documents) == 2
    assert len(documents[0]) == 4
