import pytest

from rag_core.gateway.model_runtime.providers.openai_compat import (
    OpenAICompatibleEmbeddingProvider,
)


@pytest.mark.asyncio
async def test_openai_compat_embed_uses_base_url(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"embedding": [0.2, 0.3]}]}

    class FakeClient:
        async def post(self, path, json):
            captured["path"] = path
            captured["json"] = json
            return FakeResponse()

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://127.0.0.1:1234/v1", model="e5", expected_dim=2
    )
    monkeypatch.setattr(provider, "_client", FakeClient())

    assert await provider.embed_query("q") == [0.2, 0.3]
    assert captured["path"] == "/embeddings"
    assert captured["json"]["input"] == ["q"]


@pytest.mark.asyncio
async def test_openai_compat_dim_mismatch_raises(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"embedding": [0.1]}]}

    class FakeClient:
        async def post(self, path, json):
            return FakeResponse()

    provider = OpenAICompatibleEmbeddingProvider(base_url="x", model="e5", expected_dim=2)
    monkeypatch.setattr(provider, "_client", FakeClient())

    with pytest.raises(ValueError):
        await provider.embed_query("q")
