from pathlib import Path


def test_mcp_server_uses_factory_when_connectors_not_supplied(monkeypatch):
    import rag_core.gateway.server as server

    class FakeMCP:
        def __init__(self, name):
            self.name = name

        def tool(self):
            return lambda function: function

    captured = {}

    def fake_build(config):
        captured["config"] = config
        return {"local_snapshot": object()}

    monkeypatch.setattr(server, "FastMCP", FakeMCP)
    monkeypatch.setattr(server, "build_connectors", fake_build)

    result = server.create_mcp_server()

    assert result.name == "auto-rag-gateway"
    assert captured["config"].knowledge_root == Path.home() / ".local" / "share" / "auto-rag"


def test_mcp_search_uses_embedding_runtime_from_environment(monkeypatch):
    import asyncio
    import rag_core.gateway.server as server

    class FakeMCP:
        def __init__(self, name):
            self.name = name
            self.tools = {}

        def tool(self):
            def register(function):
                self.tools[function.__name__] = function
                return function

            return register

    captured = {}

    class FakeProvider:
        def __init__(self, base_url, model, expected_dim):
            captured["provider"] = (base_url, model, expected_dim)

    async def fake_handle_search(request, connectors, *, enricher=None, reranker=None):
        captured["reranker"] = reranker
        return {"results": [], "trace": {}}

    monkeypatch.setattr(server, "FastMCP", FakeMCP)
    monkeypatch.setattr(server, "OpenAICompatibleEmbeddingProvider", FakeProvider)
    monkeypatch.setattr(server, "handle_search", fake_handle_search)
    monkeypatch.setenv("EMBED_URL", "http://embedding.test/v1/embeddings")
    monkeypatch.setenv("EMBED_MODEL", "test-embed")

    mcp_server = server.create_mcp_server(connectors={})
    asyncio.run(mcp_server.tools["search"]("q"))

    assert captured["provider"] == ("http://embedding.test/v1/embeddings", "test-embed", 1024)
    assert type(captured["reranker"]).__name__ == "RerankAdapter"
