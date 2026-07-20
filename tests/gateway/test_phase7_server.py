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

    async def fake_handle_search(request, connectors, *, enricher=None, reranker=None):
        captured["request"] = request
        captured["reranker"] = reranker
        return {"results": [{"text": "matching evidence"}], "trace": {}}

    monkeypatch.setattr(server, "FastMCP", FakeMCP)
    monkeypatch.setattr(server, "handle_search", fake_handle_search)
    monkeypatch.setenv("EMBED_URL", "http://embedding.test/v1/embeddings")
    monkeypatch.setenv("EMBED_MODEL", "test-embed")

    mcp_server = server.create_mcp_server(connectors={})
    result = asyncio.run(mcp_server.tools["search"]("q", top_k=3))

    assert type(captured["reranker"]).__name__ == "RerankAdapter"
    assert captured["request"].query == "q"
    assert captured["request"].topk == 3
    assert result["results"] == [{"text": "matching evidence"}]
