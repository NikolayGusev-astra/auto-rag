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
