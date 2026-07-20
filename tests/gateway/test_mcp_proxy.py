import asyncio

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.config_schema import GatewayConfig, SourceConfig
from rag_core.gateway.connector_factory import (
    build_connectors,
    discover_hermes_mcp_tools,
    register_mcp_session_factory,
)
from rag_core.gateway.connectors.mcp_proxy import GenericMcpConnector
from rag_core.gateway.models import EvidenceOrigin


class FakeSession:
    def __init__(self, tool_names=()):
        self.calls = []
        self.tool_names = tool_names

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {
            "content": [
                {"type": "text", "text": "Bitbucket result", "uri": "https://bitbucket.test/repo/a.py"},
                {"type": "text", "text": "Second result"},
            ]
        }

    async def list_tools(self):
        return {"tools": [{"name": name} for name in self.tool_names]}


def test_mcp_proxy_calls_tool_and_wraps_text_results_as_evidence():
    session = FakeSession(["bitbucket_search_code", "bitbucket_search_prs"])
    connector = GenericMcpConnector("bitbucket_search_code", "bitbucket", lambda: session)

    results = asyncio.run(connector.search_live(SearchRequest(query="find auth", topk=2)))

    assert session.calls == [("bitbucket_search_code", {"query": "find auth", "topk": 2})]
    assert [item.text for item in results] == ["Bitbucket result", "Second result"]
    assert results[0].source == "mcp:bitbucket"
    assert results[0].origin is EvidenceOrigin.LIVE_CORPORATE
    assert results[0].uri == "https://bitbucket.test/repo/a.py"


def test_mcp_proxy_health_checks_the_session():
    connector = GenericMcpConnector("bitbucket_search_code", "bitbucket", FakeSession)

    assert asyncio.run(connector.health()) is True


def test_factory_builds_mcp_proxy_and_auto_discovers_hermes_tools():
    session = FakeSession(["bitbucket_search_code", "bitbucket_search_prs"])
    register_mcp_session_factory("bitbucket", lambda: session)
    config = GatewayConfig(
        local_snapshot=False,
        sources={
            "bitbucket": SourceConfig(
                name="bitbucket",
                kind="mcp-proxy",
                extra={"tool": "bitbucket_search_code", "server": "bitbucket"},
            )
        },
    )

    configured = build_connectors(config)
    discovered = asyncio.run(discover_hermes_mcp_tools("bitbucket", lambda: session))

    assert isinstance(configured["bitbucket"], GenericMcpConnector)
    assert set(discovered) == {
        "mcp:bitbucket:bitbucket_search_code",
        "mcp:bitbucket:bitbucket_search_prs",
    }
