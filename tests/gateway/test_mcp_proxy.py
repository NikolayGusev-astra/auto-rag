import asyncio

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.connectors.mcp_proxy import GenericMcpConnector
from rag_core.gateway.models import EvidenceOrigin


class FakeSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {
            "content": [
                {"type": "text", "text": "Bitbucket result", "uri": "https://bitbucket.test/repo/a.py"},
                {"type": "text", "text": "Second result"},
            ]
        }

    async def list_tools(self):
        return {"tools": []}


def test_mcp_proxy_calls_tool_and_wraps_text_results_as_evidence():
    session = FakeSession()
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
