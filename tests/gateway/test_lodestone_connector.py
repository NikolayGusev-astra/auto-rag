from unittest.mock import AsyncMock, patch

import httpx
import pytest

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.connectors.lodestone_connector import LodestoneConnector


@pytest.mark.asyncio
async def test_no_token_returns_empty():
    c = LodestoneConnector(token="", endpoint="https://lodestone.example.test/mcp/")
    result = await c.search_live(SearchRequest(query="test", topk=3))
    assert result == []


@pytest.mark.asyncio
async def test_health_no_token():
    c = LodestoneConnector(token="")
    health = await c.health()
    assert health == {"source": "lodestone", "available": False, "reason": "no token"}


@pytest.mark.asyncio
async def test_init_failure_returns_empty():
    c = LodestoneConnector(token="fake-token", endpoint="https://lodestone.example.test/mcp/")
    with patch.object(httpx.AsyncClient, "post", new=AsyncMock(
        side_effect=httpx.ConnectError("connection refused")
    )):
        result = await c.search_live(SearchRequest(query="test", topk=3))
    assert result == []


@pytest.mark.asyncio
async def test_empty_response_returns_empty():
    """No results from lodestone = empty evidence list."""
    init_resp = httpx.Response(200, headers={"mcp-session-id": "sid-123"})
    call_resp = httpx.Response(
        200, content=b'data: {"jsonrpc":"2.0","id":2,"result":{"content":[{"text":""}]}}'
    )
    c = LodestoneConnector(token="fake-token", endpoint="https://lodestone.example.test/mcp/")
    with patch.object(httpx.AsyncClient, "post", new=AsyncMock(side_effect=[init_resp, call_resp, call_resp])):
        result = await c.search_live(SearchRequest(query="test", topk=3))
    assert result == []


@pytest.mark.asyncio
async def test_structured_parsing():
    """Multi-result lodestone output produces one Evidence per result block."""
    raw = """## Lodestone Search Results

**Query:** test
**Documents returned:** 2

---

### Result 1 — source_id: tech-wiki — score: 0.85

**Title:** Replication conflicts resolution
**Source URL (cite this link in your answer):** https://wiki.astralinux.ru/x/abc

This is the body of result 1.

---

### Result 2 — source_id: aa-docs — score: 0.70

**Title:** ALD Pro upgrade guide
**Source URL (cite this link in your answer):** https://docs.astra-automation.ru/aldpro/upgrade

This is the body of result 2.

"""
    c = LodestoneConnector(token="fake-token")
    results = c._parse_structured(raw, "test query")
    assert len(results) == 2
    assert results[0].title == "Replication conflicts resolution"
    assert results[0].uri == "https://wiki.astralinux.ru/x/abc"
    assert results[0].metadata["lodestone_source_id"] == "tech-wiki"
    assert results[0].metadata["lodestone_score"] == 0.85
    assert results[1].title == "ALD Pro upgrade guide"
    assert results[1].uri == "https://docs.astra-automation.ru/aldpro/upgrade"


@pytest.mark.asyncio
async def test_stable_id():
    """Same query always produces same document_id."""
    c = LodestoneConnector(token="fake-token")
    r1 = c._parse_structured("## Lodestone Search Results\n", "SIRIUS-195479")
    r2 = c._parse_structured("## Lodestone Search Results\n", "SIRIUS-195479")
    assert r1[0].document_id == r2[0].document_id
