from unittest.mock import AsyncMock, patch

import httpx
import pytest

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.connectors.confluence_connector import ConfluenceConnector
from rag_core.gateway.models import EvidenceOrigin


@pytest.mark.asyncio
async def test_search_live_maps_confluence_storage_body_to_text():
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://wiki.example.test/rest/api/content/search"),
        json={
            "results": [
                {
                    "id": "42",
                    "title": "Runbook",
                    "body": {"storage": {"value": "<h1>Runbook</h1><p>Restart the <strong>gateway</strong>.</p>"}},
                }
            ]
        },
    )
    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=response)) as get:
        result = await ConfluenceConnector("https://wiki.example.test/", "secret").search_live(
            SearchRequest(query="restart gateway", topk=2)
        )

    assert get.await_args.kwargs == {
        "params": {"cql": "text~\"restart gateway\"", "limit": 2, "expand": "body.storage"}
    }
    assert result[0].document_id == "42"
    assert result[0].title == "Runbook"
    assert result[0].text == "Runbook Restart the gateway."
    assert result[0].uri == "https://wiki.example.test/pages/viewpage.action?pageId=42"
    assert result[0].origin is EvidenceOrigin.LIVE_CORPORATE


@pytest.mark.asyncio
async def test_health_reports_available_after_successful_request():
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://wiki.example.test/rest/api/content"),
        json={"results": []},
    )
    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=response)):
        health = await ConfluenceConnector("https://wiki.example.test", "secret").health()

    assert health == {"source": "confluence", "available": True}
