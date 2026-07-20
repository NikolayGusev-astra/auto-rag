from unittest.mock import AsyncMock, patch

import httpx
import pytest

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.connectors.jira_connector import JiraConnector
from rag_core.gateway.models import EvidenceOrigin


@pytest.mark.asyncio
async def test_search_live_maps_jira_issues():
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://jira.example.test/rest/api/2/search"),
        json={
            "issues": [
                {
                    "key": "PROJ-123",
                    "fields": {
                        "summary": "Fix search",
                        "description": "The search is broken.",
                        "updated": "2026-07-20T12:00:00.000+0000",
                    },
                }
            ]
        },
    )
    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=response)) as get:
        result = await JiraConnector("https://jira.example.test/", "secret").search_live(
            SearchRequest(query="search text", topk=3)
        )

    assert get.await_args.kwargs == {
        "params": {"jql": "text~\"search text\"", "maxResults": 3, "fields": "summary,description,updated"}
    }
    assert result[0].document_id == "PROJ-123"
    assert result[0].title == "Fix search"
    assert result[0].text == "Fix search\nThe search is broken."
    assert result[0].uri == "https://jira.example.test/browse/PROJ-123"
    assert result[0].source == "jira"
    assert result[0].origin is EvidenceOrigin.LIVE_CORPORATE


@pytest.mark.asyncio
async def test_health_reports_unavailable_when_request_fails():
    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(side_effect=httpx.HTTPError("offline"))):
        health = await JiraConnector("https://jira.example.test", "secret").health()

    assert health == {"source": "jira", "available": False, "reason": "offline"}
