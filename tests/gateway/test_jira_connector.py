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
async def test_search_live_merges_exact_issue_key_and_text_results():
    exact_response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://jira.example.test/rest/api/2/search"),
        json={"issues": [{"key": "INT-6515", "fields": {"summary": "Exact"}}]},
    )
    text_response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://jira.example.test/rest/api/2/search"),
        json={
            "issues": [
                {"key": "INT-6515", "fields": {"summary": "Duplicate"}},
                {"key": "INT-6516", "fields": {"summary": "Related"}},
            ]
        },
    )
    with patch.object(
        httpx.AsyncClient, "get", new=AsyncMock(side_effect=[exact_response, text_response])
    ) as get:
        result = await JiraConnector("https://jira.example.test", "secret").search_live(
            SearchRequest(query="INT-6515", topk=2)
        )

    assert [call.kwargs["params"]["jql"] for call in get.await_args_list] == [
        "issueKey=INT-6515",
        'text~"INT-6515"',
    ]
    assert [evidence.document_id for evidence in result] == ["INT-6515", "INT-6516"]


@pytest.mark.asyncio
async def test_health_reports_unavailable_when_request_fails():
    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(side_effect=httpx.HTTPError("offline"))):
        health = await JiraConnector("https://jira.example.test", "secret").health()

    assert health == {"source": "jira", "available": False, "reason": "offline"}
