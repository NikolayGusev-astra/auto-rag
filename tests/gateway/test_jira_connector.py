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
        "params": {"jql": "text~\"search text\"", "maxResults": 3, "fields": "summary,description,updated,issuelinks"}
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
    empty_response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://jira.example.test/rest/api/2/issue/INT-6515/comment"),
        json={"comments": [], "total": 0},
    )
    issue_detail = httpx.Response(
        200,
        request=httpx.Request("GET", "https://jira.example.test/rest/api/2/issue/INT-6515"),
        json={
            "fields": {
                "summary": "Exact",
                "issuelinks": [
                    {
                        "type": {"name": "Связь"},
                        "outwardIssue": {
                            "key": "SIRIUS-189661",
                            "fields": {"summary": "content-sync-plugin error"},
                        },
                    }
                ],
            },
        },
    )
    linked_detail = httpx.Response(
        200,
        request=httpx.Request("GET", "https://jira.example.test/rest/api/2/issue/SIRIUS-189661"),
        json={
            "fields": {
                "summary": "content-sync-plugin error",
                "description": "DB retried operation fix.",
            },
        },
    )

    side_effect = [exact_response, text_response, empty_response, issue_detail, linked_detail]
    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(side_effect=side_effect)) as get:
        result = await JiraConnector("https://jira.example.test", "secret").search_live(
            SearchRequest(query="INT-6515", topk=2)
        )

    jql_calls = [c for c in get.await_args_list if (c.kwargs.get("params") or {}).get("jql")]
    assert [c.kwargs["params"]["jql"] for c in jql_calls] == [
        "issueKey=INT-6515",
        'text~"INT-6515"',
    ]
    assert [evidence.document_id for evidence in result] == ["INT-6515", "INT-6516"]

    # Enrichment metadata on exact match
    exact = result[0]
    enrichment = exact.metadata.get("enrichment", {})
    assert enrichment["comments_total"] == 0
    assert enrichment["comments_loaded"] == 0
    assert enrichment["comments_status"] == "ok"
    assert enrichment["linked_issues_loaded"] == 1
    assert "SIRIUS-189661" in exact.text


@pytest.mark.asyncio
async def test_comments_error_diagnostics():
    """When Jira returns 403 for comments, metadata captures the error."""
    search_resp = httpx.Response(
        200,
        request=httpx.Request("GET", "https://jira.example.test/rest/api/2/search"),
        json={"issues": [{"key": "BUG-1", "fields": {"summary": "Broken"}}]},
    )
    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(side_effect=[
        search_resp,  # issueKey search
        search_resp,  # text search
        httpx.HTTPError("403 Forbidden"),  # comments fail
    ])):
        result = await JiraConnector("https://jira.example.test", "secret").search_live(
            SearchRequest(query="BUG-1", topk=1)
        )

    assert len(result) == 1
    enrichment = result[0].metadata.get("enrichment", {})
    assert enrichment["comments_status"] == "failed"
    assert "403" in enrichment.get("comments_error", "")


@pytest.mark.asyncio
async def test_linked_issue_error_diagnostics():
    search_response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://jira.example.test/rest/api/2/search"),
        json={"issues": [{"key": "BUG-1", "fields": {"summary": "Broken"}}]},
    )
    comments_response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://jira.example.test/rest/api/2/issue/BUG-1/comment"),
        json={"comments": [], "total": 0},
    )
    issue_response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://jira.example.test/rest/api/2/issue/BUG-1"),
        json={
            "fields": {
                "issuelinks": [
                    {"outwardIssue": {"key": "BUG-2", "fields": {"summary": "Related"}}}
                ]
            }
        },
    )

    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(side_effect=[
        search_response,
        search_response,
        comments_response,
        issue_response,
        httpx.HTTPError("linked issue unavailable"),
    ])):
        result = await JiraConnector("https://jira.example.test", "secret").search_live(
            SearchRequest(query="BUG-1", topk=1)
        )

    enrichment = result[0].metadata["enrichment"]
    assert enrichment["linked_issues_status"] == "failed"
    assert "unavailable" in enrichment["linked_issues_error"]


@pytest.mark.asyncio
async def test_health_reports_unavailable_when_request_fails():
    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(side_effect=httpx.HTTPError("offline"))):
        health = await JiraConnector("https://jira.example.test", "secret").health()

    assert health == {"source": "jira", "available": False, "reason": "offline"}
