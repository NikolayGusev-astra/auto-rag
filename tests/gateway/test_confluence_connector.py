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

    cql_calls = [c for c in get.await_args_list if "params" in c.kwargs and "cql" in c.kwargs["params"]]
    assert [c.kwargs["params"]["cql"] for c in cql_calls] == [
        'title~"restart gateway"',
        'text~"restart gateway"',
    ]
    assert result[0].document_id == "42"
    assert result[0].title == "Runbook"
    assert result[0].text == "Runbook Restart the gateway."
    assert result[0].uri == "https://wiki.example.test/pages/viewpage.action?pageId=42"
    assert result[0].origin is EvidenceOrigin.LIVE_CORPORATE
    assert result[0].metadata["content_status"] == "body"


@pytest.mark.asyncio
async def test_search_live_merges_exact_page_id_and_text_results():
    exact_response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://wiki.example.test/rest/api/content/search"),
        json={"results": [{"id": "123456", "title": "Exact"}]},
    )
    text_response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://wiki.example.test/rest/api/content/search"),
        json={
            "results": [
                {"id": "123456", "title": "Duplicate"},
                {"id": "123457", "title": "Related"},
            ]
        },
    )
    empty_attachments = httpx.Response(
        200,
        request=httpx.Request("GET", "https://wiki.example.test/rest/api/content/123456/child/attachment"),
        json={"results": []},
    )

    side_effect = [exact_response, text_response, empty_attachments, empty_attachments]
    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(side_effect=side_effect)) as get:
        result = await ConfluenceConnector("https://wiki.example.test", "secret").search_live(
            SearchRequest(query="123456", topk=2)
        )

    cql_calls = [c for c in get.await_args_list if (c.kwargs.get("params") or {}).get("cql")]
    assert [c.kwargs["params"]["cql"] for c in cql_calls] == [
        "id=123456",
        'text~"123456"',
    ]
    assert [evidence.document_id for evidence in result] == ["123456", "123457"]
    assert result[0].metadata["content_status"] == "no_pdf"
    assert result[1].metadata["content_status"] == "no_pdf"


@pytest.mark.asyncio
async def test_empty_body_without_pdf_marks_metadata_only():
    """Page body empty + no PDF attachments → content_status=no_pdf."""
    search_resp = httpx.Response(
        200,
        request=httpx.Request("GET", "https://wiki.example.test/rest/api/content/search"),
        json={"results": [{"id": "999", "title": "Some_PDF.pdf"}]},
    )
    no_attach = httpx.Response(
        200,
        request=httpx.Request("GET", "https://wiki.example.test/rest/api/content/999/child/attachment"),
        json={"results": []},
    )
    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(side_effect=[search_resp, no_attach])):
        result = await ConfluenceConnector("https://wiki.example.test", "secret").search_live(
            SearchRequest(query="999", topk=1)
        )
    assert len(result) == 1
    assert result[0].metadata["content_status"] == "no_pdf"
    assert result[0].text == ""


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
