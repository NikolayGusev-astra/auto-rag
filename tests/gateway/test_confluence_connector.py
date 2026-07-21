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


def _sync_page(page_id: str, title: str, when: str, body: str = "<p>Body</p>") -> dict:
    return {
        "id": page_id,
        "title": title,
        "body": {"storage": {"value": body}},
        "version": {"number": 3, "when": when},
    }


def _response(url: str, payload: dict | None = None, content: bytes = b"") -> httpx.Response:
    if payload is not None:
        return httpx.Response(200, request=httpx.Request("GET", url), json=payload)
    return httpx.Response(200, request=httpx.Request("GET", url), content=content)


@pytest.mark.asyncio
async def test_sync_returns_batch():
    page = _sync_page("42", "Runbook", "2026-07-01T10:00:00Z", "<p>Restart gateway.</p>")
    with patch.object(
        httpx.AsyncClient,
        "get",
        new=AsyncMock(side_effect=[
            _response("https://wiki.example.test/rest/api/content/search", {"results": [page]}),
            _response("https://wiki.example.test/rest/api/content/42/child/attachment", {"results": []}),
        ]),
    ) as get:
        batch = await ConfluenceConnector("https://wiki.example.test", "secret").sync_changes(None)

    assert [document.id for document in batch.added] == ["confluence:42"]
    assert batch.added[0].text == "Restart gateway."
    assert batch.cursor == "2026-07-01T10:00:00Z"
    assert get.await_args_list[0].kwargs["params"]["limit"] == 100


@pytest.mark.asyncio
async def test_sync_with_pdf_attachments():
    page = _sync_page("42", "Runbook", "2026-07-01T10:00:00Z")
    attachments = {"results": [{"title": "guide.pdf"}]}
    with patch("rag_core.gateway.connectors.confluence_connector._parse_pdf_bytes", return_value="PDF text"), patch.object(
        httpx.AsyncClient,
        "get",
        new=AsyncMock(side_effect=[
            _response("https://wiki.example.test/rest/api/content/search", {"results": [page]}),
            _response("https://wiki.example.test/rest/api/content/42/child/attachment", attachments),
            _response("https://wiki.example.test/download/attachments/42/guide.pdf", content=b"pdf"),
        ]),
    ):
        batch = await ConfluenceConnector("https://wiki.example.test", "secret").sync_changes(None)

    assert batch.added[0].text == "Body\n\n[ATTACHED PDF]\n[guide.pdf]\nPDF text"
    assert batch.added[0].metadata["pdf_status"] == "ok"


@pytest.mark.asyncio
async def test_sync_incremental():
    page = _sync_page("43", "Newer", "2026-07-02T10:00:00Z")
    with patch.object(
        httpx.AsyncClient,
        "get",
        new=AsyncMock(side_effect=[
            _response("https://wiki.example.test/rest/api/content/search", {"results": [page]}),
            _response("https://wiki.example.test/rest/api/content/43/child/attachment", {"results": []}),
        ]),
    ) as get:
        batch = await ConfluenceConnector("https://wiki.example.test", "secret").sync_changes("2026-07-01T00:00:00Z")

    assert batch.cursor == "2026-07-02T10:00:00Z"
    assert get.await_args_list[0].kwargs["params"]["cql"] == 'lastModified >= "2026-07-01T00:00:00Z" AND (type=page)'


@pytest.mark.asyncio
async def test_sync_pagination():
    first_page = [_sync_page(str(index), f"Page {index}", "2026-07-01T10:00:00Z") for index in range(100)]
    with patch.object(
        httpx.AsyncClient,
        "get",
        new=AsyncMock(side_effect=[
            _response("https://wiki.example.test/rest/api/content/search", {"results": first_page}),
            _response("https://wiki.example.test/rest/api/content/search", {"results": []}),
            *[_response(f"https://wiki.example.test/rest/api/content/{index}/child/attachment", {"results": []}) for index in range(100)],
        ]),
    ) as get:
        batch = await ConfluenceConnector("https://wiki.example.test", "secret").sync_changes(None)

    search_calls = [call for call in get.await_args_list if call.kwargs.get("params", {}).get("cql")]
    assert len(batch.added) == 100
    assert [call.kwargs["params"]["start"] for call in search_calls] == [0, 100]


@pytest.mark.asyncio
async def test_sync_empty_cursor():
    with patch.object(
        httpx.AsyncClient,
        "get",
        new=AsyncMock(return_value=_response("https://wiki.example.test/rest/api/content/search", {"results": []})),
    ) as get:
        batch = await ConfluenceConnector("https://wiki.example.test", "secret").sync_changes(None)

    assert batch.added == []
    assert batch.cursor is None
    assert get.await_args.kwargs["params"]["cql"] == 'lastModified >= "2020-01-01" AND (type=page)'
