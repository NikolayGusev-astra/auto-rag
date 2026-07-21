from unittest.mock import AsyncMock, patch

import httpx
import pytest

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.connectors.allowlisted_web import (
    AllowlistedWebConnector,
    _build_domain_query,
    _extract_full_text,
    is_public_doc_query,
)


def test_internal_query_returns_false():
    assert is_public_doc_query("SIRIUS-195479 ошибки") is False
    assert is_public_doc_query("BT-37295 VLV index") is False
    assert is_public_doc_query("INT-1234 задача") is False
    assert is_public_doc_query("PROJECT-14960 обновление") is False
    assert is_public_doc_query("PRESALE-11471 ЦБ РФ") is False


def test_public_doc_query_returns_true():
    assert is_public_doc_query("матрица совместимости ALD Pro") is True
    assert is_public_doc_query("инструкция по обновлению контроллера домена") is True
    assert is_public_doc_query("руководство администратора Astra Linux") is True
    assert is_public_doc_query("поддерживаемая версия ALD Pro") is True
    assert is_public_doc_query("release notes 3.2.1") is True


def test_public_doc_query_returns_false_for_generic():
    """Generic operational queries without doc intent must be blocked."""
    assert is_public_doc_query("как обновить контроллер домена") is False
    assert is_public_doc_query("как отладить kerberos") is False
    assert is_public_doc_query("что такое ansible role") is False


def test_domain_query_builder():
    q = _build_domain_query("матрица обновлений")
    assert "site:" in q
    assert "aldpro.ru" in q
    assert "astralinux.ru" in q
    assert "матрица обновлений" in q


@pytest.mark.asyncio
async def test_internal_query_skipped():
    c = AllowlistedWebConnector("http://localhost:8888")
    result = await c.search_live(SearchRequest(query="SIRIUS-195479", topk=3))
    assert result == []


@pytest.mark.asyncio
async def test_public_query_searches():
    c = AllowlistedWebConnector("http://localhost:8888")
    resp = httpx.Response(
        200,
        request=httpx.Request("GET", "http://localhost:8888/search"),
        json={
            "results": [
                {
                    "url": "https://aldpro.ru/materials/3.2.0/matrix",
                    "title": "Матрица обновлений",
                    "content": "Путь 2.4.4 → 3.2.0",
                }
            ]
        },
    )
    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=resp)):
        result = await c.search_live(SearchRequest(query="матрица совместимости ALD Pro", topk=3))
    assert len(result) == 1
    assert result[0].metadata["authoritative"] is True
    assert result[0].metadata["domain"] == "aldpro.ru"


@pytest.mark.asyncio
async def test_search_discards_results_outside_the_allowlist():
    connector = AllowlistedWebConnector("http://localhost:8888")
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "http://localhost:8888/search"),
        json={
            "results": [
                {"url": "https://example.test/copied-doc", "title": "Copy", "content": "unsafe"},
                {"url": "https://aldpro.ru/docs/guide", "title": "Guide", "content": "safe"},
            ]
        },
    )

    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=response)):
        result = await connector.search_live(SearchRequest(query="руководство администратора Astra Linux", topk=1))

    assert [evidence.uri for evidence in result] == ["https://aldpro.ru/docs/guide"]


@pytest.mark.asyncio
async def test_fulltext_extraction_enriches_snippet():
    """When Trafilatura returns text, evidence is replaced with full text."""
    connector = AllowlistedWebConnector("http://localhost:8888")
    # SearXNG response
    searxng_resp = httpx.Response(
        200,
        request=httpx.Request("GET", "http://localhost:8888/search"),
        json={
            "results": [
                {
                    "url": "https://aldpro.ru/materials/3.2.0/matrix",
                    "title": "Матрица совместимости",
                    "content": "Путь 2.4.4 → 3.2.0",
                }
            ]
        },
    )
    # Full-text response (the page HTML)
    fulltext_resp = httpx.Response(
        200,
        request=httpx.Request("GET", "https://aldpro.ru/materials/3.2.0/matrix"),
        text="<html><body><p>Полная таблица совместимости</p></body></html>",
    )

    # Mock: first call → SearXNG, second → fulltext fetch
    get_mock = AsyncMock(side_effect=[searxng_resp, fulltext_resp])
    # Trafilatura will extract "Полная таблица совместимости" from the HTML
    with patch.object(httpx.AsyncClient, "get", new=get_mock):
        result = await connector.search_live(SearchRequest(query="матрица совместимости ALD Pro", topk=3))

    assert len(result) == 1
    assert result[0].metadata["extraction_method"] == "fulltext"
    assert "таблица совместимости" in result[0].text.lower()
    assert result[0].metadata["authoritative"] is True


@pytest.mark.asyncio
async def test_fulltext_failure_falls_back_to_snippet():
    """When Trafilatura/TraHTTP fails, snippet is preserved."""
    connector = AllowlistedWebConnector("http://localhost:8888")
    # SearXNG response
    searxng_resp = httpx.Response(
        200,
        request=httpx.Request("GET", "http://localhost:8888/search"),
        json={
            "results": [
                {
                    "url": "https://aldpro.ru/docs/guide",
                    "title": "Guide",
                    "content": "snippet text",
                }
            ]
        },
    )
    # Full-text fetch fails
    get_mock = AsyncMock(side_effect=[searxng_resp, httpx.ConnectError("timeout")])
    with patch.object(httpx.AsyncClient, "get", new=get_mock):
        result = await connector.search_live(SearchRequest(query="руководство администратора", topk=1))

    assert len(result) == 1
    assert result[0].metadata["extraction_method"] == "snippet_fallback"
    assert result[0].text == "snippet text"


def test_extract_full_text_returns_empty_on_error():
    """Network/protocol errors yield empty string."""
    import asyncio as _asyncio
    result = _asyncio.run(_extract_full_text("https://invalid.test/404"))
    assert result == ""
