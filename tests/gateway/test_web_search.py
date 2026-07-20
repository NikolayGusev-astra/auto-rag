from __future__ import annotations

import pytest

from rag_core.gateway.models import Evidence, EvidenceOrigin


@pytest.mark.asyncio
async def test_web_search_returns_evidence_duckduckgo(monkeypatch):
    from rag_core.gateway.connectors.web_search import WebSearchConnector

    class FakeDDGS:
        def text(self, query, max_results):
            assert query == "ADR-005 web research"
            assert max_results == 2
            return [{"href": "https://example.test/article", "title": "Article", "body": "Snippet"}]

    connector = WebSearchConnector(ddgs_factory=FakeDDGS, sleep=lambda _: None)

    results = await connector.search_live("ADR-005 web research", 2)

    assert len(results) == 1
    assert results[0].uri == "https://example.test/article"
    assert results[0].title == "Article"
    assert results[0].text == "Snippet"
    assert results[0].source == "web"
    assert results[0].origin is EvidenceOrigin.WEB
    assert results[0].metadata["retrieval_kind"] == "live"


@pytest.mark.asyncio
async def test_web_extract_trafilatura(monkeypatch):
    from rag_core.gateway.connectors.web_extract import WebExtractConnector

    connector = WebExtractConnector(extractor=lambda html: html)

    async def fake_download(url):
        assert url == "https://example.test/article"
        return "<article>Useful text</article><script>unsafe()</script>"

    monkeypatch.setattr(connector, "_download", fake_download)

    evidence = await connector.fetch("https://example.test/article")

    assert evidence.text == "<article>Useful text</article>"
    assert evidence.uri == "https://example.test/article"
    assert evidence.origin is EvidenceOrigin.WEB
    assert evidence.metadata["url"] == "https://example.test/article"


@pytest.mark.asyncio
async def test_web_pipeline_search_extract_fallback():
    from rag_core.gateway.web_pipeline import WebPipeline

    short = Evidence("search:1", "https://one.test", "First", "snippet", "web", uri="https://one.test", origin=EvidenceOrigin.WEB)
    long = Evidence("search:2", "https://two.test", "Second", "snippet", "web", uri="https://two.test", origin=EvidenceOrigin.WEB)

    class Search:
        async def search_live(self, query, topk):
            assert (query, topk) == ("test query", 6)
            return [short, long]

    class Extract:
        async def fetch(self, url):
            return Evidence(url, url, "", "short", "web", uri=url, origin=EvidenceOrigin.WEB)

    class Browser:
        async def fetch(self, url):
            return Evidence(url, url, "", "browser " * 40, "web", uri=url, origin=EvidenceOrigin.WEB)

    results = await WebPipeline(True, search=Search(), extract=Extract(), browser=Browser()).research("test query", topk=2)

    assert len(results) == 2
    assert all(result.text.startswith("browser") for result in results)
    assert [result.title for result in results] == ["First", "Second"]
    assert all(result.origin is EvidenceOrigin.WEB for result in results)


@pytest.mark.asyncio
async def test_web_off_by_default():
    from rag_core.gateway.web_pipeline import WebPipeline

    class Search:
        async def search_live(self, query, topk):
            raise AssertionError("web search must not run while disabled")

    assert await WebPipeline(search=Search()).research("test query") == []
