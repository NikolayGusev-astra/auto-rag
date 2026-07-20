"""Web pipeline integration tests — ADR-005 acceptance."""
from __future__ import annotations

import pytest

from rag_core.gateway.models import Evidence, EvidenceOrigin
from rag_core.gateway.web_pipeline import WebPipeline


class _Search:
    def __init__(self, results=None):
        self.results = results or []
        self.called = False

    async def search_live(self, request, topk=None):
        self.called = True
        return self.results


class _Extract:
    def __init__(self, text="", fail=False):
        self.text = text
        self.fail = fail
        self.calls = []

    async def fetch(self, url):
        self.calls.append(url)
        if self.fail:
            raise RuntimeError("extract error")
        return Evidence(
            id=f"extract:{url}", document_id=url, title=url,
            text=self.text, source="web", uri=url,
            origin=EvidenceOrigin.PUBLIC_WEB,
            retrieval_score=0.5 if len(self.text) >= 200 else 0.3,
            metadata={"url": url},
        )


class _Browser:
    def __init__(self, text="", fail=False):
        self.text = text
        self.fail = fail
        self.calls = []

    async def fetch(self, url):
        self.calls.append(url)
        if self.fail:
            raise RuntimeError("browser error")
        return Evidence(
            id=f"browser:{url}", document_id=url, title=url,
            text=self.text, source="web", uri=url,
            origin=EvidenceOrigin.PUBLIC_WEB,
            retrieval_score=0.8,
            metadata={"url": url, "extractor": "camoufox"},
        )


@pytest.mark.asyncio
async def test_web_off_by_default():
    search = _Search()
    assert await WebPipeline(allow_web=False, search=search).research("q") == []
    assert not search.called


@pytest.mark.asyncio
async def test_trafilatura_sufficient_no_camoufox():
    search = _Search([Evidence("s:1", "https://a.test", "A", "", "web",
                               uri="https://a.test", origin=EvidenceOrigin.PUBLIC_WEB)])
    extract = _Extract(text="Long " * 100)  # 500 chars ≥ 200
    browser = _Browser()
    results = await WebPipeline(allow_web=True, search=search, extract=extract, browser=browser).research("q", topk=1)
    assert len(results) == 1
    assert results[0].text.startswith("Long")
    assert len(browser.calls) == 0  # Camoufox NOT called


@pytest.mark.asyncio
async def test_trafilatura_short_camoufox_fallback():
    search = _Search([Evidence("s:1", "https://b.test", "B", "", "web",
                               uri="https://b.test", origin=EvidenceOrigin.PUBLIC_WEB)])
    extract = _Extract(text="short")  # < 200 chars
    browser = _Browser(text="Rich content " * 40)
    results = await WebPipeline(allow_web=True, search=search, extract=extract, browser=browser).research("q", topk=1)
    assert len(results) == 1
    assert results[0].text.startswith("Rich content")
    assert len(browser.calls) == 1  # Camoufox called


@pytest.mark.asyncio
async def test_camoufox_unavailable_returns_short_extract():
    search = _Search([Evidence("s:1", "https://c.test", "C", "", "web",
                               uri="https://c.test", origin=EvidenceOrigin.PUBLIC_WEB)])
    extract = _Extract(text="short")
    browser = _Browser(fail=True)
    results = await WebPipeline(allow_web=True, search=search, extract=extract, browser=browser).research("q", topk=1)
    # Falls back to extract result (short but not None)
    assert len(results) == 1
    assert results[0].text == "short"


@pytest.mark.asyncio
async def test_search_failure_returns_empty():
    class FailingSearch:
        async def search_live(self, *args, **kwargs):
            raise RuntimeError("search down")
    assert await WebPipeline(allow_web=True, search=FailingSearch()).research("q") == []


@pytest.mark.asyncio
async def test_non_http_url_skipped():
    search = _Search([Evidence("s:1", "file:///etc/passwd", "Bad", "", "web",
                               uri="file:///etc/passwd", origin=EvidenceOrigin.PUBLIC_WEB)])
    extract = _Extract()
    results = await WebPipeline(allow_web=True, search=search, extract=extract).research("q")
    assert len(results) == 0  # non-http skipped
