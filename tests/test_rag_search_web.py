from unittest import mock

import rag_core.rag_search as rag_search


def test_web_enrichment_returns_all_search_candidates(monkeypatch):
    """C2/C3: the loop must not return after enriching only the first page."""
    response = mock.MagicMock()
    response.json.return_value = {
        "results": [
            {"title": "one", "url": "https://one.example", "content": "first"},
            {"title": "two", "url": "https://two.example", "content": "second"},
            {"title": "three", "url": "https://three.example", "content": "third"},
        ]
    }
    response.raise_for_status.return_value = None
    monkeypatch.setattr(rag_search, "SEARXNG_ENABLED", True)
    monkeypatch.setattr(rag_search, "_TRAFILATURA_AVAILABLE", False)
    monkeypatch.setattr(rag_search.requests, "get", mock.MagicMock(return_value=response))

    results = rag_search.searxng_search("test", max_results=3)

    assert [item["title"] for item in results] == ["one", "two", "three"]
