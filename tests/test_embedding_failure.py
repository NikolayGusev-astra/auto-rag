

import os
from unittest import mock

import pytest

from conftest import skip_if_no_chromadb, skip_if_no_embedding

os.environ.setdefault("RAG_FEDERATED_ENABLED", "true")

from rag_core import rag_async
from rag_core.unified_searcher import UnifiedSearcher


@skip_if_no_chromadb
@skip_if_no_embedding
def test_embedding_failure_visible_in_search():
    """R1: сбой embedding (нулевой вектор) не молчит, а возвращает
    is_error-чанк — иначе пользователь видит пустой ответ без причины."""
    with mock.patch("rag_core.unified_searcher._get_embedding", return_value=[0.0] * 1024):
        searcher = UnifiedSearcher(collection="wiki")
        results = searcher.search("любой запрос")
        assert len(results) == 1
        assert results[0].get("is_error") is True, "embedding-сбой должен быть помечен is_error"
        assert "unavailable" in results[0]["text"]


@pytest.mark.asyncio
async def test_embedding_failure_excluded_from_pool():
    """R1 + A1: error-чанк embedding не попадает в pool ответа памяти."""
    def fake_zvec(q):
        return {"chunks": [], "max_score": 0.0}

    def fake_web(q, domain, collection):
        return []

    def fake_entities(q, chunks):
        return True

    async def fake_mcp(q, domain, collection, loop, trace):
        return {"chunks": []}

    with mock.patch("rag_core.unified_searcher._get_embedding", return_value=[0.0] * 1024), \
         mock.patch.object(rag_async, "_blocking_zvec", side_effect=fake_zvec), \
         mock.patch.object(rag_async, "_blocking_web", side_effect=fake_web), \
         mock.patch.object(rag_async, "_check_entities_in_query", side_effect=fake_entities), \
         mock.patch.object(rag_async, "_fallback_to_mcp_web", side_effect=fake_mcp), \
         mock.patch.dict(os.environ, {"RAG_FEDERATED_ENABLED": "true"}):
        result = await rag_async._async_rag_search_impl(
            query="тест embedding сбоя",
            dcd_result={"domain": "astra", "collection": "wiki", "confidence": 0.5},
            trace=rag_async.RagTrace("тест embedding сбоя", "astra", "wiki"),
        )
        chunks = result.get("chunks", [])
        pooled_texts = " ".join(str(c.get("text", "")) for c in chunks)
        assert "embedding service unavailable" not in pooled_texts, \
            "embedding-сбой попал в ответ памяти (R1/A1 не исправлен)"
