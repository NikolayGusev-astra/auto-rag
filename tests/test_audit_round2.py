import os
from unittest import mock

import pytest

# --- S1: federated_endpoint bind host ---
import federated_endpoint as fe


def test_bind_host_localhost_without_key(monkeypatch):
    """Security HIGH #2: без API-ключа сервер биндится на localhost,
    а не на 0.0.0.0 (иначе открыт всей сети без аутентификации)."""
    monkeypatch.delenv("RAG_FEDERATED_API_KEY", raising=False)
    assert fe.get_bind_host() == "127.0.0.1"


def test_bind_host_any_with_key(monkeypatch):
    """С заданным ключом — разрешаем 0.0.0.0 (узел защищён auth)."""
    monkeypatch.setenv("RAG_FEDERATED_API_KEY", "secret")
    assert fe.get_bind_host() == "0.0.0.0"


# --- A1: federated error chunk не попадает в pool ответа памяти ---
os.environ.setdefault("RAG_FEDERATED_ENABLED", "true")

from rag_core import rag_async


@pytest.mark.asyncio
async def test_federated_error_chunk_excluded_from_pool():
    """A1: error-чанк (is_error=True) не должен попадать в pool результатов
    памяти (иначе fuser выведет текст ошибки как ответ)."""
    pooled = {}

    async def fake_fed(query, max_results=3, domain=""):
        # имитируем error-чанк от federated клиента
        return {"node1": [{"text": "Federated RAG node1 error: timeout",
                           "source": "node1", "score": 0, "is_error": True}]}

    def fake_zvec(q):
        return {"chunks": [], "max_score": 0.0}

    def fake_web(q, domain, collection):
        return []

    def fake_entities(q, chunks):
        return True

    async def fake_mcp(q, domain, collection, loop, trace):
        return {"chunks": []}

    with mock.patch("rag_federated.query_federated_servers", side_effect=fake_fed), \
         mock.patch.object(rag_async, "_blocking_zvec", side_effect=fake_zvec), \
         mock.patch.object(rag_async, "_blocking_web", side_effect=fake_web), \
         mock.patch.object(rag_async, "_check_entities_in_query", side_effect=fake_entities), \
         mock.patch.object(rag_async, "_fallback_to_mcp_web", side_effect=fake_mcp), \
         mock.patch.dict(os.environ, {"RAG_FEDERATED_ENABLED": "true"}):
        result = await rag_async._async_rag_search_impl(
            query="тест",
            dcd_result={"domain": "astra", "collection": "wiki", "confidence": 0.5},
            trace=rag_async.RagTrace("тест", "astra", "wiki"),
        )
        # pool формируется внутри; проверяем, что в итоговом результате
        # нет текста ошибки федерации
        chunks = result.get("chunks", [])
        pooled_texts = " ".join(str(c.get("text", "")) for c in chunks)
        assert "Federated RAG node1 error" not in pooled_texts, \
            "error-чанк федерации попал в ответ памяти (A1 не исправлен)"
