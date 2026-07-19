import os
import pytest
from unittest import mock

os.environ.setdefault("RAG_FEDERATED_ENABLED", "true")

from rag_core import rag_async


@pytest.mark.asyncio
async def test_federation_called_without_nameerror():
    """Регрессия бага k:553 — query_federated_servers вызывался с неопределённой k.
    До фикса бросал NameError (молча глотался в except, федерация мертва).
    После фикса: max_results=5, вызов проходит без NameError и доходит до federation."""
    captured = {}

    async def fake_fed(query, max_results=3, domain=""):
        captured["called"] = True
        captured["query"] = query
        captured["max_results"] = max_results
        captured["domain"] = domain
        return {}

    # Изолируем пайплайн: все источники пусты, чтобы дошёл до federation-блока
    def fake_zvec(q):
        return {"chunks": [], "max_score": 0.0}

    def fake_web(q, domain, collection):
        return []

    def fake_entities(q, chunks):
        return True  # не False → не уходим в MCP по entity mismatch

    async def fake_mcp(q, domain, collection, loop, trace, tenant_id="default"):
        return {"chunks": []}

    with mock.patch("rag_federated.query_federated_servers", side_effect=fake_fed), \
         mock.patch.object(rag_async, "_blocking_zvec", side_effect=fake_zvec), \
         mock.patch.object(rag_async, "_blocking_web", side_effect=fake_web), \
         mock.patch.object(rag_async, "_check_entities_in_query", side_effect=fake_entities), \
         mock.patch.object(rag_async, "_fallback_to_mcp_web", side_effect=fake_mcp), \
         mock.patch.dict(os.environ, {"RAG_FEDERATED_ENABLED": "true"}):
        try:
            await rag_async._async_rag_search_impl(
                query="тест федерации",
                dcd_result={"domain": "astra", "collection": "astra",
                             "confidence": 0.5, "fallback": False},
                trace=rag_async.RagTrace("тест федерации", "astra", "astra"),
            )
        except NameError as e:
            pytest.fail(f"Federation block raises NameError (bug k:553 not fixed): {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error before reaching federation: {type(e).__name__}: {e}")

    assert captured.get("called") is True, "query_federated_servers не вызван — federation мертва"
    assert captured.get("max_results") == 5, "ожидался max_results=5 (фикс k:553)"