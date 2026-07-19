import pytest

from rag_core.gateway.adaptive.contracts import QueryPlan
from rag_core.gateway.adaptive.loop import AdaptiveLoop
from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.models import Evidence


@pytest.mark.asyncio
async def test_reference_mode_skips_memory_and_learning():
    class Local:
        source = "local"

        async def search_live(self, request):
            return [Evidence("d1", "d1", "", "x", "local", retrieval_score=0.7)]

        async def health(self):
            return {"available": True}

    response = await AdaptiveLoop(enabled=False).run(SearchRequest(query="q"), {"local": Local()})
    assert "results" in response
    assert all(item.get("origin") != "agent_memory" for item in response["results"])


@pytest.mark.asyncio
async def test_adaptive_mode_merges_memory_no_short_circuit():
    class Local:
        source = "local"

        async def search_live(self, request):
            return [Evidence("d1", "d1", "", "x", "local", retrieval_score=0.7)]

        async def health(self):
            return {"available": True}

    async def memory_health():
        return {"available": True}

    async def memory_search(request):
        return []

    memory = type("Memory", (), {
        "source": "agent_memory",
        "search_live": staticmethod(memory_search),
        "health": staticmethod(memory_health),
        "as_memory_evidence": staticmethod(lambda index: None),
        "is_compatible": staticmethod(lambda profile: True),
    })()
    planner = type("Planner", (), {"plan": staticmethod(lambda query, availability, hints: QueryPlan(
        query, (query,), ("astra",), ("local",), True, True, False, 5, hints={},
    ))})()

    response = await AdaptiveLoop(enabled=True).run(
        SearchRequest(query="q"), {"local": Local()}, memory=memory, planner=planner,
    )
    assert any(item.get("document_id") == "d1" for item in response["results"])
