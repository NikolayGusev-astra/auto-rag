import pytest

from rag_core.gateway.adaptive.contracts import QueryPlan
from rag_core.gateway.adaptive.loop import AdaptiveLoop
from rag_core.gateway.connector import SearchRequest


@pytest.mark.asyncio
async def test_reference_mode_skips_memory_and_learning():
    class Local:
        source = "local"

        async def search_live(self, request):
            return [{"document_id": "d1", "text": "x", "score": 0.7}]

    response = await AdaptiveLoop(enabled=False).run(SearchRequest(query="q"), {"local": Local()})
    assert "results" in response
    assert all(item.get("origin") != "agent_memory" for item in response["results"])


@pytest.mark.asyncio
async def test_adaptive_mode_merges_memory_no_short_circuit():
    class Local:
        source = "local"

        async def search_live(self, request):
            return [{"document_id": "d1", "text": "x", "score": 0.7}]

    memory = type("Memory", (), {
        "search_live": staticmethod(lambda request: []),
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
