import asyncio

import pytest

from rag_core.gateway.adaptive.contracts import QueryPlan
from rag_core.gateway.adaptive.loop import AdaptiveLoop
from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.models import Evidence


class Connector:
    def __init__(self, source, retrieval_kind, evidence=(), *, available=True, delay=0):
        self.source = source
        self.retrieval_kind = retrieval_kind
        self.evidence = list(evidence)
        self.available = available
        self.delay = delay
        self.calls = 0

    async def health(self):
        return {"available": self.available}

    async def search_live(self, request):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.evidence


class Planner:
    def __init__(self, plan):
        self._plan = plan
        self.availability = None

    def plan(self, query, availability, hints):
        self.availability = availability
        return self._plan


def plan(*, sources, include_local=False, include_live=False, include_web=False, queries=("q",), budget=None):
    return QueryPlan(
        original_query="q",
        queries=queries,
        domains=(),
        sources=sources,
        include_local=include_local,
        include_live=include_live,
        include_web=include_web,
        max_results=5,
        retrieval_budget_ms=budget,
    )


def evidence(document_id):
    return Evidence(document_id, document_id, document_id, "", document_id, retrieval_score=1)


@pytest.mark.asyncio
async def test_kind_routing_calls_local_snapshot_and_jira():
    snapshot = Connector("local_snapshot", "local", [evidence("snapshot")])
    jira = Connector("jira", "live", [evidence("jira")])
    route_planner = Planner(plan(sources=("local", "live"), include_local=True, include_live=True))

    await AdaptiveLoop(enabled=True).run(
        SearchRequest(query="q"),
        {"snapshot": snapshot, "jira-prod": jira},
        planner=route_planner,
    )

    assert snapshot.calls == 1
    assert jira.calls == 1
    assert route_planner.availability == {"local": True, "live": True}


@pytest.mark.asyncio
async def test_explicit_source_routes_jira_without_local_snapshot():
    snapshot = Connector("local_snapshot", "local", [evidence("snapshot")])
    jira = Connector("jira", "live", [evidence("jira")])

    await AdaptiveLoop(enabled=True).run(
        SearchRequest(query="q"),
        {"snapshot": snapshot, "jira-prod": jira},
        planner=Planner(plan(sources=("jira",))),
    )

    assert snapshot.calls == 0
    assert jira.calls == 1


@pytest.mark.asyncio
async def test_unavailable_memory_is_not_selected():
    memory = Connector("agent_memory", "memory", available=False)

    await AdaptiveLoop(enabled=True).run(
        SearchRequest(query="q"), {}, memory=memory,
        planner=Planner(plan(sources=("local",), include_local=True)),
    )

    assert memory.calls == 0


@pytest.mark.asyncio
async def test_retrieval_budget_records_timeout_and_preserves_prior_evidence():
    class QueryAwareConnector(Connector):
        async def search_live(self, request):
            self.calls += 1
            if request.query == "slow":
                await asyncio.sleep(0.05)
                return [evidence("late")]
            return [evidence("early")]

    jira = QueryAwareConnector("jira", "live")

    response = await AdaptiveLoop(enabled=True).run(
        SearchRequest(query="q"), {"jira-prod": jira},
        planner=Planner(plan(
            sources=("live",), include_live=True, queries=("fast", "slow"), budget=10,
        )),
    )

    assert [item["document_id"] for item in response["results"]] == ["early"]
    assert response["metadata"]["failed_sources"] == ["jira-prod"]
    assert response["metadata"]["timed_out_sources"] == ["jira-prod"]
