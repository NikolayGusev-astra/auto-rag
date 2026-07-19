import logging

import pytest

from rag_core.gateway.adaptive.contracts import QueryPlan
from rag_core.gateway.adaptive.feedback import FeedbackStore
from rag_core.gateway.adaptive.loop import AdaptiveLoop
from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.models import Evidence, EvidenceOrigin


class Connector:
    def __init__(self, source, evidence=(), *, available=True, error=None):
        self.source = source
        self.evidence = list(evidence)
        self.available = available
        self.error = error
        self.calls = 0

    async def health(self):
        return {"available": self.available}

    async def search_live(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return self.evidence


def planner(*, include_web, sources=("local", "web"), top_k=5):
    return type("Planner", (), {
        "plan": staticmethod(lambda query, availability, hints: QueryPlan(
            query, (query,), (), sources, True, True, include_web, top_k, hints=hints,
        ))
    })()


@pytest.mark.asyncio
async def test_plan_excludes_web_connector():
    local = Connector("local", [Evidence("l", "l", "local", "", "local", retrieval_score=1)])
    web = Connector("web", [Evidence("w", "w", "web", "", "web", retrieval_score=1)])

    response = await AdaptiveLoop(enabled=True).run(
        SearchRequest(query="q"), {"local": local, "web": web}, planner=planner(include_web=False),
    )

    assert local.calls == 1
    assert web.calls == 0
    assert [item["document_id"] for item in response["results"]] == ["l"]


@pytest.mark.asyncio
async def test_dead_connector_is_logged_and_healthy_result_is_returned(caplog):
    healthy = Connector("local", [Evidence("l", "l", "local", "", "local", retrieval_score=1)])
    broken = Connector("jira", error=ValueError("bad response"))

    with caplog.at_level(logging.WARNING):
        response = await AdaptiveLoop(enabled=True).run(
            SearchRequest(query="q"), {"local": healthy, "jira": broken},
            planner=planner(include_web=False, sources=("local", "jira")),
        )

    assert [item["document_id"] for item in response["results"]] == ["l"]
    assert response["metadata"]["failed_sources"] == ["jira"]
    assert any(
        record.source == "jira" and record.error_type == "ValueError"
        and record.error_message == "bad response"
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_plan_top_k_truncates_fused_results():
    local = Connector("local", [
        Evidence("l1", "l1", "one", "", "local", retrieval_score=0.9),
        Evidence("l2", "l2", "two", "", "local", retrieval_score=0.8),
    ])
    jira = Connector("jira", [Evidence("j", "j", "three", "", "jira", retrieval_score=0.7)])

    response = await AdaptiveLoop(enabled=True).run(
        SearchRequest(query="q", topk=9), {"local": local, "jira": jira},
        planner=planner(include_web=False, sources=("local", "jira"), top_k=2),
    )

    assert len(response["results"]) == 2


@pytest.mark.asyncio
async def test_coordinator_preserves_connector_origin():
    jira = Connector("jira", [Evidence(
        "j", "j", "ticket", "", "jira", origin=EvidenceOrigin.LIVE_CORPORATE, retrieval_score=1,
    )])

    response = await AdaptiveLoop(enabled=True).run(
        SearchRequest(query="q"), {"jira": jira},
        planner=planner(include_web=False, sources=("jira",)),
    )

    assert response["results"][0]["origin"] is EvidenceOrigin.LIVE_CORPORATE


@pytest.mark.asyncio
async def test_feedback_is_persisted_after_run(tmp_path):
    feedback_path = tmp_path / "routing-feedback.jsonl"
    feedback = FeedbackStore(feedback_path)
    local = Connector("local", [Evidence("l", "l", "local", "", "local", retrieval_score=1)])

    await AdaptiveLoop(enabled=True).run(
        SearchRequest(query="q"), {"local": local}, planner=planner(include_web=False, sources=("local",)),
        feedback=feedback,
    )

    assert feedback_path.read_text(encoding="utf-8").strip()
    event = feedback.events[0]
    assert event.successful_sources == ("local",)
    assert event.useful_document_ids == ("l",)
    assert event.latency_ms >= 0


@pytest.mark.asyncio
async def test_episode_is_built_with_active_index_context_and_persisted():
    class Enricher:
        def __init__(self):
            self.calls = []
            self.persisted = []

        def build_episode(self, query, evidence, **kwargs):
            self.calls.append((query, evidence, kwargs))
            return "episode"

        def persist_episode(self, episode):
            self.persisted.append(episode)

    enricher = Enricher()
    local = Connector("local", [Evidence("l", "l", "local", "", "local", retrieval_score=1)])

    await AdaptiveLoop(enabled=True).run(
        SearchRequest(query="q"), {"local": local}, planner=planner(include_web=False, sources=("local",)),
        enricher=enricher, active_revision_path="/indexes/revision-7", embedding_profile_id="embed-v2",
    )

    assert enricher.calls[0][2]["index_revision"] == "/indexes/revision-7"
    assert enricher.calls[0][2]["embedding_profile_id"] == "embed-v2"
    assert enricher.persisted == ["episode"]
