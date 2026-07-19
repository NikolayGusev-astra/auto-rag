from rag_core.gateway.adaptive.contracts import RoutingFeedback
from rag_core.gateway.adaptive.enrichment import MemvidEnricher
from rag_core.gateway.adaptive.feedback_store import FeedbackStore
from rag_core.gateway.models import Evidence


def test_feedback_store_persists_and_reloads_events(tmp_path):
    path = tmp_path / "feedback.jsonl"
    store = FeedbackStore(path)
    store.record(RoutingFeedback(
        "query", "plan", ("local", "web"), ("local",), ("doc-1",), 1, 25,
    ))

    assert path.exists()
    reloaded = FeedbackStore(path)
    assert reloaded.events == store.events
    assert reloaded.aggregate()["total"] == 1
    assert reloaded.aggregate()["by_source"] == {"local": 1, "web": 1}


def test_feedback_store_canaries_forbidden_sources(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    store.record(RoutingFeedback("query", "plan", ("bad",), (), (), 0, 25))

    assert store.evaluate({"forbidden_sources": ["bad"]})["canary"] is True
    assert FeedbackStore().evaluate({"forbidden_sources": ["bad"]})["canary"] is False


def test_enricher_persists_and_reloads_episodes(tmp_path):
    path = tmp_path / "episodes.jsonl"
    enricher = MemvidEnricher(path)
    episode = enricher.build_episode("query", [Evidence("chunk", "doc", "", "text", "local")])
    enricher.persist_episode(episode)

    assert path.exists()
    assert len(MemvidEnricher(path).episodes) == 1
