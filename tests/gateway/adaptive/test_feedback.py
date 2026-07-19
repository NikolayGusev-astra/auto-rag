from rag_core.gateway.adaptive.contracts import RoutingFeedback
from rag_core.gateway.adaptive.feedback import FeedbackStore


def test_feedback_aggregates_source_usefulness():
    store = FeedbackStore()
    store.record(RoutingFeedback("q", "p1", ("local", "web"), ("local",), ("d1",), 3, 40, None, True))
    stats = store.aggregate()
    assert stats["local"]["useful"] == 1
    assert stats["web"]["useful"] == 0


def test_feedback_evaluate_is_a_golden_hook():
    store = FeedbackStore()
    assert store.evaluate([{"query": "q"}]) == {"events": 0, "golden_size": 1}
