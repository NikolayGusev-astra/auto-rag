import pytest
from dataclasses import FrozenInstanceError

from rag_core.gateway.adaptive.contracts import (
    MemoryEpisode,
    QueryPlan,
    RoutingFeedback,
)


def test_queryplan_is_frozen_and_has_flags():
    plan = QueryPlan("q", ("q",), ("astra",), ("local",), True, True, False, 5, hints={})
    assert plan.include_web is False
    with pytest.raises(FrozenInstanceError):
        plan.include_web = True


def test_routingfeedback_captures_usefulness():
    feedback = RoutingFeedback("q", "p1", ("local",), ("local",), ("d1",), 3, 42, explicit_success=True)
    assert feedback.explicit_success is True
    assert "d1" in feedback.useful_document_ids


def test_memoryepisode_requires_provenance():
    episode = MemoryEpisode("e1", "q", "s", ("local",), ("d1",), ("u1",), ("x",), True, None, "rev1", "prof1")
    assert episode.document_ids == ("d1",)
    assert episode.index_revision == "rev1"
