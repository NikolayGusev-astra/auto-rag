from rag_core.gateway.adaptive.contracts import QueryPlan
from rag_core.gateway.adaptive.planner import DcdPlanner


def test_planner_returns_plan_not_retrieval():
    plan = DcdPlanner().plan(
        "update astra cluster", {"local": True, "live": True, "web": False}, {},
    )
    assert isinstance(plan, QueryPlan)
    assert plan.include_local is True
    assert plan.include_web is False


def test_planner_compound_splits_queries():
    plan = DcdPlanner().plan(
        "astra product and infrastructure", {"local": True, "live": True, "web": False}, {},
    )
    assert len(plan.queries) >= 2
    assert plan.include_live is True
