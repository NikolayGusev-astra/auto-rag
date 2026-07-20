import json

from rag_core.gateway.adaptive.dcd_learner import DcdLearner
from rag_core.gateway.adaptive.planner import DcdPlanner


def test_learn_from_episodes_builds_keyword_to_source_map(tmp_path):
    learner = DcdLearner(tmp_path / "episodes.jsonl", tmp_path / "routing.json")

    routing = learner.learn_from_episodes(
        [
            {
                "query": "Kubernetes deployment guide",
                "route": ["live", "web"],
                "document_ids": ["doc-1", "doc-2"],
                "reranker_score": 0.8,
            },
            {
                "query": "Kubernetes deployment troubleshooting",
                "route": ["live"],
                "document_ids": ["doc-3"],
                "reranker_score": 1.0,
            },
        ]
    )

    assert routing["source_affinity"]["kubernetes"] == {"live": 0.9, "web": 0.8}
    assert routing["source_affinity"]["deployment"] == {"live": 0.9, "web": 0.8}


def test_empty_episodes_returns_empty_dict(tmp_path):
    learner = DcdLearner(tmp_path / "episodes.jsonl", tmp_path / "routing.json")

    assert learner.learn_from_episodes([]) == {}


def test_merge_with_existing_routing_preserves_manual_entries(tmp_path):
    routing_path = tmp_path / "routing.json"
    routing_path.write_text(
        json.dumps({"astra-linux": {"name": "Astra Linux", "space": "AL"}}),
        encoding="utf-8",
    )
    learner = DcdLearner(tmp_path / "episodes.jsonl", routing_path)

    routing = learner.learn_from_episodes(
        [{"query": "Astra update", "route": ["local"], "document_ids": ["doc-1"]}]
    )

    assert routing["astra-linux"] == {"name": "Astra Linux", "space": "AL"}
    assert routing["source_affinity"]["astra"] == {"local": 1.0}
    assert json.loads(routing_path.read_text(encoding="utf-8")) == routing


def test_planner_boosts_source_with_strong_keyword_affinity(tmp_path):
    routing_path = tmp_path / "routing.json"
    routing_path.write_text(
        json.dumps({"source_affinity": {"kubernetes": {"web": 0.9}}}),
        encoding="utf-8",
    )

    plan = DcdPlanner(routing_path=routing_path).plan(
        "Kubernetes deployment", {"local": True, "live": True, "web": True}, {}
    )

    assert plan.sources == ("web", "local", "live")
