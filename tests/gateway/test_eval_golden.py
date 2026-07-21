from __future__ import annotations

import json

import pytest

from rag_core.eval_golden import (
    Qwen25Judge,
    citation_correctness,
    compute_mrr,
    compute_ndcg,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_retrieval,
    source_coverage,
)


def test_precision_recall_mrr_ndcg() -> None:
    expected = ["doc-1", "doc-2"]
    returned = ["miss", "doc-2", "doc-1"]

    assert compute_precision_at_k(expected, returned, 2) == 0.5
    assert compute_recall_at_k(expected, returned, 2) == 0.5
    assert compute_mrr(expected, returned, 3) == 0.5
    assert compute_ndcg(expected, [{"document_id": item} for item in returned]) == pytest.approx(0.6934, abs=0.0001)


def test_source_coverage() -> None:
    coverage = source_coverage(
        [
            {"returned_sources": ["wiki", "drive"]},
            {"returned_sources": ["wiki"]},
            {"returned_sources": []},
        ]
    )

    assert coverage == {"wiki": 2 / 3, "drive": 1 / 3}


def test_citation_correctness() -> None:
    assert citation_correctness(["doc-1", "doc-2"], ["doc-2", "wrong"]) == 1.0  # found doc-2
    assert citation_correctness(["doc-1"], ["wrong", "wrong2"]) == 0.0  # none found
    assert citation_correctness([], []) == 1.0  # nothing expected
    assert citation_correctness(["doc-1"], []) == 0.0  # expected but returned empty


def test_empty_rate() -> None:
    report = evaluate_retrieval(
        [
            {"query": "one", "expected_document_ids": ["doc-1"]},
            {"query": "", "expected_document_ids": []},
        ],
        [
            {"returned_document_ids": ["doc-1"], "returned_sources": ["wiki"], "latency_s": 0.2},
            {"returned_document_ids": [], "returned_sources": [], "latency_s": 0.4},
        ],
    )

    assert report["empty_query_rate"] == 0.5
    assert report["empty_result_rate"] == 0.5
    assert report["latency_s"]["p50"] == 0.3


def test_qwen_judge_skipped_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = Qwen25Judge(base_url="http://unavailable/v1")
    monkeypatch.setattr(judge, "is_available", lambda: False)

    assert judge.evaluate_evidence("q", "answer", ["evidence"]) == {"llm_judge_status": "skipped"}


def test_qwen_judge_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = Qwen25Judge(base_url="http://lm-studio/v1", judge_revision="qwen-rev")
    monkeypatch.setattr(judge, "is_available", lambda: True)

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": json.dumps({
                "relevance": 0.9,
                "coverage": 0.8,
                "groundedness": 1.0,
                "conflicts": [],
                "missing_aspects": ["minor detail"],
                "pass": True,
            })}}]}

    monkeypatch.setattr("rag_core.eval_golden.requests.post", lambda *args, **kwargs: Response())
    result = judge.evaluate_evidence("q", "answer", ["evidence"])

    assert result["llm_judge_status"] == "completed"
    assert result["pass"] is True
    assert result["judge_model_id"] == "qwen2.5-7b-instruct"
    assert result["judge_revision"] == "qwen-rev"
    assert result["temperature"] == 0
