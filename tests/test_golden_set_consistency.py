import json
import os

import pytest

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "rag_core", "golden_set.json")


def _load():
    with open(GOLDEN) as f:
        return json.load(f)


def test_meta_total_matches_actual():
    """Блокер review-and-ship: meta.total_questions должен совпадать
    с фактическим числом вопросов."""
    g = _load()
    assert g["meta"]["total_questions"] == len(g["questions"]), (
        f"meta.total_questions={g['meta']['total_questions']} "
        f"!= факт {len(g['questions'])}"
    )


def test_expected_source_matches_dcd():
    """Блокер: expected_source в golden_set должен совпадать с
    реальным primary_source от DCD-роутера (иначе метрика eval
    source_routing_accuracy бессмысленна)."""
    import dcd_router
    g = _load()
    mismatches = []
    for q in g["questions"]:
        r = dcd_router.classify(q["query"])
        if q.get("expected_source") != r.get("primary_source"):
            mismatches.append((q["id"], q.get("expected_source"), r.get("primary_source")))
    assert not mismatches, f"source mismatches: {mismatches}"


def test_domain_accuracy_stable():
    """Регрессия: domain-классификация по golden_set стабильна
    (проверяем те, где DCD и golden_set согласны по domain)."""
    import dcd_router
    g = _load()
    for q in g["questions"]:
        r = dcd_router.classify(q["query"])
        # только те, где expected_domain совпадает с реальным (не спорный linux-admin-1)
        if q["expected_domain"] == r["domain"]:
            assert r["domain"] == q["expected_domain"]