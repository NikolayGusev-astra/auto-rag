"""Tests for dcd_learner (dev tool).

dcd_learner.read_routing_log() reads routing_log.jsonl from the rag_core dir.
We monkeypatch os.path.join so it points at a temp file with main's format.
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_core"))

import dcd_learner as dl


def test_read_routing_log_parses_main_format(tmp_path, monkeypatch):
    log = tmp_path / "routing_log.jsonl"
    entries = [
        {"query": "альд postgresql", "dcd_domain": "rusbitech",
         "dcd_collection": "rusbitech-products", "dcd_confidence": 0.6,
         "actual_source": "jira", "has_content": True, "chunks_count": 3},
        {"query": "kubernetes helm", "dcd_domain": "devops",
         "dcd_collection": "deployment", "dcd_confidence": 0.7,
         "actual_source": "context7", "has_content": True, "chunks_count": 2},
    ]
    log.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries), encoding="utf-8")

    orig_join = os.path.join
    def fake_join(*a):
        if a and a[-1] == "routing_log.jsonl":
            return str(log)
        return orig_join(*a)
    monkeypatch.setattr(os.path, "join", fake_join)

    records = dl.read_routing_log()
    assert len(records) == 2
    assert records[0]["dcd_domain"] == "rusbitech"
    assert records[1]["source"] == "context7"


def test_extract_keywords_basic():
    kws = dl.extract_keywords("настройка aldpro postgresql replication")
    assert "aldpro" in kws or "postgresql" in kws


def test_analyze_flags_misroute():
    """misrouted: DCD=rusbitech, но source=jira (ожидается context7/lodestone/confluence).

    analyze ожидает поле 'actual_source_domain' — его заполняет main()
    на основе SOURCE_EXPECTED_DOMAINS перед вызовом analyze.
    """
    misrouted = [{
        "dcd_domain": "rusbitech",
        "dcd_collection": "rusbitech-products",
        "source": "jira",
        "actual_source_domain": "rusbitech",  # ожидаемый домен для jira
        "has_content": True,
        "chunks_count": 2,
        "query": "альд pro интеграция",
    }]
    suggestions = dl.analyze(misrouted)
    assert len(suggestions) > 0
    types = {s["type"] for s in suggestions}
    assert "add_keywords" in types or "add_anti_keywords" in types
