import json
import logging
from unittest import mock

import pytest

import rag_core.indexer as indexer


def test_load_state_logs_on_corrupt(tmp_path, caplog):
    """techdebt #5: битый state-файл — warning в лог, не молчаливый pass."""
    bad = tmp_path / "state.json"
    bad.write_text("{ невалидный json", encoding="utf-8")
    with mock.patch.object(indexer, "STATE_FILE", str(bad)):
        with caplog.at_level(logging.WARNING):
            result = indexer.load_state()
    assert result == {}
    assert any("load_state" in r.message for r in caplog.records), \
        "load_state не залогировал сбой чтения (techdebt #5 не исправлен)"


def test_parse_frontmatter_logs_on_bad_yaml(caplog):
    """techdebt #5: битый YAML frontmatter — warning, возврат пустых метаданных."""
    bad = "---\n: : :\n---\nтело документа"
    with caplog.at_level(logging.WARNING):
        meta, body = indexer.parse_frontmatter(bad)
    assert meta == {}
    assert body == "тело документа"
    assert any("parse_frontmatter" in r.message for r in caplog.records), \
        "parse_frontmatter не залогировал битый YAML (techdebt #5 не исправлен)"


def test_parse_frontmatter_valid_yaml():
    """Регрессия: валидный YAML парсится как раньше."""
    good = "---\ntitle: Тест\ncategory: wiki\n---\nконтент"
    meta, body = indexer.parse_frontmatter(good)
    assert meta.get("title") == "Тест"
    assert body == "контент"
