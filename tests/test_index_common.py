import os
import tempfile

import pytest

import index_common


def test_file_hash_stable():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test\n\ncontent here")
        path = f.name
    try:
        h1 = index_common.file_hash(path)
        h2 = index_common.file_hash(path)
        assert h1 == h2
        assert len(h1) == 32  # md5 hex
    finally:
        os.unlink(path)


def test_parse_frontmatter_valid():
    text = "---\ntitle: X\ncategory: wiki\n---\nтело"
    meta, body = index_common.parse_frontmatter(text)
    assert meta.get("title") == "X"
    assert body == "тело"


def test_parse_frontmatter_no_frontmatter():
    text = "просто текст без метаданных"
    meta, body = index_common.parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_parse_frontmatter_broken_yaml_logged(caplog):
    import logging
    text = "---\n: : :\n---\nтело"
    with caplog.at_level(logging.WARNING):
        meta, body = index_common.parse_frontmatter(text)
    assert meta == {}
    assert body == "тело"
    assert any("parse_frontmatter" in r.message for r in caplog.records)


def test_safe_id_basic():
    sid = index_common._safe_id("wiki/foo.md", "любой контент")
    assert len(sid) <= 64
    assert sid[0] != "_" or sid.startswith("doc")
    # детерминированность
    assert sid == index_common._safe_id("wiki/foo.md", "любой контент")


def test_safe_id_long_truncated():
    long_src = "a" * 100
    sid = index_common._safe_id(long_src, "x")
    assert len(sid) <= 64