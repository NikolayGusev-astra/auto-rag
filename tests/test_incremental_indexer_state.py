"""Regression coverage for incremental indexing correctness.

Audit findings:
- changed/deleted files left stale chunks because IDs derive from content;
- failed insert batches were still marked done in the state file.
"""
from pathlib import Path
from unittest import mock

import pytest

import indexer


class _Stats:
    doc_count = 0


class _FakeCollection:
    def __init__(self, fail_insert=False):
        self.deleted_filters = []
        self.inserted = []
        self.fail_insert = fail_insert
        self.stats = _Stats()

    def delete_by_filter(self, expression):
        self.deleted_filters.append(expression)

    def insert(self, docs):
        if self.fail_insert:
            raise RuntimeError("simulated insert failure")
        self.inserted.extend(docs)
        self.stats.doc_count += len(docs)

    def flush(self):
        pass


def _write_doc(root: Path, name: str, text: str):
    path = root / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_incremental_deletes_changed_and_removed_sources(tmp_path, monkeypatch):
    """Changed/removed sources delete their old chunks before insert/state update."""
    docs = tmp_path / "docs"
    docs.mkdir()
    changed = _write_doc(docs, "changed.md", "# Changed\nnew content")

    coll = _FakeCollection()
    old_state = {"changed.md": "old-hash", "removed.md": "old-hash"}
    saved = {}

    monkeypatch.setattr(indexer, "WIKI_PATHS", [str(docs)])
    monkeypatch.setattr(indexer, "COLL_PATH", str(tmp_path / "collection"))
    monkeypatch.setattr(indexer, "collect_files", lambda: [changed])
    monkeypatch.setattr(indexer, "load_state", lambda: old_state)
    monkeypatch.setattr(indexer, "save_state", lambda s: saved.update(s))
    monkeypatch.setattr(indexer, "get_embeddings_batch", lambda texts: [[0.1] * 1024 for _ in texts])

    # Avoid real zvec init/open; inject collection at the open point.
    fake_zvec = mock.MagicMock()
    fake_zvec.init.return_value = None
    fake_zvec.open.return_value = coll
    monkeypatch.setitem(__import__("sys").modules, "zvec", fake_zvec)
    monkeypatch.setattr(indexer.os.path, "exists", lambda p: True if p == indexer.COLL_PATH else Path(p).exists())

    indexer.index(incremental=True, clear=False)

    assert 'source = "changed.md"' in coll.deleted_filters
    assert 'source = "removed.md"' in coll.deleted_filters
    assert "removed.md" not in saved
    assert "changed.md" in saved


def test_incremental_does_not_mark_failed_insert_done(tmp_path, monkeypatch):
    """A batch insert failure leaves that source out of state for retry."""
    docs = tmp_path / "docs"
    docs.mkdir()
    pending = _write_doc(docs, "retry.md", "# Retry\ncontent that must retry")

    coll = _FakeCollection(fail_insert=True)
    saved = {}

    monkeypatch.setattr(indexer, "WIKI_PATHS", [str(docs)])
    monkeypatch.setattr(indexer, "COLL_PATH", str(tmp_path / "collection"))
    monkeypatch.setattr(indexer, "collect_files", lambda: [pending])
    monkeypatch.setattr(indexer, "load_state", lambda: {})
    monkeypatch.setattr(indexer, "save_state", lambda s: saved.update(s))
    monkeypatch.setattr(indexer, "get_embeddings_batch", lambda texts: [[0.1] * 1024 for _ in texts])

    fake_zvec = mock.MagicMock()
    fake_zvec.init.return_value = None
    fake_zvec.open.return_value = coll
    monkeypatch.setitem(__import__("sys").modules, "zvec", fake_zvec)
    monkeypatch.setattr(indexer.os.path, "exists", lambda p: True if p == indexer.COLL_PATH else Path(p).exists())

    indexer.index(incremental=True, clear=False)

    assert "retry.md" not in saved
