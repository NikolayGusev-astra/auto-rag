"""Regression test for C4: read-path open failure must not wipe the index.

zvec_adapter.ZVecSearcher._ensure_collection used to `shutil.rmtree` the
collection on ANY zvec.open() exception (lock contention, SDK version skew,
OOM). A single transient open error silently destroyed the whole knowledge
base. Now it logs and returns None so the caller degrades to empty results.
"""
import sys
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("zvec")

import zvec_adapter


class _FakeZvec:
    """Minimal zvec stub whose open() raises on demand."""

    def __init__(self, open_side_effect):
        self._open_side_effect = open_side_effect

    def open(self, path):  # noqa: D401 - matches zvec.open signature shape
        raise self._open_side_effect

    def init(self):
        return None


def _patch_zvec(open_side_effect):
    """Patch the `import zvec` inside _ensure_collection."""
    fake = _FakeZvec(open_side_effect)
    return mock.patch.dict(sys.modules, {"zvec": fake})


def test_open_failure_preserves_collection(tmp_path):
    """A failing zvec.open() on an existing collection must NOT rmtree it."""
    coll = tmp_path / "wiki"
    coll.mkdir()
    sentinel = coll / "do_not_delete.txt"
    sentinel.write_text("knowledge base contents")

    searcher = zvec_adapter.ZVecSearcher(collection="wiki")
    searcher.coll_path = str(coll)

    rmtree_calls = []

    def fake_rmtree(path, ignore_errors=False):
        rmtree_calls.append(str(path))

    with mock.patch.object(shutil, "rmtree", side_effect=fake_rmtree), \
         _patch_zvec(RuntimeError("lock busy")):
        result = searcher._ensure_collection()

    assert result is None, "open failure should return None, not a wiped/recreated collection"
    assert not rmtree_calls, f"collection was deleted on open failure: {rmtree_calls}"
    assert sentinel.exists(), "knowledge base was destroyed by a read-path open error"
    assert searcher._coll is None


def test_missing_collection_open_failure_is_safe(tmp_path):
    """A failing zvec.open() when the collection is missing must not raise."""
    coll = tmp_path / "wiki"  # does NOT exist

    searcher = zvec_adapter.ZVecSearcher(collection="wiki")
    searcher.coll_path = str(coll)

    with _patch_zvec(RuntimeError("no such collection")):
        result = searcher._ensure_collection()

    assert result is None
    assert searcher._coll is None
