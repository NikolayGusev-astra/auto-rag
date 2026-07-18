
import os
import sys

# Добавляем корень репо в путь, чтобы tests.conftest из проекта шел первым.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from tests.conftest import skip_if_no_chromadb, skip_if_no_embedding
except Exception:
    import pytest as _pytest_mod

    def _has_mod(name):
        try:
            __import__(name)
            return True
        except Exception:
            return False

    def _embedding_available():
        url = os.environ.get("RAG_EMBEDDING_URL", "") or os.environ.get("RAG_MEMVID_EMBED_URL", "")
        model = os.environ.get("RAG_EMBEDDING_MODEL", "") or os.environ.get("RAG_MEMVID_EMBED_MODEL", "")
        return bool(url) and bool(model)

    skip_if_no_chromadb = _pytest_mod.mark.skipif(
        not _has_mod("chromadb"), reason="chromadb not installed"
    )
    skip_if_no_embedding = _pytest_mod.mark.skipif(
        not _embedding_available(), reason="no embedding service configured"
    )

from unittest import mock

import pytest

from rag_core import unified_searcher


class _FakeClient:
    def get_collection(self, name):
        raise Exception("CollectionNotFound")


class _FakeChromadb:
    @staticmethod
    def PersistentClient(path=None):
        return _FakeClient()


def _make_searcher():
    s = unified_searcher.UnifiedSearcher.__new__(unified_searcher.UnifiedSearcher)
    s._zvec = None
    s._chroma = None
    s._collection = "wiki"
    s._backend = "chroma"
    return s


def test_chroma_missing_returns_none():
    """_ensure_chroma возвращает None при сбое get_collection (R3)."""
    with mock.patch.dict("sys.modules", {"chromadb": _FakeChromadb()}):
        s = _make_searcher()
        assert s._ensure_chroma() is None


@skip_if_no_chromadb
@skip_if_no_embedding
def test_chroma_missing_collection_no_crash():
    """R3: отсутствующая chroma-коллекция не ломает search —
    _ensure_chroma ловит исключение, search корректно возвращает []."""
    with mock.patch.dict("sys.modules", {"chromadb": _FakeChromadb()}), \
         mock.patch("unified_searcher._get_embedding", return_value=[1.0] * 1024):
        s = _make_searcher()
        result = s.search("любой запрос", topk=5)
        assert result == [], f"ожидали [], получили {result}"
