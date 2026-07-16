from unittest import mock

import pytest

import unified_searcher


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


def test_chroma_missing_collection_no_crash():
    """R3: отсутствующая chroma-коллекция не ломает search —
    _ensure_chroma ловит исключение, search корректно возвращает []."""
    with mock.patch.dict("sys.modules", {"chromadb": _FakeChromadb()}), \
         mock.patch("unified_searcher._get_embedding", return_value=[1.0] * 1024):
        s = _make_searcher()
        result = s.search("любой запрос", topk=5)
        assert result == [], f"ожидали [], получили {result}"
