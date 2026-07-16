from unittest import mock

import pytest

import unified_searcher


def test_chroma_missing_collection_no_crash():
    """R3: отсутствующая chroma-коллекция не ломает search
    (раньше get_collection бросал исключение наружу)."""
    class _FakeClient:
        def get_collection(self, name):
            raise Exception("CollectionNotFound")

    class _FakeChromadb:
        PersistentClient = staticmethod(lambda path: _FakeClient())

    searcher = unified_searcher.UnifiedSearcher(collection="wiki")
    searcher._backend = "chroma"
    with mock.patch.dict("sys.modules", {"chromadb": _FakeChromadb()}):
        # пересоздаём searcher так, чтобы _ensure_chroma увидел мок
        s2 = unified_searcher.UnifiedSearcher.__new__(unified_searcher.UnifiedSearcher)
        s2._zvec = None
        s2._chroma = None
        s2._collection = "wiki"
        s2._backend = "chroma"
        # мокаем модуль chromadb в sys.modules для импорта внутри _ensure_chroma
        import sys
        sys.modules["chromadb"] = _FakeChromadb()
        try:
            result = s2.search("любой запрос", topk=5)
            assert result == [], f"ожидали [], получили {result}"
        finally:
            del sys.modules["chromadb"]


def test_chroma_missing_returns_none():
    """_ensure_chroma возвращает None при сбое, не бросает."""
    class _FakeClient:
        def get_collection(self, name):
            raise Exception("CollectionNotFound")

    class _FakeChromadb:
        PersistentClient = staticmethod(lambda path: _FakeClient())

    import sys
    sys.modules["chromadb"] = _FakeChromadb()
    try:
        s = unified_searcher.UnifiedSearcher.__new__(unified_searcher.UnifiedSearcher)
        s._zvec = None
        s._chroma = None
        s._collection = "wiki"
        s._backend = "chroma"
        assert s._ensure_chroma() is None
    finally:
        del sys.modules["chromadb"]
