import importlib

import pytest


def _load():
    import memvid_memory
    return memvid_memory


def test_module_imports_clean():
    """techdebt #1: испорченный дубликат _attr удалён, модуль
    импортируется без SyntaxError/NameError."""
    m = _load()
    assert hasattr(m, "_attr")
    assert hasattr(m, "_cosine")


def test_attr_works():
    """_attr резолвит attr/dict после удаления сломанного дубликата."""
    m = _load()
    class _O:
        def __init__(self): self.x = 42
    assert m._attr(_O(), "x") == 42
    assert m._attr({"y": 7}, "y") == 7
    assert m._attr({"z": 1}, "missing") is None
    assert m._attr("не объект", "x") is None