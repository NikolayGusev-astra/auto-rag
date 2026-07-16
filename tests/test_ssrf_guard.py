import pytest

from rag_core import rag_async


def test_ssrf_blocks_private():
    """Security MEDIUM (SSRF): приватные/локальные диапазоны блокируются."""
    for url in [
        "http://127.0.0.1/admin",
        "http://localhost/secret",
        "http://192.168.1.1/router",
        "http://10.0.0.5/internal",
        "http://172.16.5.5/api",
        "http://169.254.169.254/metadata",  # cloud metadata
        "http://[::1]/x",
    ]:
        assert rag_async._is_safe_url(url) is False, f"{url} должен быть заблокирован"


def test_ssrf_allows_public():
    """Публичные хосты разрешены (проверка через резолв или DNS)."""
    # example.com резолвится в публичный IP — должен пройти
    assert rag_async._is_safe_url("https://example.com/page") is True


def test_ssrf_rejects_garbage():
    assert rag_async._is_safe_url("not-a-url") is False
    assert rag_async._is_safe_url("") is False
    assert rag_async._is_safe_url("ftp:///broken") is False
