import pytest
from unittest import mock

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


# ── S1-S3 regression: 0.0.0.0/8, IPv4-mapped IPv6, redirect bypass ──

def test_ssrf_blocks_zero_and_mapped():
    """S3: 0.0.0.0/8 and IPv4-mapped IPv6 must be blocked."""
    for url in [
        "http://0.0.0.0/",
        "http://0.0.0.1/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:169.254.169.254]/metadata",
        "http://[::ffff:10.0.0.5]/",
    ]:
        assert rag_async._is_safe_url(url) is False, f"{url} должен быть заблокирован (S3)"


def test_ssrf_blocks_non_global_reserved():
    """S3: reserved/multicast/link-local IPv6 blocked via is_global."""
    # Mock DNS resolution so the test does not hit the network.
    real = rag_async._host_ips
    def fake_host_ips(host):
        try:
            return {ipaddress.ip_address(host)}
        except ValueError:
            return set()
    with mock.patch.object(rag_async, "_host_ips", side_effect=fake_host_ips):
        for url in [
            "http://[fc00::1]/",       # unique local
            "http://[fe80::1]/",       # link-local
            "http://[ff00::1]/",       # multicast
        ]:
            assert rag_async._is_safe_url(url) is False, f"{url} должен быть заблокирован (S3)"


def test_safe_get_blocks_redirect_to_metadata():
    """S1: a public URL that 302-redirects to cloud metadata must NOT be fetched."""
    import requests

    class _FakeResp:
        status_code = 302
        is_redirect = True
        headers = {"Location": "http://169.254.169.254/latest/meta-data/"}

    # First URL is safe (public); redirect target is unsafe (metadata).
    def fake_safe(url):
        return url == "https://example.com/page"

    with mock.patch.object(rag_async, "_is_safe_url", side_effect=fake_safe), \
         mock.patch.object(requests, "get", return_value=_FakeResp()) as mocked:
        resp = rag_async._safe_get("https://example.com/page")
        # No second request to the unsafe redirect target.
        assert resp is None, "redirect to metadata must be blocked"
        # Only the first (safe) request was attempted; no fetch of metadata.
        assert mocked.call_count == 1


def test_safe_get_allows_safe_redirect():
    """S1: a redirect to another public URL is followed (per-hop re-validated)."""
    import requests

    class _Redirect:
        status_code = 301
        is_redirect = True
        headers = {"Location": "https://other.example.com/final"}

    class _Final:
        status_code = 200
        is_redirect = False
        text = "<html>ok</html>"

    calls = {"n": 0}

    def _fake_get(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Redirect()
        return _Final()

    # Treat both example.com hosts as public without hitting DNS.
    with mock.patch.object(rag_async, "_is_safe_url", return_value=True), \
         mock.patch.object(requests, "get", side_effect=_fake_get):
        resp = rag_async._safe_get("https://example.com/start")
        assert resp is not None
        assert resp.text == "<html>ok</html>"
        assert calls["n"] == 2


def test_safe_get_never_fetches_unsafe_direct():
    """S1-S3: a directly-unsafe URL returns None without any network call."""
    import requests

    with mock.patch.object(requests, "get") as mocked:
        resp = rag_async._safe_get("http://169.254.169.254/metadata")
        assert resp is None
        mocked.assert_not_called()
