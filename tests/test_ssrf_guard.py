import pytest
from unittest import mock

from rag_core import rag_async
from rag_core.security import safe_http


def test_ssrf_blocks_private():
    """Security MEDIUM (SSRF): private/local ranges blocked."""
    for url in [
        "http://127.0.0.1/admin",
        "http://localhost/secret",
        "http://192.168.1.1/router",
        "http://10.0.0.5/internal",
        "http://172.16.5.5/api",
        "http://169.254.169.254/metadata",
        "http://[::1]/x",
    ]:
        assert safe_http.url_targets_public(url) is False, f"{url} должен быть заблокирован"


def test_ssrf_allows_public():
    assert safe_http.url_targets_public("https://example.com/page") is True


def test_ssrf_rejects_garbage():
    assert safe_http.url_targets_public("not-a-url") is False
    assert safe_http.url_targets_public("") is False
    assert safe_http.url_targets_public("ftp:///broken") is False


def test_ssrf_blocks_zero_and_mapped():
    """S3: 0.0.0.0/8 and IPv4-mapped IPv6 must be blocked."""
    for url in [
        "http://0.0.0.0/",
        "http://0.0.0.1/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:169.254.169.254]/metadata",
        "http://[::ffff:10.0.0.5]/",
    ]:
        assert safe_http.url_targets_public(url) is False, f"{url} должен быть заблокирован (S3)"


def test_ssrf_blocks_non_global_reserved():
    """S3: reserved/multicast/link-local IPv6 blocked via is_global."""
    real = safe_http._host_ips

    def fake_host_ips(host):
        try:
            import ipaddress
            return {ipaddress.ip_address(host)}
        except ValueError:
            return set()

    with mock.patch.object(safe_http, "_host_ips", side_effect=fake_host_ips):
        for url in [
            "http://[fc00::1]/",
            "http://[fe80::1]/",
            "http://[ff00::1]/",
        ]:
            assert safe_http.url_targets_public(url) is False, f"{url} должен быть заблокирован (S3)"


def test_safe_get_blocks_redirect_to_metadata():
    """S1: a public URL that 302-redirects to cloud metadata must NOT be fetched."""
    import requests

    class _FakeResp:
        status_code = 302
        is_redirect = True
        headers = {"Location": "http://169.254.169.254/latest/meta-data/"}
        raw = None

    def fake_safe(url):
        return url == "https://example.com/page"

    with mock.patch.object(safe_http, "url_targets_public", side_effect=fake_safe), \
         mock.patch.object(safe_http.requests.Session, "get", return_value=_FakeResp()) as mocked:
        resp = rag_async._safe_get("https://example.com/page")
        assert resp is None, "redirect to metadata must be blocked"
        assert mocked.call_count == 1


def test_safe_get_allows_safe_redirect():
    """S1: a redirect to another public URL is followed (per-hop re-validated)."""
    import requests

    class _Redirect:
        status_code = 301
        is_redirect = True
        headers = {"Location": "https://other.example.com/final"}
        raw = None

    class _Final:
        status_code = 200
        is_redirect = False
        text = "<html>ok</html>"
        _content = b"<html>ok</html>"
        raw = None
        headers = {}

    calls = {"n": 0}

    def _fake_get(self, url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Redirect()
        return _Final()

    with mock.patch.object(safe_http, "url_targets_public", return_value=True), \
         mock.patch.object(safe_http.requests.Session, "get", _fake_get):
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