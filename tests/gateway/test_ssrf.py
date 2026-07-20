"""SSRF security regression tests for CamoufoxConnector (ADR-005 §9)."""
import pytest

from rag_core.gateway.connectors.web_browser import _validate_url, CamoufoxConnector
from rag_core.gateway.models import EvidenceOrigin


class TestSSRF:
    """URL validation gate — no browser needed."""

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1/admin",
        "http://[::1]/admin",
        "http://localhost:8080/secret",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://169.254.169.254/latest",
        "http://0.0.0.0/",
        "http://[fc00::1]/",
        "http://[fe80::1]/",
        "file:///etc/passwd",
        "ftp://example.com",
    ])
    def test_blocked_urls(self, url):
        with pytest.raises(ValueError):
            _validate_url(url, allow_dns=False)

    @pytest.mark.parametrize("url", [
        "https://example.com",
        "http://93.184.216.34",  # example.com IP
        "https://astralinux.ru/page?id=1",
    ])
    def test_allowed_urls(self, url):
        _validate_url(url, allow_dns=False)  # should NOT raise

    def test_decimal_ip_rejected(self):
        # 2130706433 = 127.0.0.1 in decimal
        with pytest.raises(ValueError):
            _validate_url("http://2130706433/", allow_dns=False)

    def test_hostname_no_dns_passes_prefilter(self):
        # Without DNS, hostnames pass the pre-filter but DNS check is skipped
        _validate_url("http://internal.example/", allow_dns=False)

    @pytest.mark.asyncio
    async def test_fetch_blocked_url_returns_blocked_evidence(self):
        connector = CamoufoxConnector(allow_dns=False)
        result = await connector.fetch("http://127.0.0.1:9999/admin")
        assert result.metadata["blocked"] is True
        assert result.text == ""
        assert result.origin == EvidenceOrigin.PUBLIC_WEB
        assert result.retrieval_score == 0.0

    @pytest.mark.asyncio
    async def test_fetch_allowed_url_skips_block(self):
        connector = CamoufoxConnector(allow_dns=False)
        result = await connector.fetch("https://example.com")
        # May fail (no real browser in CI), but must NOT be blocked
        assert result.metadata.get("blocked") is not True
