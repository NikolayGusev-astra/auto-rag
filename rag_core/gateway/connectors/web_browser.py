"""Camoufox/Playwright connector — JS-rendering fallback with security isolation (ADR-005).

Security:
- SSRF filter: DNS-resolve hostname, check ALL IPs before navigation
- IPv4/IPv6 private, loopback, link-local, multicast, reserved, unspecified — blocked
- Redirect chain validated per-hop, max_redirects enforced
- Response size capped at MAX_BODY_BYTES
- Scheme allowlist: http/https only
- Sandbox enabled by default, reused in health check
"""
from __future__ import annotations

import ipaddress
import logging
import re
from socket import getaddrinfo, AI_ADDRCONFIG
from urllib.parse import urlparse

from rag_core.gateway.models import Evidence, EvidenceOrigin

logger = logging.getLogger(__name__)

TIMEOUT_MS = 30_000
MAX_BODY_BYTES = 1_048_576
_SCHEME_ALLOWLIST = {"http", "https"}
# Blocklist: localhost, IPv4 private, link-local — fast pre-filter before DNS
_BLOCKED_HOST_RE = re.compile(
    r"^(localhost|127\.|10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|169\.254\.|0\.0\.0\.0|\[::1\]|\[fe80:)"
)


def _is_bad_ip(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _validate_url(url: str, allow_dns: bool = True) -> None:
    """SSRF gate: scheme, hostname, DNS resolution, IP check."""
    parsed = urlparse(url)
    if parsed.scheme not in _SCHEME_ALLOWLIST:
        raise ValueError(f"blocked scheme: {parsed.scheme}")
    hostname = parsed.hostname or ""
    if not hostname:
        raise ValueError(f"no hostname in URL: {url}")

    # Static pre-filter — catches well-known private patterns without DNS
    if _BLOCKED_HOST_RE.match(hostname):
        raise ValueError(f"blocked host pattern: {hostname}")

    # Decimal IP notation (e.g. http://2130706433/ → 127.0.0.1)
    if hostname.isdigit():
        try:
            addr = ipaddress.IPv4Address(int(hostname))
        except (ValueError, ipaddress.AddressValueError):
            addr = None
        if addr is not None and _is_bad_ip(addr):
            raise ValueError(f"blocked decimal IP: {hostname} → {addr}")

    # IP literal check
    addr = _parse_ip(hostname)
    if addr is not None:
        if _is_bad_ip(addr):
            raise ValueError(f"blocked IP: {addr}")
        return  # IP literal validated, no DNS needed

    if not allow_dns:
        return  # DNS disabled in test mode

    # DNS resolution — check ALL resolved addresses
    try:
        infos = getaddrinfo(hostname, None, family=0, type=0, proto=0, flags=AI_ADDRCONFIG)
    except OSError as exc:
        raise ValueError(f"DNS resolution failed for {hostname}: {exc}") from exc

    for family, _type, _proto, _canon, sockaddr in infos:
        ip_str = sockaddr[0]
        ip = _parse_ip(ip_str)
        if ip is None:
            # Skip non-IP results (AF_UNIX, etc.)
            if family_ip := _family_to_ip(family, sockaddr):
                ip = family_ip
            else:
                continue
        if _is_bad_ip(ip):
            raise ValueError(f"DNS resolved {hostname} → blocked IP: {ip}")

    # All resolved IPs passed


def _parse_ip(s: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(s)
    except ValueError:
        return None


def _family_to_ip(family, sockaddr):
    """Extract IP from sockaddr when string parse fails (covers edge cases)."""
    raw = sockaddr[0]
    try:
        if family == 2:  # AF_INET
            return ipaddress.IPv4Address(raw)
        elif family == 10:  # AF_INET6
            return ipaddress.IPv6Address(raw)
    except (ValueError, ipaddress.AddressValueError):
        pass
    return None


class CamoufoxConnector:
    """Headless Chromium via Playwright — expensive fallback.

    Security: SSRF filter (scheme + DNS + IP), redirect chain validation,
    body size cap, sandboxed browser.
    """

    source = "web"
    retrieval_kind = "web"

    def __init__(self, *, sandbox: bool = True, max_redirects: int = 3, allow_dns: bool = True) -> None:
        self._sandbox = sandbox
        self._max_redirects = max_redirects
        self._allow_dns = allow_dns

    # ── browser lifecycle ────────────────────────────────────────

    def _launch_args(self) -> list[str]:
        args = ["--disable-dev-shm-usage", "--disable-gpu"]
        if not self._sandbox:
            args.extend(["--no-sandbox", "--disable-setuid-sandbox"])
        return args

    async def _ensure_browser(self):
        if hasattr(self, "_browser") and self._browser is not None:
            return self._browser
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "playwright not installed: pip install playwright && playwright install chromium"
            )
        pw = await async_playwright().start()
        self._browser = await pw.chromium.launch(
            headless=True, args=self._launch_args() or None
        )
        return self._browser

    # ── fetch ────────────────────────────────────────────────────

    async def fetch(self, url: str) -> Evidence:
        try:
            _validate_url(url, allow_dns=self._allow_dns)
        except ValueError as exc:
            logger.warning("Camoufox blocked %s: %s", url, exc)
            return Evidence(
                id=f"camoufox:{url}", document_id=url, title=url, text="",
                source=self.source, uri=url, origin=EvidenceOrigin.PUBLIC_WEB,
                retrieval_score=0.0,
                metadata={"url": url, "blocked": True, "reason": str(exc)},
            )

        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()
        except Exception as exc:
            return self._error_evidence(url, str(exc))

        redirect_count = 0
        blocked = False

        async def _validate_route(route):
            nonlocal redirect_count, blocked
            req_url = route.request.url
            # SSRF gate for EVERY request (subresources included)
            try:
                _validate_url(req_url, allow_dns=self._allow_dns)
            except ValueError as exc:
                blocked = True
                logger.warning("Camoufox SSRF blocked %s: %s", req_url, exc)
                await route.abort()
                return

            # Redirect counter: only count navigation redirects, not subresources
            if route.request.is_navigation_request():
                redirected_from = route.request.redirected_from
                if redirected_from is not None:
                    redirect_count += 1
                    if self._max_redirects > 0 and redirect_count > self._max_redirects:
                        blocked = True
                        logger.warning("Camoufox max redirects (%d) exceeded for %s",
                                       self._max_redirects, url)
                        await route.abort()
                        return
            await route.continue_()

        try:
            await page.route("**/*", _validate_route)
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
            if blocked:
                return Evidence(
                    id=f"camoufox:{url}", document_id=url, title=url, text="",
                    source=self.source, uri=url, origin=EvidenceOrigin.PUBLIC_WEB,
                    retrieval_score=0.0,
                    metadata={"url": url, "blocked": True, "reason": "redirect to blocked target"},
                )
            if resp is not None and resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}")
            text = await page.inner_text("body")
        finally:
            try:
                await page.close()
            except Exception:
                pass

        text = text.strip()[:MAX_BODY_BYTES]
        return Evidence(
            id=f"camoufox:{url}", document_id=url, title=url[:100],
            text=text, source=self.source, uri=url,
            origin=EvidenceOrigin.PUBLIC_WEB,
            retrieval_score=0.8 if len(text) > 500 else 0.4,
            metadata={"url": url, "extractor": "camoufox", "length": len(text)},
        )

    def _error_evidence(self, url: str, error: str) -> Evidence:
        return Evidence(
            id=f"camoufox:{url}", document_id=url, title=url, text="",
            source=self.source, uri=url, origin=EvidenceOrigin.PUBLIC_WEB,
            retrieval_score=0.0, metadata={"url": url, "error": error},
        )

    async def health(self) -> dict[str, object]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return {"source": self.source, "available": False, "detail": "playwright not installed"}
        try:
            pw = await async_playwright().start()
            browser = await pw.chromium.launch(headless=True, args=self._launch_args() or None)
            await browser.close()
            await pw.stop()
            return {"source": self.source, "available": True}
        except Exception as exc:
            return {"source": self.source, "available": False, "detail": str(exc)}
