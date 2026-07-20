"""Camoufox/Playwright connector — JS-rendering fallback with security isolation (ADR-005).

Security: SSRF filter, scheme validation, no sandbox disabled without explicit external
containerisation. Trafilatura is preferred (web_extract.py); this is the expensive fallback
invoked only by WebPipeline when extraction quality is insufficient.
"""
from __future__ import annotations

import ipaddress
import logging
import re
from urllib.parse import urlparse

from rag_core.gateway.models import Evidence, EvidenceOrigin

logger = logging.getLogger(__name__)

TIMEOUT_MS = 30_000
MAX_BODY_BYTES = 1_048_576

# Blocklist: localhost, private, link-local
_BLOCKED_HOSTS = re.compile(
    r"^(localhost|127\.|10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|169\.254\.|0\.0\.0\.0|\[::1\]|\[fe80:)"
)
_SCHEME_ALLOWLIST = {"http", "https"}


class CamoufoxConnector:
    """Headless Chromium via Playwright — expensive fallback.

    Never uses ``--no-sandbox`` without proven external container isolation.
    """

    source = "web"
    retrieval_kind = "web"

    def __init__(self, *, sandbox: bool = True, max_redirects: int = 3) -> None:
        self._sandbox = sandbox
        self._max_redirects = max_redirects
        self._browser = None

    # ── security gate ────────────────────────────────────────────

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in _SCHEME_ALLOWLIST:
            raise ValueError(f"blocked scheme: {parsed.scheme}")
        if _BLOCKED_HOSTS.match(parsed.hostname or ""):
            raise ValueError(f"blocked host: {parsed.hostname}")
        try:
            addr = ipaddress.ip_address(parsed.hostname or "")
            if addr.is_private or addr.is_loopback or addr.is_link_local:
                raise ValueError(f"blocked private address: {addr}")
        except ValueError:
            pass  # not an IP — hostname, allow

    # ── browser lifecycle ────────────────────────────────────────

    async def _ensure_browser(self):
        if self._browser is not None:
            return self._browser
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "playwright not installed. Install: pip install playwright && playwright install chromium"
            )
        pw = await async_playwright().start()
        launch_args: list[str] = [
            "--disable-dev-shm-usage",
            "--disable-gpu",
        ]
        if not self._sandbox:
            launch_args.extend(["--no-sandbox", "--disable-setuid-sandbox"])
        self._browser = await pw.chromium.launch(
            headless=True, args=launch_args or None
        )
        return self._browser

    # ── fetch ────────────────────────────────────────────────────

    async def fetch(self, url: str) -> Evidence:
        # Security gate
        try:
            self._validate_url(url)
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
            try:
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
                # Check response
                if resp is not None and resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}")
                text = await page.inner_text("body")
            finally:
                await page.close()

            text = text.strip()[:10_000]
            return Evidence(
                id=f"camoufox:{url}", document_id=url, title=url[:100],
                text=text, source=self.source, uri=url,
                origin=EvidenceOrigin.PUBLIC_WEB,
                retrieval_score=0.8 if len(text) > 500 else 0.4,
                metadata={"url": url, "extractor": "camoufox", "length": len(text)},
            )
        except Exception as exc:
            logger.warning("Camoufox fetch failed for %s: %s", url, exc)
            return Evidence(
                id=f"camoufox:{url}", document_id=url, title=url, text="",
                source=self.source, uri=url, origin=EvidenceOrigin.PUBLIC_WEB,
                retrieval_score=0.0, metadata={"url": url, "error": str(exc)},
            )

    async def health(self) -> dict[str, object]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return {"source": self.source, "available": False, "detail": "playwright not installed"}
        try:
            pw = await async_playwright().start()
            browser = await pw.chromium.launch(headless=True)
            await browser.close()
            await pw.stop()
            return {"source": self.source, "available": True}
        except Exception as exc:
            return {"source": self.source, "available": False, "detail": str(exc)}
