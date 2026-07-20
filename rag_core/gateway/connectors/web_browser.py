"""Camoufox/Playwright connector — expensive JS-rendering fallback (ADR-005)."""
from __future__ import annotations

import logging

from rag_core.gateway.models import Evidence, EvidenceOrigin

logger = logging.getLogger(__name__)

TIMEOUT_MS = 30_000


class CamoufoxConnector:
    """Headless Chromium via Playwright for JS-heavy pages.

    Resource isolation: separate process, 30s timeout.
    Trafilatura is preferred; this is the expensive fallback.
    """

    source = "web"
    retrieval_kind = "live"

    def __init__(self) -> None:
        self._browser = None

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
        self._browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox",
                  "--disable-dev-shm-usage", "--disable-gpu"],
        )
        return self._browser

    async def fetch(self, url: str) -> Evidence:
        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
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
            return {"source": self.source, "available": True}
        except ImportError:
            return {"source": self.source, "available": False, "detail": "playwright not installed"}
