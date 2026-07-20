"""Trafilatura web extraction connector."""
from __future__ import annotations

import logging

import httpx
import trafilatura

from rag_core.gateway.models import Evidence, EvidenceOrigin

logger = logging.getLogger(__name__)

MAX_CONTENT_BYTES = 50_000


class WebExtractConnector:
    """Downloads and extracts text from web pages via Trafilatura."""

    source = "web"
    retrieval_kind = "web"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=15, trust_env=False,
                                          follow_redirects=True,
                                          headers={"User-Agent": "AutoRAG/1.0"})

    async def fetch(self, url: str) -> Evidence:
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            content = resp.content[:MAX_CONTENT_BYTES]
            extracted = trafilatura.extract(content, include_formatting=False,
                                            include_links=False, favor_precision=True)
            text = extracted or ""
            if len(text) < 200:
                return Evidence(
                    id=f"web-extract:{url}", document_id=url, title=url, text=text,
                    source=self.source, uri=url, origin=EvidenceOrigin.PUBLIC_WEB,
                    retrieval_score=0.3, metadata={"url": url, "quality": "low"},
                )
            return Evidence(
                id=f"web-extract:{url}", document_id=url, title=url[:100], text=text[:10_000],
                source=self.source, uri=url, origin=EvidenceOrigin.PUBLIC_WEB,
                retrieval_score=0.7, metadata={"url": url, "length": len(text)},
            )
        except Exception as exc:
            logger.warning("Trafilatura extraction failed for %s: %s", url, exc)
            return Evidence(
                id=f"web-extract:{url}", document_id=url, title=url, text="",
                source=self.source, uri=url, origin=EvidenceOrigin.PUBLIC_WEB,
                retrieval_score=0.0, metadata={"url": url, "error": str(exc)},
            )

    async def health(self) -> dict[str, object]:
        return {"source": self.source, "available": True}
