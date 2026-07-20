"""Web research pipeline — search → extract → Camoufox fallback (ADR-005)."""
from __future__ import annotations

import logging

from rag_core.gateway.models import Evidence, EvidenceOrigin

logger = logging.getLogger(__name__)

QUALITY_MIN_CHARS = 200


class WebPipeline:
    """Orchestrates: web search → Trafilatura extraction → Camoufox fallback.

    Web research is OFF by default (allow_web=False per ADR-005).
    """

    def __init__(
        self,
        allow_web: bool = False,
        search: object | None = None,
        extract: object | None = None,
        browser: object | None = None,
    ) -> None:
        self._allow = allow_web
        self._search = search
        self._extract = extract
        self._browser = browser

    async def research(self, query: str, topk: int = 5) -> list[Evidence]:
        if not self._allow:
            return []
        if self._search is None:
            return []

        # 1. Search → get 3× results
        from rag_core.gateway.connector import SearchRequest
        try:
            candidates = await self._search.search_live(
                SearchRequest(query=query, topk=topk * 3, include_web=True)
            )
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            return []

        # 2. Extract each result
        results: list[Evidence] = []
        for item in candidates:
            url = item.uri or item.document_id
            if not url or not url.startswith("http"):
                continue

            evidence = await self._extract_url(url)
            if evidence is not None:
                # Preserve original title from search result
                if item.title and not evidence.title.startswith("http"):
                    evidence = Evidence(
                        id=evidence.id, document_id=evidence.document_id,
                        title=item.title, text=evidence.text,
                        source=evidence.source, uri=evidence.uri,
                        origin=evidence.origin, retrieval_score=evidence.retrieval_score,
                        metadata=evidence.metadata,
                    )
                results.append(evidence)

        # 3. Return top-k
        results.sort(key=lambda e: e.retrieval_score or 0.0, reverse=True)
        return results[:topk]

    async def _extract_url(self, url: str) -> Evidence | None:
        if self._extract is None:
            return None

        # Try Trafilatura
        evidence = await self._extract.fetch(url)
        text = evidence.text.strip() if evidence.text else ""

        # Quality check: if too short → Camoufox fallback
        if len(text) < QUALITY_MIN_CHARS and self._browser is not None:
            logger.info("Trafilatura short (%d chars), trying Camoufox for %s", len(text), url)
            try:
                fallback = await self._browser.fetch(url)
                if fallback.text and len(fallback.text.strip()) >= len(text):
                    return fallback
            except Exception as exc:
                logger.warning("Camoufox fallback failed for %s: %s", url, exc)

        return evidence if len(text) >= 50 else None
