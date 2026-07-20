"""DuckDuckGo web search connector."""
from __future__ import annotations

import asyncio
import logging

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.models import Evidence, EvidenceOrigin

logger = logging.getLogger(__name__)


class WebSearchConnector(SourceConnector):
    """DuckDuckGo instant answers + web results. Rate-limited, offline-safe."""

    source = "web"
    retrieval_kind = "web"

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("duckduckgo_search not installed; web search unavailable")
            return []

        results: list[Evidence] = []
        try:
            with DDGS() as ddgs:
                for item in ddgs.text(request.query, max_results=request.topk):
                    results.append(Evidence(
                        id=f"web:{item.get('href','')}",
                        document_id=item.get("href", ""),
                        title=item.get("title", ""),
                        text=item.get("body", ""),
                        source=self.source,
                        uri=item.get("href"),
                        origin=EvidenceOrigin.PUBLIC_WEB,
                        retrieval_score=0.5,
                        metadata={"url": item.get("href", ""), "snippet": item.get("body", "")},
                    ))
            await asyncio.sleep(1.0)  # rate-limit
        except Exception as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)

        return results

    async def health(self) -> dict[str, object]:
        return {"source": self.source, "available": True}
