"""SearXNG web search connector — remote trusted node via SSH tunnel."""
from __future__ import annotations

import logging

import httpx

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.models import Evidence, EvidenceOrigin

logger = logging.getLogger(__name__)


class SearXNGConnector(SourceConnector):
    """SearXNG meta-search on trusted node (autolycus-agent.ru:8080 → localhost:8888)."""

    source = "web"
    retrieval_kind = "live"

    def __init__(self, base_url: str = "http://localhost:8888") -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=20, trust_env=False)

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        try:
            resp = await self._client.get(
                f"{self._base}/search",
                params={"q": request.query, "format": "json", "categories": "general"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("SearXNG search failed: %s", exc)
            return []

        results: list[Evidence] = []
        for item in data.get("results", [])[:request.topk]:
            results.append(Evidence(
                id=f"searxng:{item.get('url','')}",
                document_id=item.get("url", ""),
                title=item.get("title", ""),
                text=item.get("content", ""),
                source=self.source,
                uri=item.get("url"),
                origin=EvidenceOrigin.PUBLIC_WEB,
                retrieval_score=0.6,
                metadata={
                    "url": item.get("url", ""),
                    "engine": ",".join(item.get("engines", [])),
                    "snippet": item.get("content", ""),
                },
            ))
        return results

    async def health(self) -> dict[str, object]:
        try:
            resp = await self._client.get(f"{self._base}/health", timeout=5)
            return {"source": self.source, "available": resp.status_code == 200}
        except Exception:
            return {"source": self.source, "available": False}
