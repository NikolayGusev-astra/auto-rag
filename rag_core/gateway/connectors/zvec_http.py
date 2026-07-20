"""HTTP connector for the long-lived local ZVec search server."""
from __future__ import annotations

import httpx

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.models import Evidence, EvidenceOrigin


class ZVecHttpConnector(SourceConnector):
    """Retrieve from ZVec without opening its collection in the gateway process."""

    source = "zvec"
    retrieval_kind = "local"

    def __init__(self, base_url: str = "http://127.0.0.1:8678") -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30, trust_env=False)

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        response = await self._client.get(
            f"{self._base}/search", params={"q": request.query, "topk": request.topk}
        )
        data = response.json()
        return [
            Evidence(
                id=chunk_id,
                document_id=chunk_id,
                title="",
                text=chunk["text"],
                source=self.source,
                retrieval_score=chunk["score"],
                origin=EvidenceOrigin.LOCAL_SNAPSHOT,
            )
            for index, chunk in enumerate(data["chunks"])
            for chunk_id in (str(chunk.get("id") or chunk.get("source") or f"zvec:{index}"),)
        ]

    async def health(self) -> dict[str, bool]:
        try:
            response = await self._client.get(f"{self._base}/health")
        except httpx.HTTPError:
            return {"available": False}
        return {"available": response.status_code == 200}
