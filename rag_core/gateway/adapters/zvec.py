from __future__ import annotations

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.models import Evidence, EvidenceOrigin
from rag_core.zvec_adapter import ZVecSearcher


class ZvecConnector:
    source = "local_zvec"
    retrieval_kind = "local"

    def __init__(self, zvec: ZVecSearcher | None = None) -> None:
        self._zvec = zvec or ZVecSearcher()

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        records = self._zvec.search_hybrid(
            request.query, topk=request.topk, domain=request.domain
        )
        return [
            Evidence(
                id=record.get("id", f"{record.get('document_id', '?')}#c0"),
                document_id=record.get("document_id", record.get("id", "?")),
                title=record.get("title", ""),
                text=record.get("text", record.get("content", "")),
                source=self.source,
                uri=record.get("uri"),
                origin=EvidenceOrigin.LOCAL_SNAPSHOT,
                retrieval_score=float(record.get("score", 0.0)),
                metadata=record.get("metadata", {}),
            )
            for record in records
        ]

    async def fetch(self, ref: object) -> object:
        raise NotImplementedError("fetch is not implemented for ZVec")

    async def sync_changes(self, cursor: str | None) -> object:
        raise NotImplementedError("sync is not implemented for ZVec")

    async def health(self) -> dict[str, object]:
        return {"source": self.source, "available": True, "detail": "hybrid ready"}
