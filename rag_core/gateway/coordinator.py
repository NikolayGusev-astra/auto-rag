from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.models import Evidence


EvidenceFilter = Callable[[Evidence], bool]


class RetrievalCoordinator:
    def __init__(
        self,
        connectors: Mapping[str, SourceConnector] | None = None,
        filters: Iterable[EvidenceFilter] = (),
        reranker: object | None = None,
    ) -> None:
        self._connectors = dict(connectors or {})
        self._filters = tuple(filters)
        self._reranker = reranker

    async def search(self, request: SearchRequest) -> list[Evidence]:
        evidence = []
        for connector in self._connectors.values():
            evidence.extend(await connector.search_live(request))
        fused = self.fuse(evidence)
        if self._reranker is not None:
            fused = await self._reranker.rerank(request.query, fused, request.topk)
        return fused[:request.topk]

    def fuse(self, evidence: Iterable[Evidence]) -> list[Evidence]:
        best_by_document: dict[str, Evidence] = {}
        for item in evidence:
            if item.metadata.get("deprecated") or any(
                not evidence_filter(item) for evidence_filter in self._filters
            ):
                continue
            current = best_by_document.get(item.document_id)
            if current is None or item.retrieval_score > current.retrieval_score:
                best_by_document[item.document_id] = item
        return sorted(
            best_by_document.values(), key=lambda item: item.retrieval_score, reverse=True
        )
