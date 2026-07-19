import pytest

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.coordinator import RetrievalCoordinator
from rag_core.gateway.models import Evidence, EvidenceOrigin


def _evidence(document_id: str, score: float = 0.5) -> Evidence:
    return Evidence(
        id=f"{document_id}#c0",
        document_id=document_id,
        title="title",
        text="body",
        source="local",
        origin=EvidenceOrigin.LOCAL_SNAPSHOT,
        retrieval_score=score,
    )


@pytest.mark.asyncio
async def test_dedup_by_document_id_works():
    class Connector:
        source = "local"

        async def search_live(self, request):
            return [_evidence("a", 0.1), _evidence("a", 0.9), _evidence("b", 0.5)]

    results = await RetrievalCoordinator({"local": Connector()}).search(
        SearchRequest(query="q")
    )

    assert [result.document_id for result in results] == ["a", "b"]
    assert results[0].retrieval_score == 0.9
