import pytest

from rag_core.gateway.adapters.zvec import ZvecConnector
from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.models import EvidenceOrigin


@pytest.mark.asyncio
async def test_zvec_connector_returns_evidence():
    class FakeZvec:
        def search_hybrid(self, query, topk=5, domain=None):
            return [{
                "id": "doc1#c0",
                "document_id": "doc1",
                "title": "T",
                "text": "body",
                "score": 0.9,
                "uri": None,
            }]

    results = await ZvecConnector(zvec=FakeZvec()).search_live(
        SearchRequest(query="q", topk=1)
    )

    assert len(results) == 1
    assert results[0].document_id == "doc1"
    assert results[0].origin == EvidenceOrigin.LOCAL_SNAPSHOT
    assert results[0].retrieval_score == 0.9
