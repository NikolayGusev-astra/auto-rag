import pytest
import httpx

from rag_core.gateway.models import Evidence, EvidenceOrigin
from rag_core.gateway.rerank_adapter import RerankAdapter


@pytest.mark.asyncio
async def test_rerank_orders_evidence_by_cosine_score():
    class EmbeddingProvider:
        async def embed_query(self, text):
            return {
                "query": [1.0, 0.0],
                "less_relevant": [0.0, 1.0],
                "most_relevant": [3.0, 0.0],
            }[text]

    documents = [
        Evidence(
            id="less", document_id="less", title="Less", text="less_relevant", source="local",
            origin=EvidenceOrigin.LOCAL_SNAPSHOT,
        ),
        Evidence(
            id="most", document_id="most", title="Most", text="most_relevant", source="local",
            origin=EvidenceOrigin.LOCAL_SNAPSHOT,
        ),
    ]

    results = await RerankAdapter(EmbeddingProvider()).rerank("query", documents, top_k=1)

    assert [result.document_id for result in results] == ["most"]
    assert results[0].reranker_score == 1.0


@pytest.mark.asyncio
async def test_rerank_keeps_retrieval_results_when_embedding_service_is_unavailable():
    class UnavailableEmbeddingProvider:
        async def embed_query(self, text):
            raise httpx.ConnectError("LM Studio is unavailable")

    document = Evidence(
        id="doc", document_id="doc", title="Document", text="body", source="local",
        origin=EvidenceOrigin.LOCAL_SNAPSHOT,
    )

    results = await RerankAdapter(UnavailableEmbeddingProvider()).rerank("query", [document], top_k=1)

    assert results == [document]


@pytest.mark.asyncio
async def test_rerank_does_not_embed_when_there_are_no_documents():
    class EmbeddingProvider:
        async def embed_query(self, text):
            raise AssertionError("empty result sets should not call the embedding service")

    assert await RerankAdapter(EmbeddingProvider()).rerank("query", [], top_k=1) == []


@pytest.mark.asyncio
async def test_rerank_propagates_programming_errors():
    """Programming errors (AttributeError, TypeError) must surface, not be masked."""
    class BuggyProvider:
        async def embed_query(self, text):
            raise AttributeError("provider is misconfigured — should not be masked")

    document = Evidence(
        id="doc", document_id="doc", title="Document", text="body", source="local",
        origin=EvidenceOrigin.LOCAL_SNAPSHOT,
    )

    with pytest.raises(AttributeError, match="provider is misconfigured"):
        await RerankAdapter(BuggyProvider()).rerank("query", [document], top_k=1)
