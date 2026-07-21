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

        async def health(self):
            return {"available": True}

        async def search_live(self, request):
            return [_evidence("a", 0.1), _evidence("a", 0.9), _evidence("b", 0.5)]

    results = await RetrievalCoordinator({"local": Connector()}).search(
        SearchRequest(query="q")
    )

    assert [result.document_id for result in results] == ["a", "b"]
    assert results[0].retrieval_score == 0.9


@pytest.mark.asyncio
async def test_unavailable_source_is_skipped_without_error():
    class UnavailableConnector:
        source = "offline"

        async def health(self):
            return {"available": False}

        async def search_live(self, request):
            raise AssertionError("unavailable connector must not be searched")

    results = await RetrievalCoordinator({"offline": UnavailableConnector()}).search(
        SearchRequest(query="q")
    )

    assert results == []


@pytest.mark.asyncio
async def test_coordinator_records_latency():
    class Connector:
        source = "local"

        async def health(self):
            return {"available": True}

        async def search_live(self, request):
            return []

    coordinator = RetrievalCoordinator({"local": Connector()})
    await coordinator.search(SearchRequest(query="q"))

    stage = coordinator.last_latency["local"]
    assert stage["status"] == "completed"
    assert stage["duration_ms"] >= 0


def test_final_score_computed():
    evidence = Evidence(
        id="a#c0", document_id="a", title="t", text="x", source="local",
        origin=EvidenceOrigin.LOCAL_SNAPSHOT, retrieval_score=0.6, reranker_score=0.9,
    )
    fused = RetrievalCoordinator().fuse([evidence])
    assert fused[0].final_score > 0.6


def test_memory_not_dominant_by_similarity():
    memory = Evidence(
        id="m1", document_id="m1", title="t", text="x", source="agent_memory",
        origin=EvidenceOrigin.AGENT_MEMORY, retrieval_score=0.99, final_score=0.99,
    )
    document = Evidence(
        id="d1#c0", document_id="d1", title="t", text="x", source="local",
        origin=EvidenceOrigin.LOCAL_SNAPSHOT, retrieval_score=0.7, final_score=0.7,
    )
    fused = RetrievalCoordinator().fuse([memory, document])
    assert any(item.origin == EvidenceOrigin.LOCAL_SNAPSHOT for item in fused)


@pytest.mark.asyncio
async def test_parallel_partial_failure():
    class FailingConnector:
        source = "bad"

        async def health(self):
            return {"available": True}

        async def search_live(self, request):
            raise RuntimeError("boom")

    class GoodConnector:
        source = "good"

        async def health(self):
            return {"available": True}

        async def search_live(self, request):
            return [_evidence("g", 0.8)]

    coordinator = RetrievalCoordinator({"bad": FailingConnector(), "good": GoodConnector()})
    results = await coordinator.search(SearchRequest(query="q"))

    assert [result.document_id for result in results] == ["g"]
    assert "bad" in coordinator.last_failed_sources
    assert "good" in coordinator.last_successful_sources
