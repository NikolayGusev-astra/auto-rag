import pytest

from rag_core.gateway.adapters.memory import MemoryConnector
from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.coordinator import RetrievalCoordinator
from rag_core.gateway.models import Evidence, EvidenceOrigin


@pytest.mark.asyncio
async def test_memory_tagged_agent_memory_and_not_short_circuited():
    class LocalConnector:
        source = "local"

        async def health(self):
            return {"available": True}

        async def search_live(self, request):
            return [
                Evidence(
                    id="local:0",
                    document_id="local:0",
                    title="local",
                    text="live result",
                    source=self.source,
                    origin=EvidenceOrigin.LOCAL_SNAPSHOT,
                    retrieval_score=0.8,
                )
            ]

    results = await RetrievalCoordinator(
        {
            "memory": MemoryConnector(episodes=[{"answer": "cached", "score": 0.9}]),
            "local": LocalConnector(),
        }
    ).search(SearchRequest(query="q"))

    assert results[0].origin == EvidenceOrigin.AGENT_MEMORY
    assert [result.text for result in results] == ["cached", "live result"]


def test_memory_connector_exposes_memory_evidence():
    connector = MemoryConnector(episodes=[{
        "answer": "cached", "score": 0.9, "document_ids": ["d1"],
        "source_uris": ["u1"], "route": ["local"], "episode_id": "e1",
    }])
    evidence = connector.as_memory_evidence(0)
    assert evidence.episode_id == "e1"
    assert evidence.source_document_ids == ("d1",)
    assert evidence.embedding_profile_id is None


def test_memory_skipped_on_profile_mismatch():
    connector = MemoryConnector(episodes=[{
        "episode_id": "e1", "score": 0.9, "embedding_profile_id": "prof-B",
    }])
    assert connector.is_compatible("prof-A") is False
