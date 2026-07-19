import pytest

from rag_core.gateway.connector import SearchRequest


@pytest.mark.asyncio
async def test_retrieve_merges_connectors_into_evidence():
    from rag_core.gateway.models import Evidence
    from rag_core.gateway.retrieval import retrieve

    class Connector:
        source = "source"

        async def search_live(self, request):
            return [{"document_id": "doc", "text": "body", "score": 0.5}]

    result = await retrieve(SearchRequest(query="query"), {"source": Connector()})

    assert len(result) == 1
    assert isinstance(result[0], Evidence)
    assert result[0].document_id == "doc"


def test_rag_async_remains_importable():
    from rag_core import rag_async

    assert rag_async.LEGACY_PIPELINE is True
