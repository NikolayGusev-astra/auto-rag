import pytest

from rag_core.gateway.model_providers import EmbeddingCapabilities
from rag_core.gateway.models import Document, EvidenceOrigin, SyncBatch
from rag_core.gateway.sync.engine import SyncEngine


def _document(document_id: str, text: str) -> Document:
    return Document(document_id, "jira", "project", "Snapshot document", text)


def _publish(engine: SyncEngine, batch: SyncBatch, provider=None) -> None:
    revision = engine.stage_sync("jira", batch, embed_provider=provider)
    engine.publish("jira", revision)


@pytest.mark.asyncio
async def test_searches_published_snapshot_by_lexical_term(tmp_path):
    from rag_core.gateway.connectors.snapshot import LocalSnapshotConnector

    engine = SyncEngine(tmp_path)
    _publish(engine, SyncBatch(added=[_document("jira:fish", "A zebrafish is a freshwater fish.")]))

    results = await LocalSnapshotConnector(engine, "jira").search("zebrafish")

    assert results[0].document_id == "jira:fish"
    assert "zebrafish" in results[0].text.lower()
    assert results[0].origin is EvidenceOrigin.LOCAL_SNAPSHOT
    assert results[0].metadata["chunk_id"] == "jira:fish:0"


@pytest.mark.asyncio
async def test_missing_lexical_term_returns_no_snapshot_results(tmp_path):
    from rag_core.gateway.connectors.snapshot import LocalSnapshotConnector

    engine = SyncEngine(tmp_path)
    _publish(engine, SyncBatch(added=[_document("jira:fish", "A zebrafish is a freshwater fish.")]))

    assert await LocalSnapshotConnector(engine, "jira").search("missingterm") == []


@pytest.mark.asyncio
async def test_query_vector_ranks_snapshot_vectors_by_cosine(tmp_path):
    from rag_core.gateway.connectors.snapshot import LocalSnapshotConnector

    engine = SyncEngine(tmp_path)
    _publish(
        engine,
        SyncBatch(added=[_document("jira:one", "term one"), _document("jira:two", "term two")]),
        _Provider(),
    )

    results = await LocalSnapshotConnector(engine, "jira").search("term", query_vector=[0.0, 1.0])

    assert [result.document_id for result in results] == ["jira:two", "jira:one"]


class _Provider:
    @property
    def capabilities(self):
        return EmbeddingCapabilities(
            provider_id="fake", model_id="model", revision="r", local=True,
            offline_capable=True, max_batch_size=32, dimension=2,
            normalized=True, similarity_metric="cosine",
        )

    async def embed_documents(self, texts):
        return [[1.0, 0.0] if "one" in text else [0.0, 1.0] for text in texts]
