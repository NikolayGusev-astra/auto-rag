import json

import pytest

from rag_core.gateway.model_providers import EmbeddingCapabilities
from rag_core.gateway.models import Document, SyncBatch
from rag_core.gateway.sync.engine import EmbeddingProviderUnavailable, SyncEngine


def _document(document_id: str, text: str) -> Document:
    return Document(document_id, "jira", "project", "title", text, content_hash=document_id)


class _Provider:
    @property
    def capabilities(self):
        return EmbeddingCapabilities(
            provider_id="fake",
            model_id="model",
            revision="r",
            local=True,
            offline_capable=True,
            max_batch_size=32,
            dimension=2,
            normalized=True,
            similarity_metric="cosine",
        )

    async def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class _Connector:
    source = "jira"

    def __init__(self, batch: SyncBatch):
        self.batch = batch

    async def sync_changes(self, cursor: str | None) -> SyncBatch:
        return self.batch


def _publish_vector_revision(engine: SyncEngine):
    revision = engine.stage_sync(
        "jira", SyncBatch(added=[_document("jira:1", "first document")]), embed_provider=_Provider()
    )
    engine.publish("jira", revision)
    return revision


@pytest.mark.asyncio
async def test_sync_source_rejects_silent_vector_downgrade(tmp_path):
    engine = SyncEngine(tmp_path)
    first_revision = _publish_vector_revision(engine)
    connector = _Connector(SyncBatch(added=[_document("jira:2", "second document")]))

    with pytest.raises(EmbeddingProviderUnavailable):
        await engine.sync_source(connector)

    assert engine.active_revision("jira") == str(first_revision.path)


@pytest.mark.asyncio
async def test_sync_source_rebuilds_full_vector_snapshot_with_compatible_provider(tmp_path):
    engine = SyncEngine(tmp_path)
    _publish_vector_revision(engine)
    connector = _Connector(SyncBatch(added=[_document("jira:2", "second document")]))

    revision = await engine.sync_source(connector, embed_provider=_Provider())

    assert engine.active_revision("jira") == str(revision.path)
    assert {item["id"] for item in _read_jsonl(revision.path / "docs.jsonl")} == {"jira:1", "jira:2"}
    assert {item["document_id"] for item in _read_jsonl(revision.path / "chunks.jsonl")} == {"jira:1", "jira:2"}
    assert json.loads((revision.path / "lexical.json").read_text(encoding="utf-8"))
    assert {item["document_id"] for item in _read_jsonl(revision.path / "vectors.jsonl")} == {"jira:1", "jira:2"}


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
