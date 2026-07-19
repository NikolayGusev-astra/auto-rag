import asyncio
import json
import math

import pytest

from rag_core.gateway.model_providers import EmbeddingCapabilities, EmbeddingProfile
from rag_core.gateway.models import Document, SyncBatch
from rag_core.gateway.sync.engine import EmbeddingProviderUnavailable, SyncEngine


def _document(document_id="jira:1", text="audit document"):
    return Document(document_id, "jira", "p", "title", text, content_hash="h")


def _profile(dimension=768):
    return EmbeddingProfile("sentence-transformers", "model", "r", dimension, True, "cosine", "query-passages-v1")


class _AsyncProvider:
    def __init__(self, dimension=768, vector_dimension=None, value=1.0):
        self.dimension = dimension
        self.vector_dimension = vector_dimension or dimension
        self.value = value

    @property
    def capabilities(self):
        return EmbeddingCapabilities(
            provider_id="fake", model_id="model", revision="r", local=True,
            offline_capable=True, max_batch_size=32, dimension=self.dimension,
            normalized=True, similarity_metric="cosine",
        )

    async def embed_documents(self, texts):
        await asyncio.sleep(0)
        return [[self.value] * self.vector_dimension for _ in texts]


def _publish(engine, batch, provider=None, profile=None):
    revision = engine.stage_sync("jira", batch, embed_provider=provider, active_profile=profile)
    engine.publish("jira", revision)
    return revision


def test_vector_profile_without_provider_rejects_and_keeps_active_revision(tmp_path):
    engine = SyncEngine(tmp_path)
    old = _publish(engine, SyncBatch(added=[_document("jira:old")]))

    with pytest.raises(EmbeddingProviderUnavailable):
        engine.stage_sync("jira", SyncBatch(added=[_document("jira:new")]), active_profile=_profile())

    assert engine.active_revision("jira") == str(old.path)


def test_explicit_lexical_downgrade_is_published(tmp_path):
    engine = SyncEngine(tmp_path)
    revision = engine.stage_sync(
        "jira", SyncBatch(added=[_document()]), active_profile=_profile(), allow_lexical_downgrade=True
    )
    engine.publish("jira", revision)

    assert not (revision.path / "vectors.jsonl").exists()
    assert json.loads((revision.path / "manifest.json").read_text())["embedding_profile"] is None


@pytest.mark.asyncio
async def test_async_provider_inside_running_loop_writes_vectors_without_failures(tmp_path):
    engine = SyncEngine(tmp_path)
    revision = await engine.stage_sync_async(
        "jira", SyncBatch(added=[_document()]), embed_provider=_AsyncProvider(), active_profile=_profile()
    )

    assert len((revision.path / "vectors.jsonl").read_text().splitlines()) == 1
    assert json.loads((revision.path / "manifest.json").read_text())["embedding_failures"] == []


def test_full_rebuild_accepts_provider_and_writes_all_index_artifacts(tmp_path):
    engine = SyncEngine(tmp_path)
    revision = engine.full_rebuild(
        "jira", SyncBatch(added=[_document()]), embed_provider=_AsyncProvider(), active_profile=_profile()
    )

    assert engine.active_revision("jira") == str(revision.path)
    assert all((revision.path / name).exists() for name in ("docs.jsonl", "chunks.jsonl", "lexical.json", "vectors.jsonl"))


@pytest.mark.asyncio
async def test_wrong_vector_dimension_fails_without_changing_manifest(tmp_path):
    engine = SyncEngine(tmp_path)
    old = _publish(engine, SyncBatch(added=[_document("jira:old")]))
    manifest = (tmp_path / "jira" / "manifest.json").read_text()

    with pytest.raises(ValueError, match="dimension"):
        await engine.stage_sync_async(
            "jira", SyncBatch(added=[_document("jira:new")]),
            embed_provider=_AsyncProvider(dimension=768, vector_dimension=384), active_profile=_profile(),
        )

    assert engine.active_revision("jira") == str(old.path)
    assert (tmp_path / "jira" / "manifest.json").read_text() == manifest


@pytest.mark.asyncio
async def test_nan_vector_makes_revision_invalid(tmp_path):
    engine = SyncEngine(tmp_path)

    with pytest.raises(ValueError, match="finite"):
        await engine.stage_sync_async(
            "jira", SyncBatch(added=[_document()]),
            embed_provider=_AsyncProvider(value=math.nan), active_profile=_profile(),
        )
