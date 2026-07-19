import json

import pytest

from rag_core.gateway.model_providers import EmbeddingCapabilities, EmbeddingProfile
from rag_core.gateway.models import Document, SyncBatch
from rag_core.gateway.sync.engine import SyncEngine
from rag_core.gateway.sync.index_builder import (
    build_lexical_index,
    chunk,
    validate_profile,
)


def _document(document_id, text, content_hash="h"):
    return Document(document_id, "jira", "p", "title", text, content_hash=content_hash)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _publish(engine, batch, provider=None, active_profile=None):
    revision = engine.stage_sync("jira", batch, embed_provider=provider, active_profile=active_profile)
    engine.publish("jira", revision)
    return revision


def test_chunking_is_stable_and_handles_empty_and_oversized_documents():
    empty = _document("jira:empty", "")
    oversized = _document("jira:large", "x" * 2500)

    assert chunk(empty) == []
    first = chunk(oversized)
    assert [item["id"] for item in first] == [item["id"] for item in chunk(oversized)]
    assert len(first) > 1
    assert all(len(item["text"]) <= 1000 for item in first)


def test_lexical_index_maps_normalized_terms_to_chunk_ids():
    chunks = chunk(_document("jira:1", "Alpha beta. alpha!"))

    index = build_lexical_index(chunks)

    assert index["alpha"] == {chunks[0]["id"]}
    assert index["beta"] == {chunks[0]["id"]}


def test_changed_document_replaces_old_chunks_and_reindex_never_duplicates(tmp_path):
    engine = SyncEngine(tmp_path)
    _publish(engine, SyncBatch(added=[_document("jira:1", "oldterm", "v1")]))
    revised = _publish(engine, SyncBatch(changed=[_document("jira:1", "newterm", "v2")]))

    chunks = _read_jsonl(revised.path / "chunks.jsonl")
    lexical = json.loads((revised.path / "lexical.json").read_text(encoding="utf-8"))
    assert [item["document_id"] for item in chunks] == ["jira:1"]
    assert "oldterm" not in lexical and "newterm" in lexical

    same = _publish(engine, SyncBatch(changed=[_document("jira:1", "newterm", "v2")]))
    assert _read_jsonl(same.path / "chunks.jsonl") == chunks


def test_deleted_document_removes_docs_chunks_lexical_and_vectors(tmp_path):
    engine = SyncEngine(tmp_path)
    provider = _Provider()
    _publish(engine, SyncBatch(added=[_document("jira:1", "keep"), _document("jira:2", "remove")]), provider)
    revision = _publish(engine, SyncBatch(deleted=["jira:2"]), provider)

    assert [item["id"] for item in _read_jsonl(revision.path / "docs.jsonl")] == ["jira:1"]
    assert {item["document_id"] for item in _read_jsonl(revision.path / "chunks.jsonl")} == {"jira:1"}
    assert "remove" not in json.loads((revision.path / "lexical.json").read_text(encoding="utf-8"))
    assert {item["document_id"] for item in _read_jsonl(revision.path / "vectors.jsonl")} == {"jira:1"}
    assert json.loads((revision.path / "manifest.json").read_text(encoding="utf-8"))["embedding_profile"]["model_id"] == "model"


def test_no_embedding_provider_still_builds_lexical_index(tmp_path):
    engine = SyncEngine(tmp_path)
    revision = _publish(engine, SyncBatch(added=[_document("jira:1", "offline lexical")]))

    assert (revision.path / "lexical.json").is_file()
    assert not (revision.path / "vectors.jsonl").exists()
    assert json.loads((revision.path / "manifest.json").read_text(encoding="utf-8"))["embedding_profile"] is None


def test_incompatible_profile_stops_publish_before_manifest_swap(tmp_path):
    engine = SyncEngine(tmp_path)
    old = _publish(engine, SyncBatch(added=[_document("jira:1", "old")]))
    incompatible = EmbeddingProfile("fake", "other", "r", 2, True, "cosine", "p")

    with pytest.raises(ValueError):
        engine.stage_sync("jira", SyncBatch(added=[_document("jira:2", "new")]), _Provider(), incompatible)

    assert engine.active_revision("jira") == str(old.path)


def test_partial_embedding_failure_keeps_lexical_snapshot_and_flags_document(tmp_path):
    engine = SyncEngine(tmp_path)
    revision = _publish(engine, SyncBatch(added=[_document("jira:good", "good"), _document("jira:bad", "bad")]), _Provider(fail_text="bad"))

    manifest = json.loads((revision.path / "manifest.json").read_text(encoding="utf-8"))
    assert {item["id"] for item in _read_jsonl(revision.path / "docs.jsonl")} == {"jira:good", "jira:bad"}
    assert "good" in json.loads((revision.path / "lexical.json").read_text(encoding="utf-8"))
    assert manifest["embedding_failures"] == ["jira:bad"]
    assert {item["document_id"] for item in _read_jsonl(revision.path / "vectors.jsonl")} == {"jira:good"}


def test_staged_revision_is_full_snapshot_and_corrupt_index_blocks_publish(tmp_path):
    engine = SyncEngine(tmp_path)
    _publish(engine, SyncBatch(added=[_document("jira:prior", "prior")]))
    old = engine.active_revision("jira")
    revision = engine.stage_sync("jira", SyncBatch(added=[_document("jira:new", "new")]))

    assert {item["id"] for item in _read_jsonl(revision.path / "docs.jsonl")} == {"jira:prior", "jira:new"}
    (revision.path / "lexical.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        engine.publish("jira", revision)
    assert engine.active_revision("jira") == old


def test_validate_profile_uses_complete_compatibility_contract():
    profile = EmbeddingProfile("fake", "model", "r", 2, True, "cosine", "p")
    validate_profile(profile, profile, profile)
    with pytest.raises(ValueError):
        validate_profile(profile, EmbeddingProfile("fake", "other", "r", 2, True, "cosine", "p"), profile)


class _Provider:
    def __init__(self, fail_text=None):
        self.fail_text = fail_text

    @property
    def capabilities(self):
        return EmbeddingCapabilities(provider_id="fake", model_id="model", revision="r", local=True, offline_capable=True, max_batch_size=32, dimension=2, normalized=True, similarity_metric="cosine")

    async def embed_documents(self, texts):
        if self.fail_text and self.fail_text in texts[0]:
            raise RuntimeError("embedding failed")
        return [[float(len(text)), 1.0] for text in texts]

    async def embed_query(self, text):
        return (await self.embed_documents([text]))[0]
