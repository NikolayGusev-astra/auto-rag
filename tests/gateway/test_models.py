import pytest
from dataclasses import FrozenInstanceError
from rag_core.gateway.models import Document, DocumentRef
from rag_core.gateway.models import Evidence, EvidenceOrigin


def test_document_is_frozen_and_has_required_fields():
    doc = Document(
        id="confluence:12345",
        source="confluence",
        source_instance="wiki-prod",
        title="Обновление кластера",
        text="...",
        uri="https://wiki.example/pages/12345",
        version="v3",
        updated_at=None,
        content_hash="abc123",
        metadata={},
    )
    assert doc.id == "confluence:12345"
    assert doc.source == "confluence"
    with pytest.raises(FrozenInstanceError):
        doc.title = "x"


def test_documentref_identifies_chunk():
    ref = DocumentRef(document_id="confluence:12345", chunk_id="chunk-4")
    assert ref.document_id == "confluence:12345"
    assert str(ref) == "confluence:12345#chunk-4"


def test_evidence_has_origin_and_scores():
    ev = Evidence(
        id="confluence:12345#chunk-4",
        document_id="confluence:12345",
        title="Обновление кластера",
        text="...",
        source="confluence",
        uri="https://wiki.example/pages/12345",
        origin=EvidenceOrigin.LOCAL_SNAPSHOT,
        retrieval_score=0.81,
        reranker_score=0.92,
        updated_at=None,
        synced_at=None,
        metadata={},
    )
    assert ev.origin == "local_snapshot"
    assert ev.retrieval_score == 0.81
