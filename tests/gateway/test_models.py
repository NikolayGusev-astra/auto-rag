import pytest
from dataclasses import FrozenInstanceError
from rag_core.gateway.models import Document, DocumentRef


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
