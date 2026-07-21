from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


@dataclass(frozen=True)
class Document:
    id: str
    source: str
    source_instance: str
    title: str
    text: str
    uri: str | None = None
    version: str | None = None
    updated_at: datetime | None = None
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentRef:
    document_id: str
    chunk_id: str | None = None

    def __str__(self) -> str:
        if self.chunk_id:
            return f"{self.document_id}#{self.chunk_id}"
        return self.document_id


class EvidenceOrigin(str, Enum):
    LOCAL_SNAPSHOT = "local_snapshot"
    LIVE_CORPORATE = "live_corporate"
    PUBLIC_WEB = "public_web"
    AGENT_MEMORY = "agent_memory"


@dataclass(frozen=True)
class Evidence:
    id: str
    document_id: str
    title: str
    text: str
    source: str
    uri: str | None = None
    origin: EvidenceOrigin = "local_snapshot"
    retrieval_score: float = 0.0
    reranker_score: float | None = None
    final_score: float = 0.0
    updated_at: datetime | None = None
    synced_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    canonical_id: str | None = None

    def __post_init__(self) -> None:
        if self.canonical_id is None:
            from rag_core.gateway.identity import canonical_document_id

            object.__setattr__(
                self,
                "canonical_id",
                canonical_document_id(self.source, self.document_id, self.metadata),
            )


@dataclass(frozen=True)
class SyncBatch:
    added: list[Document] = field(default_factory=list)
    changed: list[Document] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    cursor: str | None = None
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceHealth:
    source: str
    available: bool
    detail: str = ""
