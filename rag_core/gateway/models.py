from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
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
