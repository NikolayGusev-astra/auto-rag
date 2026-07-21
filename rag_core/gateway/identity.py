"""Stable document identities shared by live and snapshot retrieval."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def canonical_document_id(
    source: str, document_id: str, metadata: Mapping[str, Any] | None = None
) -> str:
    """Return the source-qualified ID used to recognize the same document."""
    normalized_source = source.strip().casefold()
    metadata = metadata or {}
    if ":" in document_id:
        prefix, value = document_id.split(":", 1)
        if prefix.casefold() in {"jira", "confluence", "wiki"} and value:
            return canonical_document_id(prefix, value, metadata)
    if normalized_source == "jira":
        return f"jira:{document_id.strip().upper()}"
    if normalized_source == "confluence":
        return f"confluence:{document_id.strip()}"
    if normalized_source == "wiki":
        slug = metadata.get("slug")
        if isinstance(slug, str) and slug.strip():
            return f"wiki:{slug.strip().casefold()}"
    return f"{normalized_source}:{document_id.strip()}"
