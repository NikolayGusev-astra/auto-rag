from __future__ import annotations

import inspect
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from rag_core.gateway.model_providers import EmbeddingProfile
from rag_core.gateway.model_runtime.compatibility import check_compatible
from rag_core.gateway.model_runtime.providers.cpu import make_cpu_profile
from rag_core.gateway.models import Document, SyncBatch


CHUNK_SIZE = 1000
_TERM = re.compile(r"\w+", re.UNICODE)


class EmbeddingProviderUnavailable(RuntimeError):
    """A vector-indexed source cannot be safely rebuilt without its provider."""


def chunk(document: Document, *, chunk_size: int = CHUNK_SIZE) -> list[dict[str, str]]:
    """Split text into fixed-size, deterministic chunks.

    Empty documents intentionally produce no chunks. Fixed character boundaries make
    oversized documents safe to index and keep IDs stable across identical rebuilds.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not document.text:
        return []
    return [
        {"id": f"{document.id}:{index}", "document_id": document.id, "text": document.text[offset : offset + chunk_size]}
        for index, offset in enumerate(range(0, len(document.text), chunk_size))
    ]


def build_lexical_index(chunks: Iterable[dict[str, str]]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for item in chunks:
        for term in set(_TERM.findall(item["text"].lower())):
            index.setdefault(term, set()).add(item["id"])
    return index


def validate_profile(
    compat: EmbeddingProfile | None,
    profile: EmbeddingProfile | None,
    active_profile: EmbeddingProfile | None,
) -> None:
    """Reject embedding contracts that cannot read the active vector data."""
    if profile is None:
        return
    checker = compat if callable(compat) else check_compatible
    expected = active_profile or (None if callable(compat) else compat)
    if expected is None:
        return
    compatible, reason = checker(profile, expected)
    if not compatible:
        raise ValueError(f"incompatible embedding profile: {reason}")


async def embed_chunks(chunks: list[dict[str, str]], provider: Any) -> list[list[float]]:
    method = provider.embed_documents
    texts = [item["text"] for item in chunks]
    if inspect.iscoroutinefunction(method):
        result = await method(texts)
    else:
        result = await _run_sync_embedding(method, texts)
    if inspect.isawaitable(result):
        result = await result
    vectors = list(result)
    if len(vectors) != len(chunks):
        raise ValueError("embedding provider returned a vector count different from chunks")
    return vectors


async def build_revision(
    source: str | Path,
    batch: SyncBatch,
    *,
    embed_provider: Any = None,
    active_profile: EmbeddingProfile | None = None,
    allow_lexical_downgrade: bool = False,
    documents: Iterable[Document | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build all index artifacts for a full document snapshot.

    When ``source`` is a directory, artifacts are written there. The engine supplies
    the merged snapshot so incremental batches cannot discard prior documents.
    """
    if active_profile is not None and embed_provider is None and not allow_lexical_downgrade:
        raise EmbeddingProviderUnavailable("an embedding provider is required for the active vector profile")
    snapshot = list(documents) if documents is not None else [*batch.added, *batch.changed]
    document_dicts = [asdict(item) if isinstance(item, Document) else dict(item) for item in snapshot]
    document_models = [Document(**item) for item in document_dicts]
    all_chunks = [item for document in document_models for item in chunk(document)]
    lexical = build_lexical_index(all_chunks)
    profile = _provider_profile(embed_provider) if embed_provider is not None else None
    validate_profile(active_profile, profile, active_profile)

    vectors: list[dict[str, Any]] = []
    failures: list[str] = []
    if embed_provider is not None:
        for document in document_models:
            document_chunks = [item for item in all_chunks if item["document_id"] == document.id]
            if not document_chunks:
                continue
            try:
                embeddings = await embed_chunks(document_chunks, embed_provider)
            except Exception:
                failures.append(document.id)
                continue
            _validate_vectors(embeddings, profile.dimension if profile is not None else None)
            vectors.extend(
                {"id": item["id"], "document_id": document.id, "vector": vector}
                for item, vector in zip(document_chunks, embeddings, strict=True)
            )

    manifest = {
        "embedding_profile": asdict(profile) if profile is not None else None,
        "embedding_failures": failures,
    }
    result = {"documents": document_dicts, "chunks": all_chunks, "lexical": lexical, "vectors": vectors, "manifest": manifest}
    if isinstance(source, Path):
        _write_revision(source, result, include_vectors=embed_provider is not None)
    return result


def _provider_profile(provider: Any) -> EmbeddingProfile:
    capabilities = provider.capabilities
    return make_cpu_profile(
        capabilities.model_id,
        capabilities.dimension,
        revision=capabilities.revision,
        normalized=capabilities.normalized,
        metric=capabilities.similarity_metric,
    )


def _write_revision(path: Path, result: dict[str, Any], *, include_vectors: bool) -> None:
    _write_jsonl(path / "docs.jsonl", result["documents"])
    _write_jsonl(path / "chunks.jsonl", result["chunks"])
    (path / "lexical.json").write_text(json.dumps({key: sorted(value) for key, value in result["lexical"].items()}, sort_keys=True), encoding="utf-8")
    if include_vectors:
        _write_jsonl(path / "vectors.jsonl", result["vectors"])
    (path / "manifest.json").write_text(json.dumps(result["manifest"], sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, default=str, sort_keys=True) + "\n")


async def _run_sync_embedding(method: Any, texts: list[str]) -> Any:
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, method, texts)


def _validate_vectors(vectors: list[list[float]], expected_dimension: int | None) -> None:
    for vector in vectors:
        if expected_dimension is not None and len(vector) != expected_dimension:
            raise ValueError(f"embedding vector dimension {len(vector)} does not match profile dimension {expected_dimension}")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
            raise ValueError("embedding vectors must contain only finite numeric values")
