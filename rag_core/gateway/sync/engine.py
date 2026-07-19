from __future__ import annotations

import json
import tempfile
import asyncio
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_core.gateway.connector import SourceConnector
from rag_core.gateway.models import SyncBatch
from rag_core.gateway.sync.manifest_store import CorruptManifestError, MissingRevisionError, RevisionManifestStore
from rag_core.gateway.sync.status import read_sync_status
from rag_core.gateway.sync.index_builder import EmbeddingProviderUnavailable, build_revision
from rag_core.gateway.model_providers import EmbeddingProfile


@dataclass(frozen=True)
class Revision:
    path: Path
    cursor: str | None


class CorruptActiveRevisionError(RuntimeError):
    def __init__(self, source: str, revision: str | None):
        self.source = source
        self.revision = revision
        super().__init__(f"active revision for source {source!r} is corrupt: {revision!r}")


class SyncEngine:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def stage_sync(
        self,
        source: str,
        batch: SyncBatch,
        embed_provider: object | None = None,
        active_profile: EmbeddingProfile | None = None,
        allow_lexical_downgrade: bool = False,
    ) -> Revision:
        documents = {document["id"]: document for document in self._previous_active_documents(source)}
        documents.update({document.id: asdict(document) for document in batch.added})
        documents.update({document.id: asdict(document) for document in batch.changed})
        for document_id in batch.deleted:
            documents.pop(document_id, None)
        effective_profile = active_profile or self._active_embedding_profile(source)
        return _await_synchronously(
            self._stage_documents(source, documents, batch, embed_provider, effective_profile, allow_lexical_downgrade)
        )

    async def stage_sync_async(
        self,
        source: str,
        batch: SyncBatch,
        embed_provider: object | None = None,
        active_profile: EmbeddingProfile | None = None,
        allow_lexical_downgrade: bool = False,
    ) -> Revision:
        documents = {document["id"]: document for document in self._previous_active_documents(source)}
        documents.update({document.id: asdict(document) for document in batch.added})
        documents.update({document.id: asdict(document) for document in batch.changed})
        for document_id in batch.deleted:
            documents.pop(document_id, None)
        effective_profile = active_profile or self._active_embedding_profile(source)
        return await self._stage_documents(source, documents, batch, embed_provider, effective_profile, allow_lexical_downgrade)

    def full_rebuild(
        self,
        source: str,
        batch: SyncBatch,
        *,
        embed_provider: object | None = None,
        active_profile: EmbeddingProfile | None = None,
        allow_lexical_downgrade: bool = False,
    ) -> Revision:
        documents = {document.id: asdict(document) for document in batch.added}
        documents.update({document.id: asdict(document) for document in batch.changed})
        for document_id in batch.deleted:
            documents.pop(document_id, None)
        revision = _await_synchronously(
            self._stage_documents(
                source,
                documents,
                batch,
                embed_provider,
                active_profile,
                allow_lexical_downgrade,
                detect_active_profile=False,
            )
        )
        self.publish(source, revision)
        return revision

    async def _stage_documents(
        self,
        source: str,
        documents: dict[str, dict],
        batch: SyncBatch,
        embed_provider: object | None = None,
        active_profile: EmbeddingProfile | None = None,
        allow_lexical_downgrade: bool = False,
        detect_active_profile: bool = True,
    ) -> Revision:
        effective_profile = active_profile or (
            self._active_embedding_profile(source) if detect_active_profile else None
        )
        source_root = self.root / source
        source_root.mkdir(parents=True, exist_ok=True)
        revision_path = Path(tempfile.mkdtemp(dir=source_root, prefix="revision-"))
        await build_revision(
            revision_path,
            batch,
            embed_provider=embed_provider,
            active_profile=effective_profile,
            allow_lexical_downgrade=allow_lexical_downgrade,
            documents=documents.values(),
        )
        if batch.deleted:
            with (revision_path / "tombstones.jsonl").open("w", encoding="utf-8") as handle:
                for document_id in batch.deleted:
                    handle.write(document_id + "\n")
        return Revision(path=revision_path, cursor=batch.cursor)

    def active_revision(self, source: str) -> str | None:
        return self._manifest_store(source).active_revision()

    def _active_embedding_profile(self, source: str) -> EmbeddingProfile | None:
        active_revision = self.active_revision(source)
        if active_revision is None:
            return None
        manifest = json.loads((Path(active_revision) / "manifest.json").read_text(encoding="utf-8"))
        profile = manifest.get("embedding_profile")
        return EmbeddingProfile(**profile) if profile is not None else None

    def publish(self, source: str, revision: Revision) -> None:
        self._validate_revision(revision)
        self._manifest_store(source).write(profile={}, active_revision=str(revision.path), cursor=revision.cursor)

    def active_documents(self, source: str) -> list[dict]:
        active_revision = self.active_revision(source)
        if active_revision is None:
            return []
        revision_path = Path(active_revision)
        tombstones_path = revision_path / "tombstones.jsonl"
        tombstones = (
            set(tombstones_path.read_text(encoding="utf-8").splitlines())
            if tombstones_path.exists()
            else set()
        )
        with (revision_path / "docs.jsonl").open(encoding="utf-8") as handle:
            return [
                document
                for line in handle
                if line.strip() and (document := json.loads(line))["id"] not in tombstones
            ]

    async def sync_source(
        self, connector: SourceConnector, cursor: str | None = None, *, embed_provider: object | None = None
    ) -> Revision:
        source = connector.source
        resume_cursor = cursor if cursor is not None else self.sync_status(source)["cursor"]
        batch = await connector.sync_changes(resume_cursor)
        if not isinstance(batch, SyncBatch):
            raise TypeError("connector.sync_changes must return SyncBatch")
        revision = await self.stage_sync_async(source, batch, embed_provider=embed_provider)
        self.publish(source, revision)
        return revision

    def sync_status(self, source: str) -> dict:
        return read_sync_status(self.root, source)

    def _previous_active_documents(self, source: str) -> list[dict]:
        try:
            active_revision = self.active_revision(source)
        except (CorruptManifestError, MissingRevisionError, json.JSONDecodeError, OSError, KeyError, TypeError) as error:
            raise CorruptActiveRevisionError(source, None) from error
        if active_revision is None:
            return []
        try:
            return self.active_documents(source)
        except (CorruptManifestError, MissingRevisionError, json.JSONDecodeError, OSError, KeyError, TypeError) as error:
            raise CorruptActiveRevisionError(source, active_revision) from error

    def _manifest_path(self, source: str) -> Path:
        return self._manifest_store(source).path

    def _manifest_store(self, source: str) -> RevisionManifestStore:
        return RevisionManifestStore(self.root, source)

    @staticmethod
    def _validate_revision(revision: Revision) -> None:
        docs_file = revision.path / "docs.jsonl"
        try:
            if not revision.path.is_dir() or not docs_file.is_file():
                raise OSError("staged revision is incomplete")
            with docs_file.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        json.loads(line)
            for filename in ("chunks.jsonl", "vectors.jsonl"):
                artifact = revision.path / filename
                if artifact.exists():
                    with artifact.open(encoding="utf-8") as handle:
                        for line in handle:
                            if line.strip():
                                json.loads(line)
            lexical_file = revision.path / "lexical.json"
            manifest_file = revision.path / "manifest.json"
            if not lexical_file.is_file() or not manifest_file.is_file():
                raise OSError("staged index artifacts are incomplete")
            lexical = json.loads(lexical_file.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if not isinstance(lexical, dict) or not isinstance(manifest, dict):
                raise ValueError("staged index artifacts have invalid structure")
            with (revision.path / "chunks.jsonl").open(encoding="utf-8") as handle:
                chunks = [json.loads(line) for line in handle if line.strip()]
            document_ids = {document["id"] for document in SyncEngine._read_jsonl(docs_file)}
            chunk_ids = {item["id"] for item in chunks}
            if any(item.get("document_id") not in document_ids for item in chunks):
                raise ValueError("chunk references a missing document")
            if any(not set(ids).issubset(chunk_ids) for ids in lexical.values() if isinstance(ids, list)):
                raise ValueError("lexical index references a missing chunk")
            vectors_file = revision.path / "vectors.jsonl"
            if vectors_file.exists():
                vectors = SyncEngine._read_jsonl(vectors_file)
                if any(item.get("id") not in chunk_ids or item.get("document_id") not in document_ids for item in vectors):
                    raise ValueError("vector references a missing document or chunk")
            if set(manifest) != {"embedding_profile", "embedding_failures"}:
                raise ValueError("staged index manifest has invalid schema")
            tombstones_file = revision.path / "tombstones.jsonl"
            if tombstones_file.exists():
                with tombstones_file.open(encoding="utf-8") as handle:
                    for _ in handle:
                        pass
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as error:
            raise ValueError("staged revision failed integrity validation") from error

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


def _await_synchronously(awaitable):
    """Keep the legacy sync API while leaving async execution to the async builder."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_in_new_loop(awaitable)

    result: list[object] = []
    error: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(_run_in_new_loop(awaitable))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def _run_in_new_loop(awaitable):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(awaitable)
    finally:
        loop.close()
