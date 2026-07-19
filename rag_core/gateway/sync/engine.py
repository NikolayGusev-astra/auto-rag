from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_core.gateway.models import SyncBatch


@dataclass(frozen=True)
class Revision:
    path: Path
    cursor: str | None


class SyncEngine:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def stage_sync(self, source: str, batch: SyncBatch) -> Revision:
        source_root = self.root / source
        source_root.mkdir(parents=True, exist_ok=True)
        revision_path = Path(tempfile.mkdtemp(dir=source_root, prefix="revision-"))
        with (revision_path / "docs.jsonl").open("w", encoding="utf-8") as handle:
            for document in batch.added:
                handle.write(json.dumps(asdict(document), default=str) + "\n")
        if batch.deleted:
            with (revision_path / "tombstones.jsonl").open("w", encoding="utf-8") as handle:
                for document_id in batch.deleted:
                    handle.write(document_id + "\n")
        return Revision(path=revision_path, cursor=batch.cursor)

    def active_revision(self, source: str) -> str | None:
        manifest = self._manifest_path(source)
        if not manifest.exists():
            return None
        return json.loads(manifest.read_text(encoding="utf-8")).get("active_index")

    def publish(self, source: str, revision: Revision) -> None:
        self._validate_revision(revision)
        source_root = self.root / source
        source_root.mkdir(parents=True, exist_ok=True)
        temporary_manifest = source_root / "manifest.tmp.json"
        temporary_manifest.write_text(
            json.dumps({"active_index": str(revision.path), "cursor": revision.cursor}),
            encoding="utf-8",
        )
        os.replace(temporary_manifest, self._manifest_path(source))

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

    def _manifest_path(self, source: str) -> Path:
        return self.root / source / "manifest.json"

    @staticmethod
    def _validate_revision(revision: Revision) -> None:
        docs_file = revision.path / "docs.jsonl"
        try:
            with docs_file.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        json.loads(line)
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError("staged revision failed integrity validation") from error
