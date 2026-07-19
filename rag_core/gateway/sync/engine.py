from __future__ import annotations

import json
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
        revision_path = self.root / source / "staged"
        revision_path.mkdir(parents=True, exist_ok=True)
        with (revision_path / "docs.jsonl").open("w", encoding="utf-8") as handle:
            for document in batch.added:
                handle.write(json.dumps(asdict(document), default=str) + "\n")
        if batch.deleted:
            with (revision_path / "tombstones.jsonl").open("w", encoding="utf-8") as handle:
                for document_id in batch.deleted:
                    handle.write(document_id + "\n")
        return Revision(path=revision_path, cursor=batch.cursor)

    def active_revision(self, source: str) -> str | None:
        return None
