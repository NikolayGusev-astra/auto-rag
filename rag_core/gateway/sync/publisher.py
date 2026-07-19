from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from rag_core.gateway.model_providers import EmbeddingProfile
from rag_core.gateway.model_runtime.reindex import ReindexPlanner
from rag_core.gateway.sync.engine import SyncEngine


class RevisionPublisher:
    """Publishes verified embedding revisions through SyncEngine's atomic swap."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._planner = ReindexPlanner(self.root)
        self._sync_engine = SyncEngine(self.root)

    def build_staged(self, profile: EmbeddingProfile, docs: list[dict]) -> Path:
        return self._planner.build_staged(profile, docs)

    def publish(self, profile: EmbeddingProfile, revision_path: Path) -> None:
        if not self._planner.check_integrity(revision_path):
            raise ValueError("staged revision failed integrity check; not published")
        self._sync_engine.atomic_write_json(
            self.root / "index_manifest.json",
            {"profile": asdict(profile), "active_revision": str(revision_path)},
        )
