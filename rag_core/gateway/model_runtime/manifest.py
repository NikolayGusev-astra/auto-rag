from __future__ import annotations

from pathlib import Path

from rag_core.gateway.model_providers import EmbeddingProfile
from rag_core.gateway.sync.manifest_store import RevisionManifestStore


class IndexManifest:
    def __init__(self, root: Path):
        self._root = Path(root)
        self._store = RevisionManifestStore(self._root, None)
        self.profile: EmbeddingProfile | None = None
        self.active_revision: str | None = None
        data = self._store.read()
        if data is not None:
            self.profile = EmbeddingProfile(**data["profile"])
            self.active_revision = data["active_revision"]

    def write(self, profile: EmbeddingProfile, active_revision: str) -> None:
        self.profile = profile
        self.active_revision = active_revision
        self._store.write(profile=profile.__dict__, active_revision=active_revision, cursor=None)
