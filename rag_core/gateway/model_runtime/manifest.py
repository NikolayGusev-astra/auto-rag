from __future__ import annotations

import json
from pathlib import Path

from rag_core.gateway.model_providers import EmbeddingProfile


class IndexManifest:
    def __init__(self, root: Path):
        self._root = Path(root)
        self._path = self._root / "index_manifest.json"
        self.profile: EmbeddingProfile | None = None
        self.active_revision: str | None = None
        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.profile = EmbeddingProfile(**data["profile"])
            self.active_revision = data["active_revision"]

    def write(self, profile: EmbeddingProfile, active_revision: str) -> None:
        self.profile = profile
        self.active_revision = active_revision
        self._root.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                {
                    "profile": profile.__dict__,
                    "active_revision": active_revision,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
