from __future__ import annotations

import json
from pathlib import Path

from rag_core.gateway.model_providers import EmbeddingProfile


class ReindexPlanner:
    def __init__(self, root: Path):
        self.root = Path(root)

    def build_staged(self, profile: EmbeddingProfile, docs: list[dict]) -> Path:
        """Write a new, inactive revision for a replacement embedding profile."""
        key = f"{profile.model_id}@{profile.model_revision}"
        revision = self.root / "reindex-staged" / key
        revision.mkdir(parents=True, exist_ok=True)
        with (revision / "docs.jsonl").open("w", encoding="utf-8") as handle:
            for document in docs:
                handle.write(json.dumps(document, ensure_ascii=False) + "\n")
        return revision

    def check_integrity(self, revision_path: Path) -> bool:
        docs_file = Path(revision_path) / "docs.jsonl"
        if not docs_file.exists():
            return False
        try:
            with docs_file.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        json.loads(line)
            return True
        except (json.JSONDecodeError, OSError):
            return False
