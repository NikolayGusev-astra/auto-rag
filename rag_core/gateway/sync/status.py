from __future__ import annotations

import json
from pathlib import Path

from rag_core.gateway.models import SourceHealth


def read_sync_status(root: Path, source: str) -> dict:
    manifest = Path(root) / source / "manifest.json"
    if not manifest.exists():
        health = SourceHealth(source=source, available=False, detail="no published revision")
        return {"source": source, "available": health.available, "cursor": None, "health": health}

    data = json.loads(manifest.read_text(encoding="utf-8"))
    health = SourceHealth(source=source, available=True, detail="published revision available")
    return {
        "source": source,
        "available": health.available,
        "cursor": data.get("cursor"),
        "health": health,
    }
