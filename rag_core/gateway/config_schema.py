"""Versioned, local-only gateway configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class UnsupportedConfigVersion(ValueError):
    """Raised when a configuration needs a schema this gateway does not support."""


@dataclass(frozen=True)
class SourceConfig:
    name: str
    kind: str
    enabled: bool = True
    credential_ref: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GatewayConfig:
    version: int = 1
    knowledge_root: Path = field(default_factory=lambda: Path.home() / ".local" / "share" / "auto-rag")
    local_snapshot: bool = True
    sources: dict[str, SourceConfig] = field(default_factory=dict)
    web: bool = False
    adaptive: bool = False

    def __post_init__(self) -> None:
        if self.version != 1:
            raise UnsupportedConfigVersion(f"unsupported gateway config version: {self.version}")
        object.__setattr__(self, "knowledge_root", Path(self.knowledge_root))
