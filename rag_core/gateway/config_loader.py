"""TOML loading for the local gateway configuration."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from rag_core.gateway.config_schema import GatewayConfig, SourceConfig


class ConfigNotFound(FileNotFoundError):
    """Raised when an explicitly requested configuration file is absent."""


def load_config(path: Path | None = None) -> GatewayConfig:
    if path is None:
        return GatewayConfig()
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigNotFound(f"gateway config not found: {config_path}")
    with config_path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)
    raw_sources = raw.pop("sources", {})
    sources = {
        name: SourceConfig(name=name, **source)
        for name, source in raw_sources.items()
    }
    return GatewayConfig(sources=sources, **raw)
