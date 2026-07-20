"""Non-blocking startup visibility for local gateway connectors."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rag_core.gateway.connector import SourceConnector
from rag_core.gateway.sync.engine import _await_synchronously


def collect_startup_diagnostics(connectors: Mapping[str, SourceConnector]) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    for name, connector in connectors.items():
        health = _connector_health(connector)
        entries[name] = {
            "source": getattr(connector, "source", name),
            "kind": getattr(connector, "retrieval_kind", "live"),
            "health": health,
        }
    healthy = [name for name, entry in entries.items() if entry["health"]]
    unhealthy = [name for name, entry in entries.items() if not entry["health"]]
    return {
        "connectors": entries,
        "offline": {"healthy": healthy, "unhealthy": unhealthy},
        "notes": list(getattr(connectors, "diagnostics", ())),
    }


def _connector_health(connector: SourceConnector) -> bool:
    if getattr(connector, "retrieval_kind", None) == "local":
        return True
    try:
        status = _await_synchronously(connector.health())
    except Exception:
        return False
    if isinstance(status, Mapping):
        return bool(status.get("available", False))
    return bool(getattr(status, "available", False))
