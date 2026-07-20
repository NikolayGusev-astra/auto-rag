"""Non-blocking startup visibility for local gateway connectors."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rag_core.gateway.connector import SourceConnector
from rag_core.gateway.sync.engine import _await_synchronously


def collect_startup_diagnostics(connectors: Mapping[str, SourceConnector]) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    for name, connector in connectors.items():
        health, reason = _connector_health(connector)
        entry = {
            "source": getattr(connector, "source", name),
            "kind": getattr(connector, "retrieval_kind", "live"),
            "health": health,
        }
        if reason is not None:
            entry["reason"] = reason
        entries[name] = entry
    healthy = [name for name, entry in entries.items() if entry["health"]]
    unhealthy = [name for name, entry in entries.items() if not entry["health"]]
    return {
        "connectors": entries,
        "offline": {"healthy": healthy, "unhealthy": unhealthy},
        "notes": list(getattr(connectors, "diagnostics", ())),
    }


def _connector_health(connector: SourceConnector) -> tuple[bool, str | None]:
    try:
        status = _await_synchronously(connector.health())
    except Exception as error:
        return False, str(error) or type(error).__name__
    if isinstance(status, Mapping):
        reason = status.get("reason")
        return bool(status.get("available", False)), reason if isinstance(reason, str) else None
    return bool(getattr(status, "available", False)), None
