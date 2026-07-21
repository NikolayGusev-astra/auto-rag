"""Read-only health checks for the minimal auto-rag reference profile."""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from rag_core.gateway.config_loader import load_config
from rag_core.gateway.config_schema import GatewayConfig, SourceConfig
from rag_core.gateway.connector_factory import _build_source
from rag_core.gateway.sync.manifest_store import RevisionManifestStore


CONFIG_ERROR = 1
SNAPSHOT_UNAVAILABLE = 2
OPTIONAL_DEGRADED = 3


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str | None = None

    def as_dict(self) -> dict[str, str]:
        result = {"name": self.name, "status": self.status}
        if self.detail:
            result["detail"] = self.detail
        return result


def run_doctor(config_path: Path | None = None) -> tuple[int, list[Check]]:
    """Check configuration and configured components without modifying local state."""
    try:
        config = load_config(config_path)
        if not config.local_snapshot:
            raise ValueError("local_snapshot must be enabled for the reference profile")
    except Exception as error:
        return CONFIG_ERROR, [Check("CONFIG", "ERROR", str(error))]

    snapshot = _snapshot_check(config)
    optional = _optional_checks(config)
    if snapshot.status != "OK":
        return SNAPSHOT_UNAVAILABLE, [Check("CONFIG", "OK"), snapshot, *optional]
    if any(check.status != "OK" for check in optional):
        return OPTIONAL_DEGRADED, [Check("CONFIG", "OK"), snapshot, *optional]
    return 0, [Check("CONFIG", "OK"), snapshot, *optional]


def _snapshot_check(config: GatewayConfig) -> Check:
    try:
        revision = RevisionManifestStore(config.knowledge_root, "local_snapshot").active_revision()
    except Exception as error:
        return Check("LOCAL SNAPSHOT", "UNAVAILABLE", str(error))
    if revision is None:
        return Check("LOCAL SNAPSHOT", "UNAVAILABLE", "no published snapshot")
    return Check("LOCAL SNAPSHOT", "OK", revision)


def _optional_checks(config: GatewayConfig) -> list[Check]:
    checks: list[Check] = []
    for name, source in config.sources.items():
        if not source.enabled:
            continue
        checks.append(_source_check(name, source))
    return checks


def _source_check(name: str, source: SourceConfig) -> Check:
    if source.kind == "memvid":
        return Check(name.upper(), "UNAVAILABLE", "Memvid enricher is unavailable")
    diagnostics: list[str] = []
    connector = _build_source(name, source, diagnostics)
    if connector is None:
        return Check(name.upper(), "UNAVAILABLE", "; ".join(diagnostics) or "connector is unavailable")
    try:
        health = asyncio.run(connector.health())
    except Exception as error:
        return Check(name.upper(), "UNAVAILABLE", str(error) or type(error).__name__)
    if isinstance(health, dict) and health.get("available"):
        return Check(name.upper(), "OK")
    detail = health.get("reason") if isinstance(health, dict) else None
    return Check(name.upper(), "UNAVAILABLE", str(detail) if detail else "; ".join(diagnostics) or None)


def render_human(exit_code: int, checks: list[Check]) -> str:
    lines = ["AUTO-RAG DOCTOR"]
    for check in checks:
        line = f"{check.name} {check.status}"
        if check.detail:
            line += f": {check.detail}"
        lines.append(line)
    summary = {0: "READY", 1: "CONFIG ERROR", 2: "SNAPSHOT UNAVAILABLE", 3: "OPTIONAL COMPONENTS DEGRADED"}
    lines.append(f"STATUS {summary[exit_code]}")
    return "\n".join(lines)


def render_json(exit_code: int, checks: list[Check]) -> str:
    return json.dumps({"exit_code": exit_code, "checks": [check.as_dict() for check in checks]}, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="read-only auto-rag health checks")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    exit_code, checks = run_doctor(args.config)
    print(render_json(exit_code, checks) if args.json_output else render_human(exit_code, checks))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
