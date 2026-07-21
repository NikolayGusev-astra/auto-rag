"""Gateway maintenance commands."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from rag_core.gateway.adaptive.dcd_learner import DcdLearner
from rag_core.gateway.adaptive.source_discovery import SourceDiscovery
from rag_core.gateway.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="auto-rag gateway maintenance")
    subcommands = parser.add_subparsers(dest="command", required=True)
    discover = subcommands.add_parser("discover", help="discover product documentation routes")
    discover.add_argument("--wiki", type=Path, default=Path(os.environ.get("LLM_WIKI_PATH", ".")))
    discover.add_argument("--config", type=Path)
    learn = subcommands.add_parser("dcd-learn", help="learn source affinities from retrieval episodes")
    learn.add_argument(
        "--episodes", type=Path,
        default=Path.home() / ".local" / "share" / "auto-rag" / "episodes.jsonl",
    )
    learn.add_argument(
        "--routing", type=Path,
        default=Path.home() / ".config" / "auto-rag" / "routing.json",
    )
    doctor = subcommands.add_parser("doctor", help="run read-only health checks")
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()
    if args.command == "discover":
        routing = asyncio.run(SourceDiscovery(args.wiki, load_config(args.config)).update_routing())
        print(json.dumps(routing, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "dcd-learn":
        routing = DcdLearner(args.episodes, args.routing).learn()
        print(json.dumps(routing, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "doctor":
        from rag_core.gateway.doctor import main as doctor_main

        doctor_args = (["--config", str(args.config)] if args.config else [])
        if args.json_output:
            doctor_args.append("--json")
        raise SystemExit(doctor_main(doctor_args))


if __name__ == "__main__":
    main()
