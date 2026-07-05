#!/usr/bin/env python3
"""Benchmark RAG pipeline с LM Studio backend."""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "rag_core"))

import psutil
import requests

DEFAULT_QUERIES = [
    "настроить postgresql replication",
    "wireguard vpn config",
    "nginx reverse proxy ssl",
]


def get_ram_usage() -> dict:
    proc = psutil.Process()
    mem = proc.memory_info()
    return {
        "rss_mb": round(mem.rss / (1024**2), 1),
        "vms_mb": round(mem.vms / (1024**2), 1),
    }


def benchmark_lm_studio() -> dict:
    from lm_studio_monitor import get_lm_studio
    monitor = get_lm_studio()
    status = monitor.get_status(force=True)

    if not status["available"]:
        return {"available": False, "error": status.get("error")}

    from embedding_service import EmbeddingService
    svc = EmbeddingService.get()

    t0 = time.time()
    svc.embed("warmup query")
    cold_s = time.time() - t0

    batch_results = []
    for batch_size in [1, 5, 10, 32]:
        texts = [f"test document number {i}" for i in range(batch_size)]
        t0 = time.time()
        svc.embed_batch(texts)
        latency = time.time() - t0
        batch_results.append({
            "batch_size": batch_size,
            "latency_ms": round(latency * 1000, 1),
        })

    return {
        "available": True,
        "embedding_cold_load_s": round(cold_s, 2),
        "batch_results": batch_results,
        "stats": svc.stats(),
    }


def main():
    parser = argparse.ArgumentParser(description="Auto-RAG benchmark")
    parser.add_argument("--queries", type=int, default=0)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--concurrent", type=int, default=1)
    parser.add_argument("--federation", type=int, default=0)
    args = parser.parse_args()

    results = {}

    if args.warmup:
        from lm_studio_monitor import get_lm_studio
        results["warmup"] = get_lm_studio().warmup_all()

    results["lm_studio"] = benchmark_lm_studio()

    report_path = Path(__file__).parent.parent / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
