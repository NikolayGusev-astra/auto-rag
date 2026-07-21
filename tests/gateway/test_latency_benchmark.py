"""Latency benchmark for parallel gateway connector fan-out.

The test writes its latest measurements to ``benchmark_results.json``.  CI can
upload or compare that artifact without calling real connector services.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import statistics
import time

import pytest

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.mcp_handlers import handle_search
from rag_core.gateway.models import Evidence, EvidenceOrigin


ITERATIONS = 20
CONNECTOR_DELAYS_S = {
    "fast": 0.01,
    "medium": 0.05,
    "slow": 0.20,
    "tail": 1.00,
}
FAN_OUT_OVERHEAD_S = 0.20
RESULTS_PATH = Path(os.environ.get("GATEWAY_LATENCY_BENCHMARK_RESULTS", "benchmark_results.json"))


class DelayedConnector:
    def __init__(self, source: str, delay_s: float) -> None:
        self.source = source
        self.delay_s = delay_s

    async def health(self) -> dict[str, bool]:
        return {"available": True}

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        await asyncio.sleep(self.delay_s)
        return [
            Evidence(
                id=f"{self.source}:1",
                document_id=f"{self.source}:1",
                title=self.source,
                text=request.query,
                source=self.source,
                origin=EvidenceOrigin.LIVE_CORPORATE,
                retrieval_score=1.0,
            )
        ]


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


@pytest.mark.asyncio
async def test_gateway_connector_latency_benchmark() -> None:
    """Prove connector searches fan out in parallel and record percentiles."""
    connectors = {
        name: DelayedConnector(name, delay_s)
        for name, delay_s in CONNECTOR_DELAYS_S.items()
    }
    samples_ms = {name: [] for name in connectors}
    total_samples_ms: list[float] = []

    for _ in range(ITERATIONS):
        started = time.perf_counter()
        response = await handle_search(SearchRequest(query="latency benchmark"), connectors)
        total_elapsed_ms = (time.perf_counter() - started) * 1000
        trace = response["trace"]
        latency = trace["latency"]

        assert trace["connector_count"] == len(connectors)
        assert set(connectors).issubset(latency)
        for name in connectors:
            stage = latency[name]
            assert stage["status"] == "completed"
            assert stage["duration_ms"] > 0
            samples_ms[name].append(stage["duration_ms"])

        slowest_connector_ms = max(latency[name]["duration_ms"] for name in connectors)
        assert total_elapsed_ms <= slowest_connector_ms + FAN_OUT_OVERHEAD_S * 1000
        total_samples_ms.append(total_elapsed_ms)

    connector_percentiles = {
        name: {
            "configured_delay_ms": delay_s * 1000,
            "p50_ms": round(statistics.median(samples_ms[name]), 3),
            "p95_ms": round(_percentile(samples_ms[name], 0.95), 3),
        }
        for name, delay_s in CONNECTOR_DELAYS_S.items()
    }
    report = {
        "iterations": ITERATIONS,
        "fan_out_overhead_ms": FAN_OUT_OVERHEAD_S * 1000,
        "connectors": connector_percentiles,
        "total_search_ms": {
            "p50_ms": round(statistics.median(total_samples_ms), 3),
            "p95_ms": round(_percentile(total_samples_ms, 0.95), 3),
        },
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    assert all(values["p50_ms"] > 0 for values in connector_percentiles.values())
    assert all(values["p95_ms"] >= values["p50_ms"] for values in connector_percentiles.values())
