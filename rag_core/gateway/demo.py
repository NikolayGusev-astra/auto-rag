"""ADR-006 Step 4: Reproducible demonstration scenarios."""
from __future__ import annotations

import asyncio, json, os, sys, time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.connector_factory import build_connectors
from rag_core.gateway.config_loader import load_config
from rag_core.gateway.coordinator import RetrievalCoordinator
from rag_core.gateway.mcp_handlers import handle_search
from rag_core.gateway.adaptive.enrichment import MemvidEnricher

DEFAULT_CONFIG = Path.home() / ".config" / "auto-rag" / "gateway.toml"
EPISODES_PATH = Path.home() / ".local" / "share" / "auto-rag" / "episodes.jsonl"


@dataclass
class DemoReport:
    scenario: str
    query: str
    passed: bool
    results: int
    canonical_ids: list[str]
    sources: list[str]
    latency_ms: float | None
    details: str = ""


async def _run_query(query: str, connectors, enricher, include_web: bool = False) -> dict:
    r = await handle_search(
        SearchRequest(query=query, topk=5, include_web=include_web),
        connectors, enricher=enricher,
    )
    return r


async def demo_a_corporate(connectors, enricher) -> DemoReport:
    """Demo A: exact corporate entity query."""
    query = "INT-6515"
    r = await _run_query(query, connectors, enricher)
    results = r["results"]
    cids = [x.get("canonical_id", "") for x in results if x.get("canonical_id")]
    sources = list({x["source"] for x in results})
    passed = any("jira:INT-6515" == cid for cid in cids)
    return DemoReport(
        scenario="A: Corporate exact",
        query=query,
        passed=passed,
        results=len(results),
        canonical_ids=cids[:5],
        sources=sources,
        latency_ms=r["trace"].get("elapsed_ms"),
        details="Jira INT-6515 in results" if passed else "INT-6515 not found",
    )


async def demo_b_offline(connectors, enricher) -> DemoReport:
    """Demo B: offline degradation — local-only retrieval."""
    query = "INT-6515"
    offline = {k: v for k, v in connectors.items() if v.source in {"local_snapshot", "memvid", "zvec"}}
    r = await run_with_connectors(query, offline, enricher)
    return DemoReport(
        scenario="B: Offline degradation",
        query=query,
        passed=len(r["results"]) > 0,
        results=len(r["results"]),
        canonical_ids=[x.get("canonical_id", "?") for x in r["results"][:3]],
        sources=list({x["source"] for x in r["results"]}),
        latency_ms=r["trace"].get("elapsed_ms") if "trace" in r else None,
        details="offline retrieval succeeded" if len(r["results"]) > 0 else "no offline results",
    )


async def demo_c_web(connectors, enricher) -> DemoReport:
    """Demo C: web enrichment."""
    query = "Astra Linux"
    r = await _run_query(query, connectors, enricher, include_web=True)
    results = r["results"]
    web_results = [x for x in results if "PUBLIC_WEB" in str(x.get("origin", ""))]
    passed = len(web_results) > 0 and any(x.get("uri") for x in web_results)
    return DemoReport(
        scenario="C: Web enrichment",
        query=query,
        passed=passed,
        results=len(results),
        canonical_ids=[x.get("canonical_id", "?") for x in web_results[:3]],
        sources=list({x["source"] for x in results}),
        latency_ms=r["trace"].get("elapsed_ms"),
        details=f"web results: {len(web_results)}",
    )


async def _no_web(query, connectors, enricher) -> DemoReport:
    """Verify include_web=False blocks web."""
    r = await _run_query(query, connectors, enricher, include_web=False)
    web_results = [x for x in r["results"] if "PUBLIC_WEB" in str(x.get("origin", ""))]
    return DemoReport(
        scenario="C-extra: Web OFF",
        query=query,
        passed=len(web_results) == 0,
        results=len(r["results"]),
        canonical_ids=[],
        sources=list({x["source"] for x in r["results"]}),
        latency_ms=r["trace"].get("elapsed_ms"),
        details=f"web blocked: {len(web_results)==0}",
    )


async def run_with_connectors(query: str, connectors, enricher) -> dict:
    coord = RetrievalCoordinator(connectors)
    results = await coord.search(SearchRequest(query=query, topk=5, include_web=False))
    return {
        "results": [asdict(x) for x in results],
        "trace": {"elapsed_ms": 0},
    }


async def main():
    config = load_config(DEFAULT_CONFIG)
    connectors = build_connectors(config)
    enricher = MemvidEnricher(EPISODES_PATH)

    reports = [
        await demo_a_corporate(connectors, enricher),
        await demo_b_offline(connectors, enricher),
        await demo_c_web(connectors, enricher),
        await _no_web("Astra Linux", connectors, enricher),
    ]

    print("=" * 60)
    print("ADR-006 DEMONSTRATION REPORT")
    print("=" * 60)
    for rep in reports:
        status = "PASS" if rep.passed else "FAIL"
        print(f"\n[{status}] {rep.scenario}")
        print(f"  Query: {rep.query}")
        print(f"  Results: {rep.results} | Sources: {rep.sources}")
        print(f"  Canonical IDs: {rep.canonical_ids}")
        if rep.latency_ms:
            print(f"  Latency: {rep.latency_ms:.0f}ms")
        print(f"  {rep.details}")

    passed = sum(1 for r in reports if r.passed)
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {passed}/{len(reports)} passed")
    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
