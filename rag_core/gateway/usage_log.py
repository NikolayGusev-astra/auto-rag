"""Minimal usage logger for ADR-006 Step 5 (dogfooding period)."""
from __future__ import annotations

import hashlib, json, time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag_core.gateway.models import Evidence

LOG_PATH = Path.home() / ".local" / "share" / "auto-rag" / "usage.jsonl"


@dataclass
class UsageEntry:
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    query_hash: str = ""
    query_class: str = ""
    sources_used: list[str] = field(default_factory=list)
    hit: bool = False
    user_accepted: bool = False
    latency_ms: int = 0
    generation_model: str = ""
    duplicate_count: int = 0
    problem: str | None = None


def hash_query(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:12]


def classify_query(query: str) -> str:
    q = query.strip().upper()
    if any(kw in q for kw in ("INT-", "ACM-", "AA-", "ALD-", "SIRIUS-", "POLLUX-", "RAIT-", "MON-", "AKNO-", "VEGA-", "ATL-")):
        return "exact_id"
    if any(kw in q for kw in ("ВЕРСИ", "ПМИ", "ДОКУМЕНТАЦ", "СЕРТИФИК")):
        return "product"
    if any(kw in q for kw in ("ОШИБК", "БАГ", "ПРОБЛЕМ", "НЕ РАБОТА", "FAIL", "TIMEOUT")):
        return "troubleshooting"
    if any(kw in q for kw in ("КАК", "ГДЕ", "ЧТО", "КОГДА", "КТО")):
        return "research"
    return "summary"


def log_usage(query: str, evidence: list[Evidence], elapsed_ms: int, model: str = "", problem: str | None = None) -> None:
    entry = UsageEntry(
        query_hash=hash_query(query),
        query_class=classify_query(query),
        sources_used=list({e.source for e in evidence}),
        hit=len(evidence) > 0,
        latency_ms=elapsed_ms,
        generation_model=model,
        duplicate_count=sum(1 for e in evidence if hasattr(e, "canonical_id") and e.canonical_id and any(
            e.canonical_id == o.canonical_id for o in evidence if o is not e
        )),
        problem=problem,
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def usage_summary() -> dict:
    if not LOG_PATH.exists():
        return {"entries": 0}
    entries = []
    for line in LOG_PATH.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    hits = sum(1 for e in entries if e["hit"])
    latencies = [e["latency_ms"] for e in entries if e["latency_ms"] > 0]
    problems = [e["problem"] for e in entries if e["problem"]]
    classes = {}
    for e in entries:
        c = e["query_class"]
        classes[c] = classes.get(c, 0) + 1
    return {
        "entries": len(entries),
        "hit_rate": hits / len(entries) if entries else 0,
        "latency_p50": sorted(latencies)[len(latencies)//2] if latencies else 0,
        "latency_p95": sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) >= 20 else 0,
        "problems": problems[-5:],
        "query_classes": classes,
    }
