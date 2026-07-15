"""
memvid_trace.py — RagTrace integration for memvid memory layer.

Wraps MemvidMemory.recall()/record() so every memory operation becomes
a traceable stage in auto-rag's RagTrace telemetry, alongside DCD /
ZVec / MCP / LLM Verify. This makes the memory layer observable in
`golden_eval_report.json` and lets canary comparisons attribute
latency/accuracy deltas to the memory stage.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    from memvid_trace import MemvidTraced
    from memvid_memory import MemvidMemory

    mem = MemvidTraced(MemvidMemory.for_tenant("hermes_default"))

    # inside rag_search.py — build RagTrace incrementally
    trace = {"stages": []}

    priors = mem.recall(query, domain=domain, trace=trace)
    #  -> appends {"stage":"memvid.recall", "latency_ms":..,
    #               "hits":3, "top_score":0.81, "used":bool} to trace

    # ... run normal RAG ...

    mem.record(Episode(...), trace=trace)
    #  -> appends {"stage":"memvid.record", "latency_ms":..,
    #               "frame_id":.., "commit":bool}

    trace["from_memory"] = used_prior   # surfaced in canary reports

------------------------------------------------------------------------------
RAGTRACE STAGE SCHEMA
------------------------------------------------------------------------------
Each stage dict:
    {
      "stage": "memvid.recall" | "memvid.record",
      "latency_ms": float,
      "ts": ISO8601,
      ...stage-specific fields
    }
The caller is responsible for merging `trace["stages"]` into the final
RagTrace that goes into golden_eval_report.json.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from memvid_memory import Episode, MemvidMemory

log = logging.getLogger("hermes.memvid.trace")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _push_stage(trace: Optional[Dict[str, Any]], stage: Dict[str, Any]) -> None:
    """Append a stage to trace; tolerate missing trace or RagTrace objects.

    If ``trace`` is a RagTrace object (has callable ``.event``), use that
    method with the ``duration_ms`` key.  Otherwise use the existing dict path.
    """
    if trace is None:
        return
    # RagTrace object path — use .event() with duration_ms
    if hasattr(trace, "event") and callable(trace.event):
        name = stage.pop("stage", "memvid.unknown")
        # Rename latency_ms -> duration_ms for RagTrace
        if "latency_ms" in stage and "duration_ms" not in stage:
            stage["duration_ms"] = stage.pop("latency_ms")
        # Rename total_latency_ms for RagTrace
        if "total_latency_ms" in stage:
            del stage["total_latency_ms"]
        trace.event(name, **stage)
        return
    # Dict path — original behaviour
    stages = trace.setdefault("stages", [])
    stages.append(stage)
    # convenience rollups
    total = trace.get("total_latency_ms", 0.0) or 0.0
    trace["total_latency_ms"] = total + float(stage.get("latency_ms", 0.0))


class MemvidTraced:
    """Decorator around MemvidMemory that emits RagTrace stages.

    Drop-in replacement: same API (recall/record/recall_as_context) plus
    an optional `trace=` kwarg on each call.
    """

    def __init__(self, inner: MemvidMemory):
        self._inner = inner

    # -- pass-throughs ------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._inner.active

    @property
    def recall_threshold(self) -> float:
        return self._inner.recall_threshold

    def close(self):
        return self._inner.close()

    # -- traced recall ------------------------------------------------------
    def recall(self, query: str, *, domain: Optional[str] = None,
               top_k: Optional[int] = None, when: Optional[str] = None,
               trace: Optional[Dict[str, Any]] = None) -> List[Episode]:
        t0 = time.perf_counter()
        err: Optional[str] = None
        eps: List[Episode] = []
        try:
            eps = self._inner.recall(query, domain=domain,
                                     top_k=top_k, when=when)
        except Exception as e:
            err = repr(e)
            log.warning("traced recall error: %s", e)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        top_score = eps[0].score if eps else 0.0
        _push_stage(trace, {
            "stage": "memvid.recall",
            "latency_ms": round(dt_ms, 2),
            "ts": _now(),
            "query_chars": len(query or ""),
            "domain": domain,
            "when": when,
            "hits": len(eps),
            "top_score": round(top_score, 4),
            "above_threshold": bool(eps and top_score >= self._inner.recall_threshold),
            "error": err,
        })
        return eps

    # -- traced record ------------------------------------------------------
    def record(self, episode: Episode, *,
               trace: Optional[Dict[str, Any]] = None) -> bool:
        t0 = time.perf_counter()
        ok = False
        err: Optional[str] = None
        try:
            ok = self._inner.record(episode)
        except Exception as e:
            err = repr(e)
            log.warning("traced record error: %s", e)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _push_stage(trace, {
            "stage": "memvid.record",
            "latency_ms": round(dt_ms, 2),
            "ts": _now(),
            "episode_id": episode.episode_id,
            "domain": episode.domain,
            "frame_id": getattr(episode, "frame_id", None),
            "commit": ok,
            "error": err,
        })
        return ok

    # -- traced context helper ---------------------------------------------
    def recall_as_context(self, query: str, *, domain: Optional[str] = None,
                          top_k: Optional[int] = None, max_chars: int = 1200,
                          trace: Optional[Dict[str, Any]] = None
                          ) -> str:
        eps = self.recall(query, domain=domain, top_k=top_k, trace=trace)
        if not eps:
            return ""
        lines = ["[PRIOR EPISODES — what Hermes already answered before]"]
        total = 0
        used = 0
        for i, ep in enumerate(eps, 1):
            if ep.score < self._inner.recall_threshold:
                continue
            block = (f"#{i} (score={ep.score:.2f}, {ep.created_at[:10]}, "
                     f"domain={ep.domain or '-'})\n"
                     f"Q: {ep.query}\nA: {ep.answer}")
            if total + len(block) > max_chars:
                break
            lines.append(block)
            total += len(block)
            used += 1
        # annotate the previously-pushed recall stage with usage info
        if trace and trace.get("stages"):
            last = trace["stages"][-1]
            last["used_in_prompt"] = used
        return "\n\n".join(lines) + "\n\n" if used else ""


# ---------------------------------------------------------------------------
# Helpers to consume trace in golden_eval_report
# ---------------------------------------------------------------------------
def summarize_stages(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a compact summary of memvid stages from a RagTrace.

    Useful when post-processing golden_eval_report.json for dashboards.
    """
    stages = trace.get("stages") or []
    recalls = [s for s in stages if s.get("stage") == "memvid.recall"]
    records = [s for s in stages if s.get("stage") == "memvid.record"]
    recall_lat = [s.get("latency_ms", 0) for s in recalls]
    record_lat = [s.get("latency_ms", 0) for s in records]
    hits = sum(s.get("hits", 0) for s in recalls)
    above = sum(1 for s in recalls if s.get("above_threshold"))
    used = sum(1 for s in recalls if s.get("used_in_prompt"))
    return {
        "memvid_recall_calls": len(recalls),
        "memvid_record_calls": len(records),
        "memvid_recall_total_hits": hits,
        "memvid_recall_above_threshold": above,
        "memvid_recall_used_in_prompt": used,
        "memvid_recall_latency_p50_ms": _pct(recall_lat, 50),
        "memvid_recall_latency_p99_ms": _pct(recall_lat, 99),
        "memvid_record_latency_mean_ms": (
            sum(record_lat) / len(record_lat)) if record_lat else 0.0,
        "memvid_errors": sum(1 for s in stages if s.get("error")),
    }


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return round(xs[k], 2)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke():
    import os
    os.environ.setdefault("RAG_MEMVID_ENABLED", "true")
    logging.basicConfig(level=logging.DEBUG)
    m = MemvidTraced(MemvidMemory.for_tenant("hermes_default"))
    trace: Dict[str, Any] = {"stages": []}
    eps = m.recall("сброс пароля astra", domain="astra", trace=trace)
    ep = Episode(query="сброс пароля astra", answer="recovery mode...",
                 domain="astra", tenant="hermes_default")
    m.record(ep, trace=trace)
    print(json_pretty(trace))
    print("summary:", summarize_stages(trace))


def json_pretty(o):
    import json
    return json.dumps(o, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    _smoke()
