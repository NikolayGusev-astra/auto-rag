"""
memvid_canary.py — Automated A/B runner for memvid integration.

Runs `eval_golden.py` twice (baseline vs memvid-on), then compares the
two `golden_eval_report.json` files and emits a verdict:

  - accuracy delta (LLM-as-Judge score)
  - latency P50/P99 delta
  - cache/memory hit-rate (when reported in trace)
  - per-question regressions (auto-flag if any single question drops >5%)
  - auto-rollback recommendation if overall regression > threshold

This mirrors the semantics of `rag_core/canary_deploy.py` but is scoped
to the memvid memory layer, so you can promote/rollback just the memory
feature without touching the rest of the pipeline.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------

    # full run (baseline + candidate)
    python3 memvid_canary.py run \\
        --golden rag_core/golden_set.json \\
        --out-dir ./canary/memvid_$(date +%Y%m%d)

    # compare two already-existing reports
    python3 memvid_canary.py compare \\
        --baseline ./canary/baseline/golden_eval_report.json \\
        --candidate ./canary/memvid/golden_eval_report.json

    # quick smoke (10 questions)
    python3 memvid_canary.py run --quick --out-dir ./canary/quick

    # promote candidate (copy capsule to 'production' slot)
    python3 memvid_canary.py promote \\
        --capsule ./memvid_capsules/memory_hermes_default.mv2 \\
        --to ./memvid_capsules/production/memory_hermes_default.mv2

    # rollback (restore previous capsule)
    python3 memvid_canary.py rollback \\
        --tenant hermes_default \\
        --backup-dir ./memvid_capsules/backup

------------------------------------------------------------------------------
ENV
------------------------------------------------------------------------------
  RAG_MEMVID_CANARY_THRESHOLD   regression threshold (default 0.05 = 5%)
  RAG_MEMVID_CANARY_GOLDEN_CMD  override the eval command template
                                default: python3 rag_core/eval_golden.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REGRESSION_THRESHOLD = float(
    os.environ.get("RAG_MEMVID_CANARY_THRESHOLD", "0.05"))

DEFAULT_EVAL_CMD = os.environ.get(
    R"RAG_MEMVID_CANARY_GOLDEN_CMD",
    "python3 rag_core/eval_golden.py")


# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------
@dataclass
class QuestionResult:
    qid: str
    question: str
    score: float            # LLM-as-Judge score, 0..1
    latency_ms: float
    from_memory: bool = False
    sources_n: int = 0
    answer: str = ""


@dataclass
class ReportSummary:
    name: str
    path: Path
    n: int
    mean_score: float
    median_score: float
    p50_latency: float
    p99_latency: float
    mean_latency: float
    memory_hit_rate: float
    per_q: List[QuestionResult] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "n": self.n,
            "mean_score": round(self.mean_score, 4),
            "median_score": round(self.median_score, 4),
            "p50_latency_ms": round(self.p50_latency, 1),
            "p99_latency_ms": round(self.p99_latency, 1),
            "mean_latency_ms": round(self.mean_latency, 1),
            "memory_hit_rate": round(self.memory_hit_rate, 4),
        }


def _load_report(path: Path, name: str) -> Optional[ReportSummary]:
    """Parse golden_eval_report.json. Tolerant to schema drift — auto-rag's
    report shape may vary; we probe common keys."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[canary] cannot parse {path}: {e}", file=sys.stderr)
        return None

    # auto-rag golden_eval_report.json usual shapes:
    #  { "questions": [ { "id","question","score","latency_ms","trace":{...} } ],
    #    "summary": {...} }
    # OR a flat list. Probe defensively.
    questions: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        questions = raw
    elif isinstance(raw, dict):
        questions = (raw.get("questions") or raw.get("results")
                     or raw.get("evaluations") or [])
        if not questions and "question" in raw:
            questions = [raw]

    per_q: List[QuestionResult] = []
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or q.get("qid") or i)
        _verdict = (q.get("answer_verdict") or "").lower()
        _score_from_verdict = (
            1.0 if _verdict == "correct" else 0.5 if _verdict == "partial" else 0.0)
        score = float(q.get("score") or q.get("llm_judge_score")
                      or q.get("accuracy") or _score_from_verdict or 0.0)
        lat = float(q.get("latency_ms") or q.get("latency")
                    or (q.get("total_latency_s", 0) * 1000)
                    or _trace_latency(q.get("trace")) or 0.0)
        trace = q.get("trace") or {}
        from_mem = bool(_trace_flag(trace, "from_memory")
                        or q.get("from_memory", False))
        srcs = q.get("sources") or trace.get("sources") or []
        per_q.append(QuestionResult(
            qid=qid,
            question=str(q.get("question") or q.get("query") or ""),
            score=score,
            latency_ms=lat,
            from_memory=from_mem,
            sources_n=len(srcs) if isinstance(srcs, list) else 0,
            answer=str(q.get("answer") or "")[:200],
        ))

    if not per_q:
        return None

    scores = [q.score for q in per_q]
    lats = [q.latency_ms for q in per_q if q.latency_ms > 0]
    mem_hits = sum(1 for q in per_q if q.from_memory)

    def pct(xs: List[float], p: float) -> float:
        if not xs:
            return 0.0
        xs = sorted(xs)
        k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
        return xs[k]

    return ReportSummary(
        name=name,
        path=path,
        n=len(per_q),
        mean_score=statistics.mean(scores),
        median_score=statistics.median(scores),
        p50_latency=pct(lats, 50),
        p99_latency=pct(lats, 99),
        mean_latency=statistics.mean(lats) if lats else 0.0,
        memory_hit_rate=mem_hits / len(per_q) if per_q else 0.0,
        per_q=per_q,
    )


def _trace_latency(trace: Any) -> float:
    if not isinstance(trace, dict):
        return 0.0
    # RagTrace may carry total_latency_ms or stages[].latency_ms
    if "total_latency_ms" in trace:
        return float(trace["total_latency_ms"])
    stages = trace.get("stages") or []
    return float(sum(s.get("latency_ms", 0) for s in stages
                     if isinstance(s, dict)))


def _trace_flag(trace: Any, key: str) -> bool:
    if not isinstance(trace, dict):
        return False
    if key in trace:
        return bool(trace[key])
    for s in (trace.get("stages") or []):
        if isinstance(s, dict) and s.get(key):
            return True
    return False


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
@dataclass
class Diff:
    baseline: ReportSummary
    candidate: ReportSummary
    delta_mean_score: float
    delta_median_score: float
    delta_p50_latency: float
    delta_p99_latency: float
    delta_memory_hit_rate: float
    regressions: List[Tuple[str, float, float]]   # (qid, base, cand)
    improvements: List[Tuple[str, float, float]]
    verdict: str            # "promote" | "rollback" | "neutral"
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline.as_dict(),
            "candidate": self.candidate.as_dict(),
            "delta_mean_score": round(self.delta_mean_score, 4),
            "delta_median_score": round(self.delta_median_score, 4),
            "delta_p50_latency_ms": round(self.delta_p50_latency, 1),
            "delta_p99_latency_ms": round(self.delta_p99_latency, 1),
            "delta_memory_hit_rate": round(self.delta_memory_hit_rate, 4),
            "regressions": [
                {"qid": q, "baseline": b, "candidate": c}
                for q, b, c in self.regressions],
            "improvements": [
                {"qid": q, "baseline": b, "candidate": c}
                for q, b, c in self.improvements],
            "verdict": self.verdict,
            "reason": self.reason,
        }


def compare(baseline: ReportSummary, candidate: ReportSummary,
            threshold: float = REGRESSION_THRESHOLD) -> Diff:
    d_mean = candidate.mean_score - baseline.mean_score
    d_med = candidate.median_score - baseline.median_score
    d_p50 = candidate.p50_latency - baseline.p50_latency
    d_p99 = candidate.p99_latency - baseline.p99_latency
    d_mem = candidate.memory_hit_rate - baseline.memory_hit_rate

    base_map = {q.qid: q for q in baseline.per_q}
    regressions: List[Tuple[str, float, float]] = []
    improvements: List[Tuple[str, float, float]] = []
    for cq in candidate.per_q:
        bq = base_map.get(cq.qid)
        if not bq:
            continue
        delta = cq.score - bq.score
        if delta <= -threshold:
            regressions.append((cq.qid, bq.score, cq.score))
        elif delta >= threshold:
            improvements.append((cq.qid, bq.score, cq.score))
    regressions.sort(key=lambda t: t[1] - t[2], reverse=True)
    improvements.sort(key=lambda t: t[2] - t[1], reverse=True)

    # verdict
    if d_mean <= -threshold and len(regressions) > max(1, 0.1 * baseline.n):
        verdict, reason = "rollback", (
            f"mean score dropped {d_mean:.3f} (<= -{threshold}) with "
            f"{len(regressions)} per-question regressions")
    elif d_mean > 0 and d_p50 <= 0.5 * baseline.p50_latency:
        verdict, reason = "promote", (
            f"mean score +{d_mean:.3f}, latency not worse "
            f"(Δp50={d_p50:.1f}ms). {len(improvements)} improvements.")
    elif d_mean >= 0:
        verdict, reason = "promote", (
            f"mean score +{d_mean:.3f}, no overall regression. "
            f"{len(improvements)} improvements, {len(regressions)} regressions.")
    else:
        verdict, reason = "neutral", (
            f"marginal change (Δmean={d_mean:.3f}); review per-question "
            f"diffs manually")

    return Diff(
        baseline=baseline, candidate=candidate,
        delta_mean_score=d_mean, delta_median_score=d_med,
        delta_p50_latency=d_p50, delta_p99_latency=d_p99,
        delta_memory_hit_rate=d_mem,
        regressions=regressions, improvements=improvements,
        verdict=verdict, reason=reason,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_eval(env_overrides: Dict[str, str], out_dir: Path,
              golden: Optional[Path], quick: bool,
              label: str) -> Optional[Path]:
    """Run eval_golden.py with given env overrides; return report path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_overrides.items()})

    cmd = DEFAULT_EVAL_CMD.split()
    if golden:
        cmd += ["--golden", str(golden)]
    if quick:
        cmd += ["--quick"]
    cmd += ["--out", str(out_dir / "golden_eval_report.json")]

    print(f"[canary] [{label}] running: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=env, timeout=60 * 30)
    except subprocess.TimeoutExpired:
        print(f"[canary] [{label}] TIMEOUT", file=sys.stderr)
        return None
    dt = time.perf_counter() - t0
    print(f"[canary] [{label}] exit={proc.returncode} in {dt:.1f}s",
          flush=True)
    report = out_dir / "golden_eval_report.json"
    return report if report.exists() else None


def run_canary(golden: Optional[Path], out_dir: Path, quick: bool,
               capsule_dir: Optional[Path] = None) -> Diff:
    """Run baseline (memvid off) and candidate (memvid on), then compare."""
    base_dir = out_dir / "baseline"
    cand_dir = out_dir / "candidate"

    # BACKUP existing capsule so baseline truly has no memory influence
    capsule_dir = capsule_dir or Path(
        os.environ.get("RAG_MEMVID_DIR", "./memvid_capsules"))
    backup_capsule: Optional[Path] = None
    if capsule_dir.exists():
        backup_capsule = out_dir / "_capsule_backup"
        backup_capsule.mkdir(parents=True, exist_ok=True)
        for f in capsule_dir.glob("memory_*.mv2*"):
            shutil.copy2(f, backup_capsule / f.name)
        # remove active capsule for baseline run
        for f in capsule_dir.glob("memory_*.mv2*"):
            f.unlink()
        print(f"[canary] backed up {capsule_dir} -> {backup_capsule}")

    try:
        # BASELINE: memvid OFF
        base_report = _run_eval(
            env_overrides={"RAG_MEMVID_ENABLED": "false"},
            out_dir=base_dir, golden=golden, quick=quick, label="baseline")

        # CANDIDATE: memvid ON, fresh capsule
        if backup_capsule:
            # restore capsule for candidate run so memvid can use/warm it
            for f in backup_capsule.glob("memory_*.mv2*"):
                shutil.copy2(f, capsule_dir / f.name)
        cand_report = _run_eval(
            env_overrides={"RAG_MEMVID_ENABLED": "true",
                           "RAG_MEMVID_MODE": "both"},
            out_dir=cand_dir, golden=golden, quick=quick, label="candidate")

        if not base_report or not cand_report:
            raise RuntimeError(
                f"missing reports: base={base_report} cand={cand_report}")

        base = _load_report(base_report, "baseline")
        cand = _load_report(cand_report, "candidate")
        if not base or not cand:
            raise RuntimeError("failed to parse one of the reports")

        diff = compare(base, cand)
        (out_dir / "canary_diff.json").write_text(
            json.dumps(diff.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8")
        _print_diff(diff)
        return diff
    finally:
        # ensure capsule restored even on failure
        if backup_capsule and capsule_dir.exists():
            present = {f.name for f in capsule_dir.glob("memory_*.mv2*")}
            for f in backup_capsule.glob("memory_*.mv2*"):
                if f.name not in present:
                    shutil.copy2(f, capsule_dir / f.name)


def _print_diff(d: Diff) -> None:
    print("\n" + "=" * 70)
    print("CANARY DIFF  (candidate - baseline)")
    print("=" * 70)
    print(f"  mean score      : {d.delta_mean_score:+.4f}")
    print(f"  median score    : {d.delta_median_score:+.4f}")
    print(f"  p50 latency ms  : {d.delta_p50_latency:+.1f}")
    print(f"  p99 latency ms  : {d.delta_p99_latency:+.1f}")
    print(f"  memory hit rate : {d.delta_memory_hit_rate:+.4f}")
    print(f"  improvements    : {len(d.improvements)}")
    print(f"  regressions     : {len(d.regressions)}")
    for qid, b, c in d.regressions[:5]:
        print(f"    [REGRESS] {qid}: {b:.3f} -> {c:.3f}")
    print("-" * 70)
    print(f"  VERDICT: {d.verdict.upper()}")
    print(f"  {d.reason}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Promote / rollback
# ---------------------------------------------------------------------------
def promote(capsule: Path, to: Path) -> None:
    to.parent.mkdir(parents=True, exist_ok=True)
    # keep one previous as .bak
    if to.exists():
        bak = to.with_suffix(to.suffix + ".bak")
        shutil.copy2(to, bak)
        print(f"[canary] previous production capsule backed up -> {bak}")
    shutil.copy2(capsule, to)
    print(f"[canary] promoted {capsule} -> {to}")


def rollback(tenant: str, backup_dir: Path,
             capsule_dir: Optional[Path] = None) -> None:
    capsule_dir = capsule_dir or Path(
        os.environ.get("RAG_MEMVID_DIR", "./memvid_capsules"))
    # find latest .bak in capsule_dir first, then backup_dir
    candidates = sorted(
        list(capsule_dir.glob(f"memory_{tenant}.mv2*.bak")) +
        list(backup_dir.glob(f"memory_{tenant}.mv2*")),
        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"no backup capsule found for tenant={tenant}")
    src = candidates[0]
    dst = capsule_dir / f"memory_{tenant}{src.suffix.replace('.bak','')}"
    shutil.copy2(src, dst)
    print(f"[canary] rollback: {src} -> {dst}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        prog="memvid_canary",
        description="A/B canary runner for memvid memory layer in auto-rag")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run baseline vs candidate")
    p_run.add_argument("--golden", type=Path, default=None)
    p_run.add_argument("--out-dir", type=Path, required=True)
    p_run.add_argument("--quick", action="store_true")
    p_run.add_argument("--capsule-dir", type=Path, default=None)

    p_cmp = sub.add_parser("compare", help="compare two existing reports")
    p_cmp.add_argument("--baseline", type=Path, required=True)
    p_cmp.add_argument("--candidate", type=Path, required=True)
    p_cmp.add_argument("--threshold", type=float, default=REGRESSION_THRESHOLD)

    p_promo = sub.add_parser("promote", help="promote a capsule to production")
    p_promo.add_argument("--capsule", type=Path, required=True)
    p_promo.add_argument("--to", type=Path, required=True)

    p_rb = sub.add_parser("rollback", help="restore previous capsule")
    p_rb.add_argument("--tenant", default="hermes_default")
    p_rb.add_argument("--backup-dir", type=Path, required=True)
    p_rb.add_argument("--capsule-dir", type=Path, default=None)

    args = ap.parse_args()

    if args.cmd == "run":
        diff = run_canary(args.golden, args.out_dir, args.quick,
                          args.capsule_dir)
        sys.exit(0 if diff.verdict != "rollback" else 1)
    elif args.cmd == "compare":
        b = _load_report(args.baseline, "baseline")
        c = _load_report(args.candidate, "candidate")
        if not b or not c:
            print("cannot load one of the reports", file=sys.stderr)
            sys.exit(2)
        d = compare(b, c, threshold=args.threshold)
        _print_diff(d)
        json.dump(d.as_dict(), sys.stdout, indent=2, ensure_ascii=False)
        sys.exit(0 if d.verdict != "rollback" else 1)
    elif args.cmd == "promote":
        promote(args.capsule, args.to)
    elif args.cmd == "rollback":
        rollback(args.tenant, args.backup_dir, args.capsule_dir)


if __name__ == "__main__":
    main()