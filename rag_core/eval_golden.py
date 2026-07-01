#!/usr/bin/env python3
"""Eval RAG accuracy against golden set.

Запуск:
  python eval_golden.py                         # полный прогон
  python eval_golden.py --dry-run               # только DCD + source routing
  python eval_golden.py --id aldpro-ip-change   # один вопрос

Результат:
  - golden_eval_report.json — детально по каждому вопросу
  - stdout — сводка метрик
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dcd_router import classify as dcd_classify
from rag_async import async_rag_search

HERE = Path(__file__).parent
GOLDEN_PATH = HERE / "golden_set.json"
REPORT_PATH = HERE / "golden_eval_report.json"


def load_golden() -> dict:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def check_key_facts(text: str, key_facts: list[str]) -> dict:
    """Check which key_facts appear in text (case-insensitive)."""
    tl = text.lower()
    results = []
    for fact in key_facts:
        found = fact.lower() in tl
        results.append({"fact": fact, "found": found})
    n_found = sum(1 for r in results if r["found"])
    return {
        "key_facts_total": len(key_facts),
        "key_facts_found": n_found,
        "key_facts_ratio": n_found / max(len(key_facts), 1),
        "details": results,
    }


def eval_answer_match(chunks_text: str, key_facts: list[str]) -> dict:
    """Evaluate answer accuracy via key facts."""
    result = check_key_facts(chunks_text, key_facts)
    ratio = result["key_facts_ratio"]
    if ratio >= 0.8:
        verdict = "correct"
    elif ratio >= 0.5:
        verdict = "partial"
    elif ratio > 0:
        verdict = "weak"
    else:
        verdict = "incorrect"
    result["verdict"] = verdict
    return result


def accuracy_report(questions: list[dict], results: list[dict]) -> dict:
    totals = {"total": len(questions), "evaluated": 0, "errors": 0}

    # Source routing accuracy
    src_correct = sum(1 for r in results if r.get("source_ok"))
    totals["source_routing_accuracy"] = round(src_correct / max(len(results), 1), 4)

    # DCD domain accuracy
    dom_correct = sum(
        1 for r in results if r.get("dcd_domain_ok")
    )
    totals["dcd_domain_accuracy"] = round(dom_correct / max(len(results), 1), 4)

    # DCD collection accuracy
    coll_correct = sum(
        1 for r in results if r.get("dcd_collection_ok")
    )
    totals["dcd_collection_accuracy"] = round(coll_correct / max(len(results), 1), 4)

    # Answer accuracy (key facts)
    answer_correct = sum(1 for r in results if r.get("answer_verdict") == "correct")
    answer_partial = sum(1 for r in results if r.get("answer_verdict") == "partial")
    totals["answer_correct"] = answer_correct
    totals["answer_partial"] = answer_partial
    totals["answer_incorrect"] = sum(
        1 for r in results if r.get("answer_verdict") == "incorrect"
    )
    totals["answer_accuracy"] = round(answer_correct / max(len(results), 1), 4)
    totals["answer_accuracy_incl_partial"] = round(
        (answer_correct + answer_partial) / max(len(results), 1), 4
    )

    # Source breakdown
    source_counts = {}
    for r in results:
        src = r.get("actual_source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    totals["source_breakdown"] = source_counts

    # By collection
    coll_stats: dict[str, dict] = {}
    for q, r in zip(questions, results):
        coll = q.get("expected_collection", "unknown")
        if coll not in coll_stats:
            coll_stats[coll] = {"count": 0, "correct": 0, "correct_total": 0.0}
        coll_stats[coll]["count"] += 1
        if r.get("source_ok"):
            coll_stats[coll]["correct"] += 1
        if r.get("answer_verdict") in ("correct", "partial"):
            coll_stats[coll]["correct_total"] += 1
    totals["by_collection"] = {
        coll: {
            "n": stats["count"],
            "source_accuracy": round(stats["correct"] / max(stats["count"], 1), 3),
            "answer_ok_pct": round(stats["correct_total"] / max(stats["count"], 1) * 100, 1),
        }
        for coll, stats in sorted(coll_stats.items())
    }

    # Average latency
    latencies = [r.get("latency", 0) for r in results if r.get("latency")]
    totals["avg_latency_s"] = round(sum(latencies) / max(len(latencies), 1), 2)
    totals["max_latency_s"] = round(max(latencies), 2) if latencies else 0

    totals["evaluated"] = len(results)
    return totals


async def evaluate_one(
    q: dict, dry_run: bool = False
) -> dict:
    """Evaluate a single golden set question."""
    rec: dict = {
        "id": q["id"],
        "query": q["query"],
        "expected_domain": q["expected_domain"],
        "expected_collection": q["expected_collection"],
        "expected_source": q["expected_source"],
    }

    # 1. DCD classification
    t0 = time.time()
    try:
        dcd_result = dcd_classify(q["query"])
        dcd_time = time.time() - t0
        rec["dcd_domain"] = dcd_result.get("domain", "")
        rec["dcd_collection"] = dcd_result.get("collection", "")
        rec["dcd_confidence"] = dcd_result.get("confidence", 0)
        rec["dcd_domain_ok"] = (
            dcd_result.get("domain") == q["expected_domain"]
        )
        rec["dcd_collection_ok"] = (
            dcd_result.get("collection") == q["expected_collection"]
        )
        rec["dcd_latency_s"] = round(dcd_time, 3)
    except Exception as e:
        rec["dcd_error"] = str(e)
        rec["dcd_domain_ok"] = False
        rec["dcd_collection_ok"] = False

    if dry_run:
        rec["dry_run"] = True
        return rec

    # 2. RAG pipeline
    t1 = time.time()
    try:
        from rag_trace import RagTrace
        trace = RagTrace(q["query"])
        result = await async_rag_search(q["query"], dcd_result, trace=trace)
        latency = time.time() - t1

        rec["actual_source"] = result.get("source", "empty")
        rec["trace"] = result.get("trace", "")
        rec["score"] = result.get("score", 0)
        rec["latency"] = round(latency, 2)
        rec["_trace_stages"] = trace.stages  # full trace for diagnostic
        rec["_trace_summary"] = trace.summary()

        # Chunks text for evaluation
        chunks = result.get("chunks", [])
        chunks_text = "\n".join(
            [c.get("text", "") for c in chunks]
        )
        rec["n_chunks"] = len(chunks)
        rec["chunks_snippet"] = chunks_text[:500] if chunks_text else ""
        rec["chunks_full_len"] = len(chunks_text)

        # Source routing accuracy
        rec["source_ok"] = result.get("source") == q["expected_source"]

        # 3. Answer accuracy (key facts)
        answer_eval = eval_answer_match(chunks_text, q["key_facts"])
        rec["answer_verdict"] = answer_eval["verdict"]
        rec["answer_key_facts"] = answer_eval
        rec["total_latency_s"] = round(latency + dcd_time, 2)

    except Exception as e:
        rec["error"] = str(e)
        rec["source_ok"] = False
        rec["answer_verdict"] = "error"
        rec["latency"] = round(time.time() - t1, 2)

    return rec


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Eval RAG golden set")
    parser.add_argument("--dry-run", action="store_true", help="Only DCD + source check")
    parser.add_argument("--id", type=str, help="Run single question by id")
    args = parser.parse_args()

    golden = load_golden()
    questions = golden["questions"]

    if args.id:
        questions = [q for q in questions if q["id"] == args.id]
        if not questions:
            print(f"❌ Question id='{args.id}' not found")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"RAG Accuracy Eval — Golden Set")
    print(f"Questions: {len(questions)}")
    print(f"Mode: {'DRY RUN (DCD only)' if args.dry_run else 'FULL'}")
    print(f"{'='*60}\n")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['id']} ... ", end="", flush=True)
        try:
            rec = await evaluate_one(q, dry_run=args.dry_run)
            verdict = rec.get("answer_verdict", rec.get("source_ok"))
            if args.dry_run:
                dcd_ok = (
                    "✓" if rec.get("dcd_domain_ok") and rec.get("dcd_collection_ok") else "✗"
                )
                print(f"DCD={dcd_ok}  coll={rec.get('dcd_collection','?')} src={rec.get('expected_source')}")
            else:
                src_ok = "✓" if rec.get("source_ok") else "✗"
                ans = rec.get("answer_verdict", "?")
                lat = rec.get("total_latency_s", 0)
                print(f"source={src_ok} answer={ans} ({lat:.1f}s)")
            results.append(rec)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"id": q["id"], "error": str(e)})

    # Summary
    report = accuracy_report(questions, results)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Questions evaluated:  {report['evaluated']}/{report['total']}")
    print(f"  Errors:               {report['errors']}")
    print(f"  Source routing acc:   {report['source_routing_accuracy']*100:.1f}%")
    print(f"  DCD domain accuracy:  {report['dcd_domain_accuracy']*100:.1f}%")
    print(f"  DCD collection acc:   {report['dcd_collection_accuracy']*100:.1f}%")
    print(f"  Answer accuracy:      {report['answer_accuracy']*100:.1f}%")
    print(f"  Answer (incl.partial):{report['answer_accuracy_incl_partial']*100:.1f}%")
    print(f"  Avg latency:          {report['avg_latency_s']:.1f}s")
    print(f"  Max latency:          {report['max_latency_s']:.1f}s")

    print(f"\n  Per-collection:")
    for coll, stats in report.get("by_collection", {}).items():
        print(f"    {coll:30s} n={stats['n']:2d}  "
              f"src={stats['source_accuracy']*100:.0f}%  "
              f"ok={stats['answer_ok_pct']:.0f}%")

    print(f"\n  Source breakdown:")
    for src, cnt in report.get("source_breakdown", {}).items():
        print(f"    {src:20s}: {cnt}")

    # Save report
    output = {
        "meta": golden["meta"],
        "summary": report,
        "results": results,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Full report saved: {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
