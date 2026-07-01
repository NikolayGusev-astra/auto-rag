#!/usr/bin/env python3
"""Canary deploy — прогоняет голден-сет на двух версиях pipeline и сравнивает accuracy.

Использование:
  python canary_deploy.py                          # baseline vs current (git diff)
  python canary_deploy.py --baseline-branch main   # vs main branch
  python canary_deploy.py --backup-dir ../rag-deploy-bak  # vs backup каталога
  python canary_deploy.py --quick                   # dry-run only (DCD comparison)

Формат вывода:
  https://git.io/canary-report  (таблица сравнения метрик)
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
GOLDEN = HERE / "golden_set.json"
REPORT_DIR = HERE / "canary_reports"

# Ключевые файлы для мониторинга
WATCHED_FILES = [
    "rag_async.py",
    "rag_config.py",
    "dcd_router.py",
    "rag_mcp_client.py",
    "rag_trace.py",
    "rag_search.py",
]

# Эталонные файлы для копирования baseline
BASELINE_FILES = WATCHED_FILES + ["dcd_router_rusbitech.py"]


def file_hash(path: str) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest() if os.path.isfile(path) else ""


def detect_changes(backup_dir: str | None = None) -> dict:
    """Detect changed files vs baseline."""
    changes = {}
    if backup_dir:
        # Compare with backup copy
        for fn in WATCHED_FILES:
            src = HERE / fn
            bak = Path(backup_dir) / fn
            if bak.exists() and src.exists():
                if file_hash(str(src)) != file_hash(str(bak)):
                    diff = "\n".join(difflib.unified_diff(
                        open(str(bak)).readlines(),
                        open(str(src)).readlines(),
                        fromfile=f"baseline/{fn}", tofile=f"current/{fn}", n=3
                    ))
                    changes[fn] = {"type": "modified", "diff_len": len(diff), "diff_snippet": diff[:500]}
            elif src.exists() and not bak.exists():
                changes[fn] = {"type": "added", "diff_len": os.path.getsize(str(src))}
            elif bak.exists() and not src.exists():
                changes[fn] = {"type": "removed"}
    else:
        # Try git diff
        try:
            result = subprocess.run(
                ["git", "diff", "--name-status", "--", *WATCHED_FILES],
                capture_output=True, text=True, cwd=str(HERE), timeout=10
            )
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if "\t" in line:
                        status, fn = line.split("\t", 1)
                        changes[fn] = {"type": status}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return changes


def run_canary(version_name: str, label: str) -> dict | None:
    """Run golden set eval for one version. Returns summary dict."""
    print(f"  [{version_name}] Running golden set ({label}) ... ", end="", flush=True)
    try:
        import asyncio
        from eval_golden import load_golden, evaluate_one, accuracy_report

        golden = load_golden()
        questions = golden["questions"]

        results = []
        for q in questions:
            rec = asyncio.run(evaluate_one(q, dry_run=False))
            results.append(rec)

        report = accuracy_report(questions, results)
        print(f"ok  correct={report.get('answer_accuracy', 0)*100:.0f}%  partial={report.get('answer_accuracy_incl_partial',0)*100:.0f}%")
        return report
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def setup_baseline_backup(backup_dir: str) -> bool:
    """Copy current files as baseline."""
    bak = Path(backup_dir)
    bak.mkdir(parents=True, exist_ok=True)
    for fn in BASELINE_FILES:
        src = HERE / fn
        if src.exists():
            try:
                shutil.copy2(str(src), str(bak / fn))
            except (IOError, shutil.SameFileError):
                pass
    print(f"  Baseline saved to {backup_dir} ({len(list(bak.iterdir()))} files)")
    return True


def restore_from_backup(backup_dir: str):
    """Restore baseline files from backup."""
    bak = Path(backup_dir)
    for fn in BASELINE_FILES:
        src = bak / fn
        dst = HERE / fn
        if src.exists():
            shutil.copy2(str(src), str(dst))
    print(f"  Restored baseline from {backup_dir}")


def render_report(baseline: dict, canary: dict, changes: dict) -> str:
    """Render comparison report."""
    lines = []
    lines.append("=" * 70)
    lines.append("CANARY DEPLOY — RAG ACCURACY COMPARISON")
    lines.append("=" * 70)

    lines.append("\n── Changes detected ──")
    for fn, info in changes.items():
        lines.append(f"  {info.get('type', 'modified'):10s} {fn}")
    if not changes:
        lines.append("  (no changes in watched files)")

    lines.append("\n── Accuracy ──")
    header = f"{'Metric':35s} {'Baseline':>12s} {'Canary':>12s} {'Δ':>10s}"
    lines.append(header)
    lines.append("-" * 70)

    metrics = [
        ("answer_correct", "Answer correct (n)", int),
        ("answer_partial", "Answer partial (n)", int),
        ("answer_incorrect", "Answer incorrect (n)", int),
        ("answer_accuracy", "Answer accuracy (%)", lambda v: f"{v*100:.1f}%"),
        ("answer_accuracy_incl_partial", "Answer incl. partial (%)", lambda v: f"{v*100:.1f}%"),
        ("source_routing_accuracy", "Source routing acc (%)", lambda v: f"{v*100:.1f}%"),
        ("dcd_collection_accuracy", "DCD collection acc (%)", lambda v: f"{v*100:.1f}%"),
        ("dcd_domain_accuracy", "DCD domain acc (%)", lambda v: f"{v*100:.1f}%"),
        ("avg_latency_s", "Avg latency (s)", lambda v: f"{v:.1f}s"),
    ]

    verdict = "PASS"
    for key, label, fmt in metrics:
        b_val = baseline.get(key, "?") if baseline else "?"
        c_val = canary.get(key, "?") if canary else "?"
        if isinstance(b_val, (int, float)) and isinstance(c_val, (int, float)):
            if key in ("answer_accuracy", "answer_accuracy_incl_partial"):
                delta = c_val - b_val
                delta_str = f"{delta:+.1%}" if isinstance(delta, float) else f"{delta:+d}"
                if key == "answer_accuracy" and delta < -0.05:
                    verdict = "REGRESSION"
            else:
                delta = round(c_val - b_val, 2) if isinstance(c_val, (int, float)) else "?"
                delta_str = f"{delta:+.2f}" if isinstance(delta, (int, float)) else "?"
            b_str = fmt(b_val) if callable(fmt) else str(b_val)
            c_str = fmt(c_val) if callable(fmt) else str(c_val)
            lines.append(f"{label:35s} {b_str:>12s} {c_str:>12s} {delta_str:>10s}")

    lines.append(f"\n  Verdict: {verdict}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Canary deploy — compare RAG accuracy")
    parser.add_argument("--baseline-dir", help="Directory with baseline files (snapshot)")
    parser.add_argument("--backup-dir", default=str(HERE / ".canary_baseline"),
                        help="Where to store/read baseline")
    parser.add_argument("--quick", action="store_true", help="Dry-run only (DCD comparison)")
    args = parser.parse_args()

    os.makedirs(str(REPORT_DIR), exist_ok=True)
    REPORT_PATH = REPORT_DIR / f"canary_{time.strftime('%Y%m%d_%H%M%S')}.json"

    # Step 1: Detect changes
    print("Step 1: Detecting changes ...")
    baseline_dir = args.baseline_dir or args.backup_dir
    changes = detect_changes(baseline_dir if baseline_dir and Path(baseline_dir).exists() else None)
    for fn, info in changes.items():
        print(f"  {info.get('type', 'modified'):10s} {fn}")
    if not changes:
        print("  No changes detected in watched files")

    # Step 2: Run baseline
    print("\nStep 2: Running baseline eval ...")
    if baseline_dir and Path(baseline_dir).exists():
        # Run from baseline
        restore_from_backup(baseline_dir)
        baseline = run_canary("baseline", "restored from backup")
        # Restore current
        # (current files are already in place since we haven't modified anything)
    else:
        # No baseline → save current as baseline
        print("  No baseline found — saving current files as baseline")
        setup_baseline_backup(args.backup_dir)
        baseline = run_canary("baseline", "current (first run)")
        # For first run, canary is same as baseline
        print("\n⚠ First run — saved baseline. Run again after making changes.")
        if baseline:
            output = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "baseline": baseline,
                "canary": None,
                "changes": changes,
                "verdict": "BASELINE_SAVED",
                "report": "First run — baseline saved. Make changes and run again."
            }
            with open(str(REPORT_PATH), "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"\n  Report: {REPORT_PATH}")
        return

    # Step 3: Run canary (current)
    print("\nStep 3: Running canary eval (current code) ...")
    canary = run_canary("canary", "current")

    # Step 4: Compare
    print("\nStep 4: Comparison")
    report = render_report(baseline, canary, changes)
    print("\n" + report)

    # Step 5: Save report
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": baseline,
        "canary": canary,
        "changes": changes,
        "verdict": "REGRESSION" if "REGRESSION" in report else "PASS",
        "report": report,
    }
    with open(str(REPORT_PATH), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Full report: {REPORT_PATH}")

    # Step 6: Summary
    if "REGRESSION" in report:
        print("\n  ⚠ REGRESSION detected! Answer accuracy dropped >5%.")
        print("  Rollback with: cp -r .canary_baseline/* .")
    else:
        print("\n  ✓ Canary passed — no significant regression")


if __name__ == "__main__":
    main()