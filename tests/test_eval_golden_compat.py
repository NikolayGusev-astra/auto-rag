"""Tests for eval_golden <-> memvid_canary compatibility (T2)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure rag_core is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "rag_core"))

from memvid_canary import _load_report


# -------------------------------------------------------------------
# Test A: canary tolerant parse
# -------------------------------------------------------------------
def test_canary_tolerant_parse() -> None:
    """_load_report should parse our eval_golden format (no score/latency_ms yet)."""
    fake_report = {
        "meta": {"version": "1.0"},
        "summary": {},
        "results": [
            {
                "id": "q1",
                "answer_verdict": "correct",
                "total_latency_s": 1.2,
            },
            {
                "id": "q2",
                "answer_verdict": "partial",
                "total_latency_s": 0.8,
            },
            {
                "id": "q3",
                "answer_verdict": "incorrect",
                "total_latency_s": 2.5,
            },
        ],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(fake_report, f)
        tmp_path = Path(f.name)

    try:
        result = _load_report(tmp_path, "cand")
        assert result is not None, "Should return ReportSummary"
        assert result.n == 3, f"Expected 3 questions, got {result.n}"
        # correct=1.0, partial=0.5, incorrect=0.0 => mean=(1.0+0.5+0.0)/3=0.5
        assert (
            abs(result.mean_score - 0.5) < 0.001
        ), f"mean_score should be ~0.5, got {result.mean_score}"
        # latencies: 1200, 800, 2500 => p99=2500
        assert (
            result.p99_latency > 0
        ), f"p99_latency should be >0, got {result.p99_latency}"
        assert (
            abs(result.p99_latency - 2500.0) < 0.1
        ), f"p99_latency should be ~2500, got {result.p99_latency}"
    finally:
        tmp_path.unlink()


# -------------------------------------------------------------------
# Test B: argparse --out and --golden flags
# -------------------------------------------------------------------
def test_argparse_help_has_new_flags() -> None:
    """eval_golden.py --help should list --out and --golden."""
    script = Path(__file__).parent.parent / "rag_core" / "eval_golden.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"help failed:\n{result.stderr}"
    combined = result.stdout + result.stderr
    assert "--out" in combined, f"--out not in help output:\n{combined}"
    assert "--golden" in combined, f"--golden not in help output:\n{combined}"