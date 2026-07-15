"""Tests for calibrate_judge.py — LLM judge calibration."""
import pytest
import json
import tempfile
from pathlib import Path


class TestCalibrateJudge:
    """Tests for the judge calibration module."""

    def test_imports(self):
        from calibrate_judge import load_or_create_calibration, compute_thresholds
        assert callable(load_or_create_calibration)
        assert callable(compute_thresholds)

    def test_load_or_create_no_file(self):
        """Без существующего файла создаёт дефолтный набор."""
        from calibrate_judge import load_or_create_calibration, CALIBRATION_PATH
        original = CALIBRATION_PATH
        try:
            # Temporarily point to a path that doesn't exist
            import calibrate_judge
            tmp = Path(tempfile.mktemp(suffix='.json'))
            calibrate_judge.CALIBRATION_PATH = tmp
            result = load_or_create_calibration()
            assert "questions" in result
            assert len(result["questions"]) > 0
            # Clean up
            if tmp.exists():
                tmp.unlink()
        finally:
            calibrate_judge.CALIBRATION_PATH = original

    def test_load_or_create_existing_file(self):
        """С существующим файлом загружает его."""
        from calibrate_judge import load_or_create_calibration
        import calibrate_judge
        
        original = calibrate_judge.CALIBRATION_PATH
        try:
            tmp = Path(tempfile.mktemp(suffix='.json'))
            test_data = {"questions": [{"id": "test", "query": "test?"}]}
            with open(tmp, 'w') as f:
                json.dump(test_data, f)
            calibrate_judge.CALIBRATION_PATH = tmp
            result = load_or_create_calibration()
            assert result["questions"][0]["id"] == "test"
            tmp.unlink()
        finally:
            calibrate_judge.CALIBRATION_PATH = original

    def test_compute_thresholds_perfect(self):
        """Идеальное совпадение → agreement = 1.0"""
        from calibrate_judge import compute_thresholds
        results = [
            {"llm_score": 0.95, "human_verdict": "correct"},
            {"llm_score": 0.85, "human_verdict": "correct"},
            {"llm_score": 0.60, "human_verdict": "partial"},
            {"llm_score": 0.30, "human_verdict": "incorrect"},
            {"llm_score": 0.10, "human_verdict": "incorrect"},
        ]
        thresholds = compute_thresholds(results)
        assert thresholds["agreement_rate"] >= 0.8

    def test_compute_thresholds_mixed(self):
        """Смешанные оценки → находит оптимальные пороги."""
        from calibrate_judge import compute_thresholds
        results = [
            {"llm_score": 0.90, "human_verdict": "correct"},
            {"llm_score": 0.70, "human_verdict": "correct"},
            {"llm_score": 0.50, "human_verdict": "partial"},
            {"llm_score": 0.30, "human_verdict": "incorrect"},
            {"llm_score": 0.10, "human_verdict": "incorrect"},
            {"llm_score": 0.80, "human_verdict": "partial"},  # mismatch
        ]
        thresholds = compute_thresholds(results)
        assert thresholds["recommended_correct_threshold"] > 0
        assert thresholds["recommended_partial_threshold"] > 0
        assert thresholds["recommended_correct_threshold"] > thresholds["recommended_partial_threshold"]

    def test_compute_thresholds_empty(self):
        """Пустой список → дефолтные пороги."""
        from calibrate_judge import compute_thresholds
        thresholds = compute_thresholds([])
        assert thresholds["agreement_rate"] == 0
        assert thresholds["recommended_correct_threshold"] == 0.7
        assert thresholds["recommended_partial_threshold"] == 0.4