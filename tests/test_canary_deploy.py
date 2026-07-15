"""Tests for canary_deploy.py — embedding stability and ML metrics."""
import pytest
import json
import tempfile
from pathlib import Path


# ── Tests for cosine_sim ──────────────────────────────────────────
class TestCosineSim:
    def test_canary_imports(self):
        """Canary module imports cleanly (no runtime deps outside functions)."""
        from canary_deploy import collect_queries_from_golden, compare_embedding_stability
        assert callable(collect_queries_from_golden)
        assert callable(compare_embedding_stability)

    def test_collect_queries_no_golden(self):
        """Без golden_set.json возвращает пустой список."""
        from canary_deploy import collect_queries_from_golden
        # Temporarily point to non-existent file
        import canary_deploy
        original = canary_deploy.GOLDEN
        canary_deploy.GOLDEN = Path("/tmp/nonexistent_golden.json")
        try:
            result = collect_queries_from_golden()
            assert result == []
        finally:
            canary_deploy.GOLDEN = original

    def test_collect_queries_with_golden(self):
        """С golden_set.json возвращает список запросов."""
        from canary_deploy import collect_queries_from_golden
        import canary_deploy
        
        # Create temp golden set
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({
            "questions": [
                {"id": "q1", "query": "What is RAG?"},
                {"id": "q2", "query": "How does chunking work?"},
            ]
        }, tmp)
        tmp.close()
        
        original = canary_deploy.GOLDEN
        canary_deploy.GOLDEN = Path(tmp.name)
        try:
            result = collect_queries_from_golden()
            assert len(result) == 2
            assert "What is RAG?" in result
            assert "How does chunking work?" in result
        finally:
            canary_deploy.GOLDEN = original
            Path(tmp.name).unlink()

    def test_empty_golden_set(self):
        """Пустой golden set не вызывает ошибок."""
        from canary_deploy import collect_queries_from_golden
        import canary_deploy
        
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({"questions": []}, tmp)
        tmp.close()
        
        original = canary_deploy.GOLDEN
        canary_deploy.GOLDEN = Path(tmp.name)
        try:
            result = collect_queries_from_golden()
            assert result == []
        finally:
            canary_deploy.GOLDEN = original
            Path(tmp.name).unlink()

    def test_compare_embedding_stability_empty(self):
        """Пустой список запросов → error dict."""
        from canary_deploy import compare_embedding_stability
        result = compare_embedding_stability([])
        assert "error" in result
        assert result["mean_similarity"] == 0.0

    def test_compare_embedding_stability_no_embeddings(self):
        """Если эмбеддинг-сервер не отвечает → error."""
        from canary_deploy import compare_embedding_stability
        import canary_deploy
        
        original_url = canary_deploy.EMBEDDING_URL
        canary_deploy.EMBEDDING_URL = "http://localhost:1/nonexistent"
        try:
            result = compare_embedding_stability(["test query"])
            assert "error" in result
        finally:
            canary_deploy.EMBEDDING_URL = original_url

    def test_cosine_symmetry(self):
        """cosine_sim(a, b) == cosine_sim(b, a)"""
        from canary_deploy import cosine_sim
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_sim(a, b) - cosine_sim(b, a)) < 1e-10

    def test_cosine_identical(self):
        """cosine_sim(a, a) == 1.0"""
        from canary_deploy import cosine_sim
        a = [0.5, 0.3, 0.2]
        assert abs(cosine_sim(a, a) - 1.0) < 1e-10

    def test_cosine_orthogonal(self):
        """cosine_sim(orthogonal) ≈ 0.0"""
        from canary_deploy import cosine_sim
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_sim(a, b)) < 1e-10

    def test_cosine_zero_vector(self):
        """cosine_sim(zero, anything) == 0.0"""
        from canary_deploy import cosine_sim
        zero = [0.0, 0.0, 0.0]
        a = [1.0, 2.0, 3.0]
        assert cosine_sim(zero, a) == 0.0
        assert cosine_sim(a, zero) == 0.0


class TestRenderReport:
    """Tests for the canary report rendering."""

    def test_render_report_basic(self):
        """Базовый рендер не падает."""
        from canary_deploy import render_report
        baseline = {"answer_accuracy": 0.85, "answer_accuracy_incl_partial": 0.90,
                    "answer_correct": 17, "answer_partial": 3, "answer_incorrect": 0,
                    "source_routing_accuracy": 0.95, "dcd_domain_accuracy": 0.90,
                    "dcd_collection_accuracy": 0.88, "avg_latency_s": 1.5, "max_latency_s": 3.0}
        canary = {"answer_accuracy": 0.88, "answer_accuracy_incl_partial": 0.92,
                  "answer_correct": 18, "answer_partial": 2, "answer_incorrect": 0,
                  "source_routing_accuracy": 0.96, "dcd_domain_accuracy": 0.91,
                  "dcd_collection_accuracy": 0.90, "avg_latency_s": 1.4, "max_latency_s": 2.8}
        report = render_report(baseline, canary, changes={"test.py": {"type": "modified"}})
        assert "CANARY DEPLOY" in report
        assert "PASS" in report or "REGRESSION" in report

    def test_render_report_regression(self):
        """Падение accuracy >5% → REGRESSION."""
        from canary_deploy import render_report
        baseline = {"answer_accuracy": 0.90, "answer_accuracy_incl_partial": 0.95,
                    "answer_correct": 18, "answer_partial": 1, "answer_incorrect": 0,
                    "source_routing_accuracy": 0.95, "dcd_domain_accuracy": 0.90,
                    "dcd_collection_accuracy": 0.88, "avg_latency_s": 1.0, "max_latency_s": 2.0}
        canary = {"answer_accuracy": 0.80, "answer_accuracy_incl_partial": 0.85,
                  "answer_correct": 16, "answer_partial": 1, "answer_incorrect": 3,
                  "source_routing_accuracy": 0.90, "dcd_domain_accuracy": 0.80,
                  "dcd_collection_accuracy": 0.78, "avg_latency_s": 1.5, "max_latency_s": 3.0}
        report = render_report(baseline, canary, changes={})
        assert "REGRESSION" in report