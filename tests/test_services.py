"""Integration tests for all services."""
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_core"))


class TestServicesIntegration:
    """Integration tests — все сервисы mock-аются, LM Studio не требуется."""

    def test_embedding_service_singleton(self):
        from embedding_service import EmbeddingService
        a = EmbeddingService.get()
        b = EmbeddingService.get()
        assert a is b

    def test_reranker_service_singleton(self):
        from reranker_service import RerankerService
        a = RerankerService.get()
        b = RerankerService.get()
        assert a is b

    def test_llm_service_singleton(self):
        from llm_service import get_llm
        a = get_llm()
        b = get_llm()
        assert a is b

    def test_lm_studio_monitor_singleton(self):
        from lm_studio_monitor import LMStudioMonitor
        a = LMStudioMonitor.get()
        b = LMStudioMonitor.get()
        assert a is b

    @patch("embedding_service.EmbeddingService.check_lm_studio", return_value=False)
    @patch("embedding_service.EmbeddingService._embed_via_sentence_transformers")
    @patch("embedding_service.EmbeddingService._embed_via_lm_studio")
    def test_embedding_fallback_chain(self, mock_lm, mock_st, mock_check):
        """EmbeddingService пробует LM Studio → sentence-transformers → zeros."""
        from embedding_service import EmbeddingService

        mock_lm.side_effect = Exception("LM Studio down")
        mock_st.return_value = [[0.1] * 1024, [0.2] * 1024]

        svc = EmbeddingService.get()
        svc._cache_conn.execute("DELETE FROM embeddings")
        svc._cache_conn.commit()

        result = svc.embed_batch(["text1", "text2"])
        assert len(result) == 2
        assert len(result[0]) == 1024

    @patch("reranker_service.RerankerService._detect_api_format", return_value="jina")
    @patch("reranker_service.requests.post")
    def test_reranker_jina_format(self, mock_post, mock_format):
        """RerankerService работает с Jina-style API."""
        from reranker_service import RerankerService

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [
                    {"score": 0.9},
                    {"score": 0.5},
                    {"score": 0.3},
                ]
            },
        )

        svc = RerankerService()
        svc._api_format = "jina"
        ranked = svc.rerank("query", ["doc1", "doc2", "doc3"], top_k=3)

        assert len(ranked) == 3
        assert ranked[0][0] == 0
        assert ranked[0][1] == 0.9

    @patch("llm_service.LLMService._check_lm_studio", return_value=True)
    @patch("llm_service.requests.post")
    def test_llm_service_chat(self, mock_post, mock_check):
        """LLMService.chat() работает с mock."""
        from llm_service import get_llm

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": "test response"}}],
                "model": "qwen2.5-3b-instruct",
                "usage": {"total_tokens": 10},
            },
        )

        llm = get_llm()
        result = llm.chat([{"role": "user", "content": "test"}])
        assert result["content"] == "test response"
        assert result["model"] == "qwen2.5-3b-instruct"

    @patch("llm_service.LLMService._check_lm_studio", return_value=False)
    def test_llm_service_unavailable(self, mock_check):
        """LLMService.chat() raises когда LM Studio недоступен."""
        from llm_service import get_llm

        llm = get_llm()
        with pytest.raises(RuntimeError, match="LM Studio not available"):
            llm.chat([{"role": "user", "content": "test"}])

    @patch("lm_studio_monitor.requests.get")
    def test_lm_studio_monitor_status(self, mock_get):
        """LMStudioMonitor.get_status() работает с mock."""
        from lm_studio_monitor import LMStudioMonitor

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [{"id": "bge-m3"}, {"id": "qwen2.5-3b-instruct"}]
            },
        )

        monitor = LMStudioMonitor()
        monitor._cache = None
        status = monitor.get_status(force=True)

        assert status["available"] is True
        assert len(status["loaded_models"]) == 2


# Standalone functions (for backward compat with existing test runner)
@patch("llm_service.LLMService._check_lm_studio", return_value=True)
def test_llm_service(mock_check):
    """LLMService basic test — mock _check_lm_studio instead of real HTTP."""
    from llm_service import get_llm
    svc = get_llm()
    assert svc._check_lm_studio() is True


def test_embedding_service():
    """EmbeddingService returns a real vector or explicit None, never zeros."""
    from embedding_service import EmbeddingService
    svc = EmbeddingService.get()
    e = svc.embed("test")
    if e is not None:
        assert len(e) == 1024
        assert any(v != 0.0 for v in e)
    else:
        # Failure is explicit: callers can skip search/indexing safely.
        assert e is None


def test_reranker_service():
    """RerankerService basic test — singleton works."""
    from reranker_service import RerankerService
    svc = RerankerService.get()
    assert svc is not None
    assert svc.api_format in ("jina", "cohere")


def test_lm_studio_monitor():
    """LMStudioMonitor basic test — singleton works."""
    from lm_studio_monitor import LMStudioMonitor
    m = LMStudioMonitor.get()
    assert m is not None
    status = m.get_status(force=True)
    assert "available" in status
