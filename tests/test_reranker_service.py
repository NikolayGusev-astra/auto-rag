import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_core"))


class TestRerankerService:
    def test_singleton(self):
        from reranker_service import RerankerService
        a = RerankerService.get()
        b = RerankerService.get()
        assert a is b

    @patch("reranker_service.requests.options")
    def test_cohere_format_detection(self, mock_options):
        from reranker_service import RerankerService
        mock_options.return_value = MagicMock(status_code=200)
        svc = RerankerService.get()
        svc._api_format = None
        assert svc._detect_api_format() == "cohere"

    @patch("reranker_service.requests.post")
    def test_rerank_jina(self, mock_post):
        from reranker_service import RerankerService
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"score": 0.9}, {"score": 0.1}]},
        )
        svc = RerankerService.get()
        scored = svc.rerank("query", ["a", "b"], top_k=2)
        assert len(scored) == 2
        assert scored[0][1] >= scored[1][1]

    @patch("reranker_service.requests.post")
    def test_rerank_chunks(self, mock_post):
        from reranker_service import RerankerService
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"score": 0.8}, {"score": 0.3}]},
        )
        svc = RerankerService.get()
        chunks = [{"text": "a"}, {"text": "b"}]
        ranked = svc.rerank_chunks("q", chunks, top_k=1)
        assert len(ranked) == 1
        assert "rerank_score" in ranked[0]
