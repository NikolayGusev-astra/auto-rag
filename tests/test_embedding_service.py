import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_core"))


class TestEmbeddingService:
    def test_singleton(self):
        from embedding_service import EmbeddingService
        a = EmbeddingService.get()
        b = EmbeddingService.get()
        assert a is b

    @patch("embedding_service.requests.get")
    def test_health_check(self, mock_get):
        from embedding_service import EmbeddingService
        mock_get.return_value = MagicMock(status_code=200)
        svc = EmbeddingService.get()
        assert svc.check_lm_studio(force=True) is True
        mock_get.assert_called_once()

    @patch("embedding_service.requests.post")
    def test_embed_batch(self, mock_post):
        from embedding_service import EmbeddingService
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"embedding": [0.1] * 10}]},
        )
        svc = EmbeddingService.get()
        result = svc.embed_batch(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 10

    def test_cache_roundtrip(self):
        from embedding_service import EmbeddingService
        svc = EmbeddingService.get()
        assert hasattr(svc, "_cache_get")
        assert hasattr(svc, "_cache_put")