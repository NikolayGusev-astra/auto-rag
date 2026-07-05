import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_core"))


def test_embedding_service():
    from embedding_service import EmbeddingService, get_embedding, get_embeddings_batch
    a = EmbeddingService.get()
    b = EmbeddingService.get()
    assert a is b

    with patch("embedding_service.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        svc = EmbeddingService.get()
        assert svc.check_lm_studio(force=True) is True

    with patch("embedding_service.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"embedding": [0.1] * 10}]},
        )
        svc = EmbeddingService.get()
        result = svc.embed_batch(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 10


def test_reranker_service():
    from reranker_service import RerankerService, rerank_chunks
    a = RerankerService.get()
    b = RerankerService.get()
    assert a is b

    with patch("reranker_service.requests.options") as mock_options:
        mock_options.return_value = MagicMock(status_code=200)
        svc = RerankerService.get()
        svc._api_format = None
        assert svc._detect_api_format() == "cohere"

    with patch("reranker_service.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"score": 0.9}, {"score": 0.1}]},
        )
        svc = RerankerService.get()
        scored = svc.rerank("query", ["a", "b"], top_k=2)
        assert len(scored) == 2
        assert scored[0][1] >= scored[1][1]


def test_llm_service():
    from llm_service import LLMService, get_llm
    a = get_llm()
    b = get_llm()
    assert a is b

    with patch("llm_service.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        svc = get_llm()
        assert svc._check_lm_studio() is True

    with patch("llm_service.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": "hello"}}],
                "model": "qwen2.5-3b",
                "usage": {},
            },
        )
        llm = get_llm()
        llm._lm_studio_available = True
        result = llm.chat([{"role": "user", "content": "hi"}])
        assert result["content"] == "hello"


def test_lm_studio_monitor():
    from lm_studio_monitor import LMStudioMonitor, get_lm_studio
    a = LMStudioMonitor.get()
    b = LMStudioMonitor.get()
    assert a is b

    with patch("lm_studio_monitor.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"id": "text-embedding-bge-m3"}]},
        )
        monitor = get_lm_studio()
        status = monitor.get_status(force=True)
        assert status["available"] is True
        assert "text-embedding-bge-m3" in status["loaded_models"]

        assert monitor.is_model_loaded("text-embedding-bge-m3") is True
        assert monitor.is_model_loaded("missing") is False


if __name__ == "__main__":
    test_embedding_service()
    print("PASS: test_embedding_service")
    test_reranker_service()
    print("PASS: test_reranker_service")
    test_llm_service()
    print("PASS: test_llm_service")
    test_lm_studio_monitor()
    print("PASS: test_lm_studio_monitor")
    print("All tests passed")
