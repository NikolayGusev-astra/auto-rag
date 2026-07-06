"""Tests for LLMService.

Все тесты mock-ают LM Studio HTTP endpoints, не требуют реального LM Studio.
"""
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_core"))


class TestLLMService:
    def test_singleton(self):
        from llm_service import get_llm
        a = get_llm()
        b = get_llm()
        assert a is b

    @patch("llm_service.requests.get")
    def test_check_lm_studio_available(self, mock_get):
        """_check_lm_studio возвращает True когда LM Studio отвечает 200."""
        from llm_service import LLMService
        mock_get.return_value = MagicMock(status_code=200)
        svc = LLMService()
        svc._lm_studio_available = None
        svc._last_check = 0
        assert svc._check_lm_studio() is True

    @patch("llm_service.requests.get")
    def test_check_lm_studio_unavailable(self, mock_get):
        """_check_lm_studio возвращает False когда LM Studio не отвечает."""
        from llm_service import LLMService
        mock_get.side_effect = Exception("Connection refused")
        svc = LLMService()
        svc._lm_studio_available = None
        svc._last_check = 0
        assert svc._check_lm_studio() is False

    @patch("llm_service.requests.post")
    @patch("llm_service.LLMService._check_lm_studio", return_value=True)
    def test_chat(self, mock_check, mock_post):
        """chat() возвращает content из LM Studio response."""
        from llm_service import get_llm
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": "hello"}}],
                "model": "qwen2.5-3b",
                "usage": {},
            },
        )
        llm = get_llm()
        result = llm.chat([{"role": "user", "content": "hi"}])
        assert result["content"] == "hello"

    @patch("llm_service.requests.post")
    @patch("llm_service.LLMService._check_lm_studio", return_value=True)
    def test_chat_json_parse(self, mock_check, mock_post):
        """chat_json() парсит JSON из ```json block."""
        from llm_service import get_llm
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '```json\n{"ok": true}\n```'}}],
            },
        )
        llm = get_llm()
        result = llm.chat_json([{"role": "user", "content": "return json"}])
        assert result["json"] == {"ok": True}

    @patch("llm_service.requests.post")
    @patch("llm_service.LLMService._check_lm_studio", return_value=True)
    def test_chat_json_plain(self, mock_check, mock_post):
        """chat_json() парсит plain JSON (без markdown wrapper)."""
        from llm_service import get_llm
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '{"domain": "devops", "confidence": 0.9}'}}],
            },
        )
        llm = get_llm()
        result = llm.chat_json([{"role": "user", "content": "classify"}])
        assert result["json"]["domain"] == "devops"
        assert result["json"]["confidence"] == 0.9

    @patch("llm_service.LLMService._check_lm_studio", return_value=False)
    def test_chat_raises_when_unavailable(self, mock_check):
        """chat() raises RuntimeError когда LM Studio недоступен."""
        from llm_service import get_llm
        llm = get_llm()
        with pytest.raises(RuntimeError, match="LM Studio not available"):
            llm.chat([{"role": "user", "content": "hi"}])

    @patch("llm_service.requests.post")
    @patch("llm_service.LLMService._check_lm_studio", return_value=True)
    def test_chat_retry_on_timeout(self, mock_check, mock_post):
        """chat() retry на timeout."""
        from llm_service import get_llm
        import requests as req

        mock_post.side_effect = [
            req.Timeout("timeout"),
            MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "recovered"}}]},
            ),
        ]
        llm = get_llm()
        import llm_service
        llm_service.LLM_MAX_RETRIES = 2
        result = llm.chat([{"role": "user", "content": "hi"}])
        assert result["content"] == "recovered"
        assert mock_post.call_count == 2
