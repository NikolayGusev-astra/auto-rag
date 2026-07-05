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
    def test_check_lm_studio(self, mock_get):
        from llm_service import LLMService
        mock_get.return_value = MagicMock(status_code=200)
        svc = LLMService()
        assert svc._check_lm_studio() is True

    @patch("llm_service.requests.post")
    def test_chat(self, mock_post):
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
    def test_chat_json_parse(self, mock_post):
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
