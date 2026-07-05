import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_core"))


class TestLMStudioMonitor:
    def test_singleton(self):
        from lm_studio_monitor import LMStudioMonitor
        a = LMStudioMonitor.get()
        b = LMStudioMonitor.get()
        assert a is b

    @patch("lm_studio_monitor.requests.get")
    def test_status_available(self, mock_get):
        from lm_studio_monitor import LMStudioMonitor
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"id": "text-embedding-bge-m3"}]},
        )
        monitor = LMStudioMonitor.get()
        status = monitor.get_status(force=True)
        assert status["available"] is True
        assert "text-embedding-bge-m3" in status["loaded_models"]

    @patch("lm_studio_monitor.requests.get")
    def test_is_model_loaded(self, mock_get):
        from lm_studio_monitor import LMStudioMonitor, get_lm_studio
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"id": "qwen2.5-3b"}]},
        )
        monitor = get_lm_studio()
        assert monitor.is_model_loaded("qwen2.5-3b") is True
        assert monitor.is_model_loaded("missing") is False
