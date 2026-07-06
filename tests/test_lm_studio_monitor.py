"""Tests for LMStudioMonitor.

Все тесты mock-ают HTTP запросы к LM Studio.
"""
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
    def test_get_status_available(self, mock_get):
        """get_status возвращает available=True когда LM Studio отвечает."""
        from lm_studio_monitor import LMStudioMonitor

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [
                    {"id": "bge-m3"},
                    {"id": "qwen2.5-3b-instruct"},
                ]
            },
        )

        monitor = LMStudioMonitor()
        monitor._cache = None  # force fresh
        status = monitor.get_status(force=True)

        assert status["available"] is True
        assert "bge-m3" in status["loaded_models"]
        assert "qwen2.5-3b-instruct" in status["loaded_models"]

    @patch("lm_studio_monitor.requests.get")
    def test_get_status_unavailable(self, mock_get):
        """get_status возвращает available=False когда LM Studio не отвечает."""
        from lm_studio_monitor import LMStudioMonitor

        mock_get.side_effect = Exception("Connection refused")

        monitor = LMStudioMonitor()
        monitor._cache = None
        status = monitor.get_status(force=True)

        assert status["available"] is False
        assert "Connection refused" in status.get("error", "") or status.get("error") is not None

    @patch.object(LMStudioMonitor, 'get_status')
    def test_is_model_loaded_true(self, mock_status):
        """is_model_loaded возвращает True если модель в loaded_models."""
        from lm_studio_monitor import LMStudioMonitor

        mock_status.return_value = {
            "available": True,
            "loaded_models": ["bge-m3", "qwen2.5-3b-instruct"],
        }

        monitor = LMStudioMonitor()
        assert monitor.is_model_loaded("bge-m3") is True
        assert monitor.is_model_loaded("qwen2.5-3b-instruct") is True

    @patch.object(LMStudioMonitor, 'get_status')
    def test_is_model_loaded_false(self, mock_status):
        """is_model_loaded возвращает False если модели нет."""
        from lm_studio_monitor import LMStudioMonitor

        mock_status.return_value = {
            "available": True,
            "loaded_models": ["bge-m3"],
        }

        monitor = LMStudioMonitor()
        assert monitor.is_model_loaded("qwen2.5-7b") is False

    @patch.object(LMStudioMonitor, 'get_status')
    def test_is_model_loaded_unavailable(self, mock_status):
        """is_model_loaded возвращает False когда LM Studio недоступен."""
        from lm_studio_monitor import LMStudioMonitor

        mock_status.return_value = {
            "available": False,
            "loaded_models": [],
        }

        monitor = LMStudioMonitor()
        assert monitor.is_model_loaded("bge-m3") is False

    @patch.object(LMStudioMonitor, 'get_status')
    def test_is_available_property(self, mock_status):
        """is_available property отражает статус."""
        from lm_studio_monitor import LMStudioMonitor

        mock_status.return_value = {"available": True, "loaded_models": []}
        monitor = LMStudioMonitor()
        assert monitor.is_available is True

        mock_status.return_value = {"available": False, "loaded_models": []}
        assert monitor.is_available is False
