"""LM Studio monitor — health, loaded models, VRAM usage.

LM Studio управляет моделями сам. Этот модуль только наблюдает.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

import requests

LM_STUDIO_BASE = os.getenv("RAG_LM_STUDIO_URL", "http://localhost:1234").rstrip("/")
LM_STUDIO_TIMEOUT = int(os.getenv("RAG_LM_STUDIO_TIMEOUT", "5"))


class LMStudioMonitor:
    """Singleton LM Studio monitor."""
    _instance: Optional[LMStudioMonitor] = None
    _lock = threading.Lock()

    def __init__(self):
        self._cache: Optional[dict] = None
        self._cache_ts = 0.0
        self._cache_ttl = 10

    @classmethod
    def get(cls) -> LMStudioMonitor:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_status(self, force: bool = False) -> dict:
        """Get LM Studio status: available, loaded models, VRAM."""
        if not force and self._cache and time.time() - self._cache_ts < self._cache_ttl:
            return self._cache

        result = {
            "available": False,
            "base_url": LM_STUDIO_BASE,
            "loaded_models": [],
            "vram": None,
            "error": None,
        }

        try:
            r = requests.get(
                f"{LM_STUDIO_BASE}/v1/models",
                timeout=LM_STUDIO_TIMEOUT,
            )
            if r.status_code == 200:
                result["available"] = True
                data = r.json()
                result["loaded_models"] = [m.get("id", "") for m in data.get("data", [])]
            else:
                result["error"] = f"HTTP {r.status_code}"
        except requests.ConnectionError:
            result["error"] = "Connection refused — LM Studio not running?"
        except Exception as e:
            result["error"] = str(e)[:200]

        try:
            r = requests.get(
                f"{LM_STUDIO_BASE}/v1/hardware",
                timeout=LM_STUDIO_TIMEOUT,
            )
            if r.status_code == 200:
                result["vram"] = r.json()
        except Exception:
            pass

        self._cache = result
        self._cache_ts = time.time()
        return result

    def is_model_loaded(self, model_name: str) -> bool:
        status = self.get_status()
        if not status["available"]:
            return False
        return model_name in status["loaded_models"]

    def wait_for_model(self, model_name: str, timeout: int = 60) -> bool:
        """Ждать пока модель загрузится в LM Studio."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.is_model_loaded(model_name):
                return True
            try:
                requests.post(
                    f"{LM_STUDIO_BASE}/v1/embeddings",
                    json={"model": model_name, "input": ["warmup"]},
                    timeout=10,
                )
            except Exception:
                pass
            time.sleep(2)
        return False

    def warmup_all(self) -> dict:
        """Pre-load все модели, которые нужны auto-rag."""
        from rag_core.rag_config import (
            EMBEDDING_MODEL,
            RERANK_MODEL,
            LLM_CLASSIFY_MODEL,
            LLM_VERIFY_MODEL,
            LLM_EVAL_MODEL,
        )

        models_to_warmup = [
            ("embedding", EMBEDDING_MODEL),
            ("reranker", RERANK_MODEL),
            ("llm_classify", LLM_CLASSIFY_MODEL),
            ("llm_verify", LLM_VERIFY_MODEL),
        ]

        results = {}
        for role, model in models_to_warmup:
            t0 = time.time()
            ok = self.wait_for_model(model, timeout=30)
            results[role] = {
                "model": model,
                "loaded": ok,
                "warmup_s": round(time.time() - t0, 1),
            }
        return results

    @property
    def is_available(self) -> bool:
        return self.get_status()["available"]


def get_lm_studio() -> LMStudioMonitor:
    return LMStudioMonitor.get()