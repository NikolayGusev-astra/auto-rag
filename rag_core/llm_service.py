"""Унифицированный LLM service для local inference через LM Studio.

Приоритет:
1. LM Studio (если запущен) — GPU inference
2. Fallback на keyword-only режим (без LLM)
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import requests

LM_STUDIO_URL = os.getenv("RAG_LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")
LLM_TIMEOUT = int(os.getenv("RAG_LLM_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.getenv("RAG_LLM_RETRIES", "2"))


class LLMService:
    """Singleton LLM client."""

    def __init__(self):
        self._lm_studio_available: Optional[bool] = None
        self._last_check = 0.0

    def _check_lm_studio(self) -> bool:
        """Проверить доступность LM Studio (кешируем на 30s)."""
        if self._lm_studio_available is not None and time.time() - self._last_check < 30:
            return self._lm_studio_available
        try:
            r = requests.get(
                LM_STUDIO_URL.replace("/chat/completions", "/models"),
                timeout=2,
            )
            self._lm_studio_available = r.status_code == 200
        except Exception:
            self._lm_studio_available = False
        self._last_check = time.time()
        return self._lm_studio_available

    def chat(
        self,
        messages: list[dict],
        model: str = "qwen2.5-3b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 500,
        response_format: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> dict:
        """Chat completion с retry.

        Returns: {"content": str, "model": str, "usage": dict}
        Raises: RuntimeError если LM Studio недоступен после retries.
        """
        if not self._check_lm_studio():
            raise RuntimeError("LM Studio not available")

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        last_error = None
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                r = requests.post(
                    LM_STUDIO_URL,
                    json=payload,
                    timeout=timeout or LLM_TIMEOUT,
                )
                r.raise_for_status()
                data = r.json()
                return {
                    "content": data["choices"][0]["message"]["content"].strip(),
                    "model": data.get("model", model),
                    "usage": data.get("usage", {}),
                }
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"LLM failed after {LLM_MAX_RETRIES} retries: {last_error}")
            except Exception as e:
                raise RuntimeError(f"LLM call failed: {e}")

        raise RuntimeError(f"LLM failed after {LLM_MAX_RETRIES} retries: {last_error}")

    def chat_json(
        self,
        messages: list[dict],
        model: str = "qwen2.5-3b-instruct",
        schema: Optional[dict] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> dict:
        """Chat completion с structured JSON output."""
        response_format = None
        if schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": schema,
                },
            }

        result = self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        text = result["content"]
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            result["json"] = json.loads(text)
        except json.JSONDecodeError as e:
            result["json"] = None
            result["parse_error"] = str(e)

        return result


_llm_service: Optional[LLMService] = None


def get_llm() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service