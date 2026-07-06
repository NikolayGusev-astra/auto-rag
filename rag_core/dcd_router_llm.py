"""
LLM-based DCD Router — Qwen 2.5 7B + JSON Schema.

Alternative to keyword dcd_router.py. Better on non-trivial queries,
but requires LM Studio and adds ~0.5s latency.

Config:
    RAG_DCD_MODE=keyword|llm|hybrid (default: keyword)

Hybrid mode:
    keyword first -> if confidence < 0.5 -> LLM override
"""
from __future__ import annotations

import json
import os
import sys
import logging

import requests as sync_requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_config import LM_STUDIO_CHAT_URL, LLM_CLASSIFY_MODEL, LLM_CLASSIFY_TIMEOUT

logger = logging.getLogger(__name__)

# ── Domain/Intent enums ──
DOMAINS = [
    "linux-admin", "networking", "devops", "software-dev", "database",
    "monitoring", "security", "storage", "virtualization", "email",
    "kernel", "scripting", "docs", "hardware", "general", "unknown",
]

INTENTS = ["list", "search", "status", "create", "unknown"]


def classify_llm(query: str) -> dict:
    """Classify query using LLM with JSON Schema structured output.

    Returns: {domain, collection, confidence, intent, fallback}
    Fallback: {domain: "unknown", confidence: 0.0} on error.
    """
    prompt = f"""Классифицируй запрос пользователя по домену и намерению.

Домены: {", ".join(DOMAINS)}
Намерения: {", ".join(INTENTS)}

Правила:
- Если запрос не относится к определённому домену — "general"
- Если insufficient info — "unknown"
- Верни только JSON

Запрос: {query}"""

    try:
        resp = sync_requests.post(
            LM_STUDIO_CHAT_URL,
            json={
                "model": LLM_CLASSIFY_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "dcd_router_llm",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "domain": {
                                    "type": "string",
                                    "enum": DOMAINS,
                                },
                                "intent": {
                                    "type": "string",
                                    "enum": INTENTS,
                                },
                                "confidence": {
                                    "type": "number",
                                },
                                "reason": {
                                    "type": "string",
                                },
                            },
                            "required": ["domain", "intent", "confidence", "reason"],
                            "additionalProperties": False,
                        },
                    },
                },
                "temperature": 0.0,
                "max_tokens": 150,
            },
            timeout=LLM_CLASSIFY_TIMEOUT,
        )
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        result = json.loads(text)
        result["fallback"] = result.get("confidence", 0) < 0.3
        return result

    except Exception as e:
        logger.warning("LLM DCD failed: %s", e)
        return {
            "domain": "unknown",
            "collection": "general",
            "confidence": 0.0,
            "intent": "unknown",
            "reason": f"llm_error: {e}",
            "fallback": True,
        }


def classify_hybrid(query: str) -> dict:
    """Hybrid classification: keyword first, LLM override if low confidence.

    Keyword router is fast and zero-cost. LLM kicks in only when
    keyword confidence is below threshold.
    """
    from dcd_router import classify as classify_keyword

    kw_result = classify_keyword(query)

    # Low confidence -> ask LLM
    if kw_result.get("confidence", 0) < 0.3:
        llm_result = classify_llm(query)

        # If LLM also unsure, keep keyword result
        if llm_result.get("confidence", 0) < 0.3:
            kw_result["router"] = "keyword"
            return kw_result

        llm_result["router"] = "llm_override"
        llm_result["keyword_domain"] = kw_result.get("domain")
        llm_result["keyword_confidence"] = kw_result.get("confidence")
        if llm_result["domain"] in _DOMAIN_COLLECTION_MAP:
            llm_result["collection"] = _DOMAIN_COLLECTION_MAP[llm_result["domain"]][0]
        else:
            llm_result["collection"] = llm_result["domain"]
        return llm_result

    kw_result["router"] = "keyword"
    return kw_result


# Collection mapping (same as dcd_router.py DOMAIN_KEYWORDS collections)
_DOMAIN_COLLECTION_MAP = {
    "linux-admin": ["linux-admin", "server-config"],
    "networking": ["networking", "vpn-config"],
    "devops": ["devops", "infra-as-code"],
    "software-dev": ["software-dev", "code-patterns"],
    "database": ["database", "postgresql"],
    "monitoring": ["monitoring", "observability"],
    "security": ["security", "hardening"],
    "storage": ["storage", "filesystems"],
    "virtualization": ["virtualization", "proxmox"],
    "email": ["email", "mail-server"],
    "kernel": ["kernel", "linux-internals"],
    "scripting": ["scripting", "automation"],
    "docs": ["docs", "architecture"],
    "hardware": ["hardware", "server-hardware"],
    "general": ["general"],
    "unknown": ["general"],
}


def classify(query: str) -> dict:
    """Main classify() — dispatches based on RAG_DCD_MODE config.

    Modes:
        keyword (default): fast keyword matching only
        llm: LLM classification only
        hybrid: keyword first, LLM override if low confidence
    """
    mode = os.getenv("RAG_DCD_MODE", "keyword").lower()

    if mode == "llm":
        result = classify_llm(query)
        result["router"] = "llm"
        if result.get("domain") in _DOMAIN_COLLECTION_MAP:
            result["collection"] = _DOMAIN_COLLECTION_MAP[result["domain"]][0]
        return result
    elif mode == "hybrid":
        return classify_hybrid(query)
    else:
        from dcd_router import classify as classify_keyword
        result = classify_keyword(query)
        result["router"] = "keyword"
        return result


if __name__ == "__main__":
    test_queries = [
        "как настроить postgresql replication",
        "найди договор номер 123",
        "wireguard vpn через nat",
        "что такое ownership в rust",
    ]
    for q in test_queries:
        for mode in ["keyword", "llm", "hybrid"]:
            os.environ["RAG_DCD_MODE"] = mode
            result = classify(q)
            print(f"  [{mode:8s}] {q:40s} -> {result['domain']:15s} (c={result.get('confidence', 0):.2f}) router={result.get('router', '?')}")
        print()
