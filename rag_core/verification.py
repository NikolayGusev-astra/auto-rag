"""LLM relevance verification with fail-closed semantics.

P1 fix (audit): the old `_llm_verify` returned `0.5` on ANY exception
(timeout, bad JSON, model down, API change). That turned an operational
failure into a neutral relevance signal — dangerous for a technical/security
RAG where an unavailable verifier must NOT look like "half relevant".

We now distinguish four outcomes and never collapse them into a single float.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    RELEVANT = "relevant"               # model said relevant (>= pass threshold)
    IRRELEVANT = "irrelevant"           # model said not relevant
    UNAVAILABLE = "unavailable"         # model/network error — NO relevance signal
    INVALID = "invalid"                # model returned unparseable output


@dataclass
class VerificationResult:
    status: VerificationStatus
    score: float | None                 # only meaningful for RELEVANT/IRRELEVANT
    detail: str = ""


_PASS_THRESHOLD = 0.3

# module-level cache (per-process). Keyed by query+chunk texts.
_CACHE: dict[str, tuple[float, float, VerificationStatus, float | None]] = {}
_CACHE_MAX = 256
_CACHE_TTL = 120.0


def _cache_key(query: str, chunks: list[dict]) -> str:
    texts = "||".join((c.get("text") or "")[:180] for c in chunks[:3])
    return hashlib.md5(f"{query}||{texts}".encode("utf-8")).hexdigest()


def verify_relevance(
    query: str,
    chunks: list[dict],
    *,
    enabled: bool,
    url: str,
    model: str,
    timeout: float,
) -> VerificationResult:
    """Soft verification: is the retrieved content relevant to the query?

    Returns UNAVAILABLE (never a relevance score) when the verifier itself
    fails. Callers must treat UNAVAILABLE as "unknown", not "half relevant".
    """
    if not chunks or not enabled:
        return VerificationResult(VerificationStatus.IRRELEVANT, 0.0,
                                 "disabled or empty")

    key = _cache_key(query, chunks)
    now = time.time()
    cached = _CACHE.get(key)
    if cached:
        ts, _, status, score = cached
        if now - ts < _CACHE_TTL:
            return VerificationResult(status, score, "cache")

    top = "\n\n".join(
        [f"[{i}] {c['text'][:500].replace(chr(10), ' ')}"
         for i, c in enumerate(chunks[:3])]
    )
    prompt = (
        f"Rate relevance 0.0-1.0. Reply ONLY a number.\n"
        f"Query: {query[:200]}\nDocuments:\n{top}"
    )
    try:
        resp = requests.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 10,
            },
            timeout=timeout,
        )
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        nums = re.findall(r"0\.\d+|1\.0", answer)
        if not nums:
            result = VerificationResult(VerificationStatus.INVALID, None,
                                        "no number in model output")
        else:
            score = float(nums[0])
            status = (VerificationStatus.RELEVANT
                      if score >= _PASS_THRESHOLD
                      else VerificationStatus.IRRELEVANT)
            result = VerificationResult(status, score, "model")
    except Exception as exc:
        # Fail-closed: operational failure is NOT a relevance signal.
        logger.warning("verifier unavailable: %s", exc)
        result = VerificationResult(VerificationStatus.UNAVAILABLE, None,
                                    f"{type(exc).__name__}: {exc}")

    _CACHE[key] = (now, 0.0, result.status, result.score)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)), None)
    return result