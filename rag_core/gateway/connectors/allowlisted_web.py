"""Allowlisted public web retrieval for authoritative domains.

Only runs when the query is explicitly about public documentation
(not internal ticket IDs like SIRIUS-* / INT-* / PROJECT-*).

Uses SearXNG with domain-scoped queries.  Evidence is marked
PUBLIC_WEB with authoritative=true.

Top-N results are enriched with Trafilatura full-text extraction
(with snippet fallback).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urlsplit

import httpx
import trafilatura

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.models import Evidence, EvidenceOrigin, SyncBatch

# ── Authoritative domains (public documentation only) ─────────────
_AUTHORITATIVE_DOMAINS = (
    "aldpro.ru",
    "astralinux.ru",
    "docs.astra-automation.ru",
    "wiki.astralinux.ru",
)

_INTERNAL_PATTERN = re.compile(
    r"\b(SIRIUS|BT|AD|PROJECT|PRESALE|INT|AKNO|NOVA)[- ]\d+\b", re.IGNORECASE
)

# ── Positive intent: only these query types trigger allowlisted web ─
_PUBLIC_DOC_INTENT = re.compile(
    r"\b("
    r"матриц[аы]\s*совместимост[ией]|"
    r"официальн[аяо][йе]\s*документаци[ия]|"
    r"поряд[окк][а]?\s*обновлени[яй]|"
    r"поддерживаем[аяо][яе]\s*верси[яи]|"
    r"release\s*notes|"
    r"инструкци[яи]\s*по\s*(установк[еи]|обновлени[ю]|настройк[еи]|миграци[и])|"
    r"руководств[оа]\s*(администратор[ау]|пользовател[яю]|по\s*эксплуатаци[и])|"
    r"системн[ыей]\s*требовани[яй]|"
    r"известн[ыей]\s*проблем[ыа]|"
    r"список\s*изменени[йя]|"
    r"changelog|"
    r"лицензионн[оа][егой]?|"
    r"сертифика[тц]\s*(соответстви[яй]|ФСТЭК)"
    r")\b",
    re.IGNORECASE,
)

# ── Full-text extraction tunables ─────────────────────────────────
_MAX_FULLTEXT_URLS = 3
_FULLTEXT_TIMEOUT = 10.0  # seconds per URL
_MAX_FULLTEXT_CHARS = 24000


def is_public_doc_query(query: str) -> bool:
    """Return True when the query triggers allowlisted public retrieval.

    Requires BOTH:
    1. Positive intent match (documentation/release/compatibility topic)
    2. Not an internal ticket ID
    """
    if _looks_like_internal(query):
        return False
    return bool(_PUBLIC_DOC_INTENT.search(query))


def _looks_like_internal(query: str) -> bool:
    return bool(_INTERNAL_PATTERN.search(query))


def _build_domain_query(query: str) -> str:
    sites = "|".join(_AUTHORITATIVE_DOMAINS)
    return f"({query}) site:{sites}"


async def _extract_full_text(url: str, timeout: float = _FULLTEXT_TIMEOUT) -> str:
    """Fetch URL and extract main text via Trafilatura. Returns '' on failure."""
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.get(url, headers={"User-Agent": "auto-rag/1.0"})
            resp.raise_for_status()
            html = resp.text
    except Exception:
        return ""

    try:
        text = await asyncio.to_thread(
            trafilatura.extract, html,
            include_comments=False,
            include_tables=True,
            include_links=False,
            include_images=False,
            output_format="txt",
            target_language="ru",
        )
    except Exception:
        return ""

    return (text or "").strip()


class AllowlistedWebConnector:
    retrieval_kind = "web"
    source = "aldpro_public"

    def __init__(self, searxng_url: str = "http://localhost:8888") -> None:
        self._base = searxng_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=20, trust_env=False)
        return self._client

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        if not is_public_doc_query(request.query):
            return []

        domain_query = _build_domain_query(request.query)
        try:
            resp = await self._http.get(
                f"{self._base}/search",
                params={"q": domain_query, "format": "json", "categories": "general"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        # ── Build evidence from snippets first ──────────────────────
        results: list[Evidence] = []
        urls_for_fulltext: list[tuple[int, str]] = []
        for idx, item in enumerate(data.get("results", [])):
            url = str(item.get("url", ""))
            if not _is_allowlisted_url(url):
                continue
            title = str(item.get("title", ""))
            snippet = str(item.get("content", ""))
            results.append(Evidence(
                id=f"aldpro_pub:{url}",
                document_id=url,
                title=title,
                text=snippet,
                source=self.source,
                uri=url,
                origin=EvidenceOrigin.PUBLIC_WEB,
                metadata={
                    "url": url,
                    "authoritative": True,
                    "domain": _extract_domain(url),
                    "extraction_method": "snippet_fallback",
                },
            ))
            if idx < _MAX_FULLTEXT_URLS:
                urls_for_fulltext.append((len(results) - 1, url))

        # ── Enrich top-N with full text (parallel, non-blocking) ────
        if urls_for_fulltext:
            extractions = await asyncio.gather(
                *[_extract_full_text(url) for _, url in urls_for_fulltext],
                return_exceptions=True,
            )
            for (result_idx, url), full_text in zip(urls_for_fulltext, extractions):
                if isinstance(full_text, str) and full_text:
                    results[result_idx] = Evidence(
                        id=f"aldpro_pub:{url}",
                        document_id=url,
                        title=results[result_idx].title,
                        text=full_text[:_MAX_FULLTEXT_CHARS],
                        source=self.source,
                        uri=url,
                        origin=EvidenceOrigin.PUBLIC_WEB,
                        metadata={
                            "url": url,
                            "authoritative": True,
                            "domain": _extract_domain(url),
                            "extraction_method": "fulltext",
                            "fulltext_chars": len(full_text),
                        },
                    )

        return results[: request.topk]

    async def health(self) -> dict[str, object]:
        try:
            resp = await self._http.get(
                f"{self._base}/search", params={"q": "health", "format": "json"},
            )
            return {"source": self.source, "available": resp.status_code == 200}
        except Exception:
            return {"source": self.source, "available": False}

    async def sync_changes(self, cursor: str | None) -> SyncBatch:
        return SyncBatch(added=[])

    async def fetch(self, ref: object) -> object:
        raise NotImplementedError


def _extract_domain(url: str) -> str:
    return urlsplit(url).hostname or ""


def _is_allowlisted_url(url: str) -> bool:
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    return parsed.scheme in {"http", "https"} and any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in _AUTHORITATIVE_DOMAINS
    )
