"""Allowlisted public web retrieval for authoritative domains.

Only runs when the query is explicitly about public documentation
(not internal ticket IDs like SIRIUS-* / INT-* / PROJECT-*).

Uses SearXNG with domain-scoped queries.  Evidence is marked
PUBLIC_WEB with authoritative=true.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

import httpx

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

        results: list[Evidence] = []
        for item in data.get("results", []):
            url = str(item.get("url", ""))
            if not _is_allowlisted_url(url):
                continue
            title = str(item.get("title", ""))
            results.append(Evidence(
                id=f"aldpro_pub:{url}",
                document_id=url,
                title=title,
                text=str(item.get("content", "")),
                source=self.source,
                uri=url,
                origin=EvidenceOrigin.PUBLIC_WEB,
                metadata={
                    "url": url,
                    "authoritative": True,
                    "domain": _extract_domain(url),
                },
            ))
            if len(results) == request.topk:
                break
        return results

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
