"""Live Confluence retrieval — with PDF attachment text extraction."""

from __future__ import annotations

from html.parser import HTMLParser
import io
import re
from typing import Any

import httpx

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.models import Evidence, EvidenceOrigin, SyncBatch


class ConfluenceConnector:
    retrieval_kind = "live"

    def __init__(self, base_url: str, token: str, source: str = "confluence") -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self.source = source
        self._client: httpx.AsyncClient | None = None

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=30.0,
                trust_env=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── search_live ────────────────────────────────────────────────

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        cql_queries: list[str] = []
        page_id = _extract_page_id(request.query)
        if page_id:
            cql_queries.append(f"id={page_id}")
        elif _looks_like_title(request.query):
            cql_queries.append(f'title~"{_escape_query(request.query)}"')
        cql_queries.append(f'text~"{_escape_query(request.query)}"')

        pages: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for cql in cql_queries:
            payload = await self._get(
                "/rest/api/content/search",
                params={"cql": cql, "limit": request.topk, "expand": "body.storage"},
            )
            for page in payload.get("results", []):
                pid = str(page["id"])
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    pages.append(page)

        evidence_list: list[Evidence] = []
        for page in pages[: request.topk]:
            ev = await _evidence(page, self._base, self.source, self._http)
            evidence_list.append(ev)
        return evidence_list

    # ── attachments (public for sync/testing) ──────────────────────

    async def list_attachments(self, page_id: str) -> list[dict[str, Any]]:
        payload = await self._get(
            f"/rest/api/content/{page_id}/child/attachment",
            params={"limit": 50, "expand": "version"},
        )
        return list(payload.get("results", []))

    async def download_attachment_bytes(self, page_id: str, filename: str) -> bytes:
        url = f"{self._base}/download/attachments/{page_id}/{filename}"
        resp = await self._http.get(url)
        resp.raise_for_status()
        return resp.content

    # ── health / sync / fetch ──────────────────────────────────────

    async def health(self) -> dict[str, object]:
        try:
            await self._get("/rest/api/content", params={"limit": 1})
        except Exception as exc:
            return {"source": self.source, "available": False, "reason": str(exc)}
        return {"source": self.source, "available": True}

    async def child_pages(self, page_id: str) -> list[dict[str, Any]]:
        payload = await self._get(
            f"/rest/api/content/{page_id}/child/page", params={"limit": 250}
        )
        return list(payload.get("results", []))

    async def sync_changes(self, cursor: str | None) -> SyncBatch:
        return SyncBatch(added=[])

    async def fetch(self, ref: object) -> object:
        raise NotImplementedError("Confluence fetch is not implemented")

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = await self._http.get(f"{self._base}{path}", params=params)
        resp.raise_for_status()
        return resp.json()


# ── helpers ────────────────────────────────────────────────────────

def extract_storage_text(page: dict[str, Any]) -> str:
    storage = (page.get("body") or {}).get("storage") or {}
    parser = _StorageTextParser()
    parser.feed(str(storage.get("value") or ""))
    return parser.text


_PDF_EXTENSIONS = {".pdf"}
_MAX_PDF_ATTACHMENTS = 10


async def _extract_pdf_text(
    http: httpx.AsyncClient, base_url: str, page_id: str, attachments: list[dict[str, Any]],
) -> tuple[str, str, list[str]]:
    """Extract text from ALL PDF attachments (up to _MAX_PDF_ATTACHMENTS).

    Returns (combined_text, status, per_file_reports).
    Status: "ok" (at least one extracted), "extraction_failed" (all failed),
    "no_pdf" (no PDFs found).
    """
    pdf_attachments = [
        att for att in attachments
        if any(str(att.get("title", "")).lower().endswith(ext) for ext in _PDF_EXTENSIONS)
    ][:_MAX_PDF_ATTACHMENTS]

    if not pdf_attachments:
        return "", "no_pdf", []

    extracted: list[str] = []
    reports: list[str] = []
    any_success = False

    for att in pdf_attachments:
        title = str(att.get("title") or "")
        try:
            url = f"{base_url}/download/attachments/{page_id}/{title}"
            resp = await http.get(url)
            resp.raise_for_status()
            text = _parse_pdf_bytes(resp.content)
            if text.strip():
                extracted.append(f"[{title}]\n{text.strip()}")
                reports.append(f"{title}: ok ({len(text)} chars)")
                any_success = True
            else:
                reports.append(f"{title}: extraction_failed:empty_pdf")
        except Exception as exc:
            reports.append(f"{title}: extraction_failed:{exc!s}"[:200])

    if any_success:
        return "\n\n---\n\n".join(extracted), "ok", reports
    return "", "extraction_failed", reports


def _parse_pdf_bytes(data: bytes) -> str:
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=data, filetype="pdf")
        parts: list[str] = []
        for page in doc:  # noqa: B007
            text = page.get_text()
            if text.strip():
                parts.append(text.strip())
        doc.close()
        return "\n".join(parts)
    except Exception:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except Exception:
        return ""


class _StorageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._parts.append(data.strip())

    @property
    def text(self) -> str:
        return re.sub(r"\s+([,.;:!?])", r"\1", " ".join(self._parts))


def _escape_query(query: str) -> str:
    return query.replace("\\", "\\\\").replace('"', '\\"')


def _extract_page_id(query: str) -> str | None:
    match = re.search(r"(?<!\w)(\d{6,})(?!\w)", query)
    return match.group(1) if match else None


def _looks_like_title(query: str) -> bool:
    return bool(query.strip()) and len(query) <= 100 and all(
        char.isalpha() or char.isspace() for char in query
    )


async def _evidence(
    page: dict[str, Any],
    base_url: str,
    source: str,
    http: httpx.AsyncClient,
) -> Evidence:
    page_id = str(page["id"])
    title = str(page.get("title") or "")
    text = extract_storage_text(page)

    content_status = "body" if text.strip() else "empty"
    attachments_checked = False
    metadata_extra: dict[str, Any] = {}

    attachments: list[dict[str, Any]] = []
    if not text.strip():
        try:
            list_payload = await _simple_get(
                http, f"{base_url}/rest/api/content/{page_id}/child/attachment",
                params={"limit": 20},
            )
            attachments = list(list_payload.get("results", []))
            attachments_checked = True
        except Exception:
            attachments = []

        if attachments:
            pdf_text, pdf_status, pdf_reports = await _extract_pdf_text(http, base_url, page_id, attachments)
            if pdf_text.strip():
                text = f"[EXTRACTED FROM ATTACHED PDF]\n{pdf_text}"
                content_status = "pdf_extracted"
                metadata_extra["pdf_reports"] = pdf_reports
            else:
                content_status = pdf_status
                metadata_extra["pdf_reports"] = pdf_reports
        else:
            content_status = "no_pdf"  # body empty, no attachments found

    return Evidence(
        id=f"{source}:{page_id}",
        document_id=page_id,
        title=title,
        text=text,
        source=source,
        uri=f"{base_url}/pages/viewpage.action?pageId={page_id}",
        origin=EvidenceOrigin.LIVE_CORPORATE,
        metadata={
            "content_status": content_status,
            "attachments_checked": attachments_checked,
            **metadata_extra,
        },
    )


async def _simple_get(
    http: httpx.AsyncClient, url: str, *, params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resp = await http.get(url, params=params)
    resp.raise_for_status()
    return resp.json()
