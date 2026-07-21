# ADR-009: Full-Text Extraction for Allowlisted Web

**Status:** Proposed
**Date:** 2026-07-22
**Extends:** ADR-006 (web research), ADR-004 (trusted execution domain)

## 1. Context

The gateway has two web retrieval paths:

| Path | Connector | Extraction |
|------|-----------|------------|
| Allowlisted (authoritative) | `AllowlistedWebConnector` | SearXNG snippet only (~200 chars) |
| Generic (disabled) | `CamoufoxConnector` via `web_pipeline.py` | Trafilatura full-text |

The `web_pipeline.py` module implements `SearXNG → Trafilatura → Camoufox` but `AllowlistedWebConnector` does not use it. Instead, it takes `item["content"]` directly from SearXNG results — a snippet, not the full page text.

**Audit finding:** the authoritative path returns snippets, which may not contain the specific version/matrix/path information the user needs.

## 2. Problem

For a query like "матрица совместимости ALD Pro 3.2.0", the SearXNG snippet might show:
```text
"Матрица совместимости ALD Pro 3.2.0. Пути обновления: 2.4.4 → 3.2.0 ..."
```

But the actual page `aldpro.ru/materials/3.2.0/matrix` contains a full HTML table with dozens of upgrade paths. The snippet may not include the specific row for the source version the user is asking about.

Trafilatura extraction would capture the full table, making authoritative retrieval actually authoritative.

## 3. Decision

**Call Trafilatura from `AllowlistedWebConnector` for the top-N results.**

```python
# In AllowlistedWebConnector.search_live():
for item in data.get("results", [])[:request.topk]:
    # 1. Try full-text extraction
    full_text = await _extract_full_text(item["url"])
    # 2. Fall back to snippet if extraction fails
    evidence_text = full_text or item.get("content", "")
```

`_extract_full_text()` calls Trafilatura (already in base dependencies) with:
- `fetch_url(url)` or `extract(html_content)` for pre-fetched pages
- Configurable timeout (default 10s per URL)
- Configurable max chars per page (default 16000)

**NOT included:** Camoufox headless browser. Trafilatura alone handles HTML pages; Camoufox remains in `web_pipeline.py` for JS-heavy pages, which is irrelevant for `aldpro.ru`/`astralinux.ru` (static documentation sites).

## 4. Consequences

- **Positive:** authoritative retrieval returns full page text, not snippets. Matrix/version/path information is complete.
- **Positive:** `aldpro.ru` and `astralinux.ru` are static HTML — Trafilatura handles them without a browser.
- **Negative:** adds HTTP latency per result (mitigated by per-URL timeout and parallel extraction via `asyncio.gather`).
- **Risk:** some authoritative pages may be JS-rendered. Mitigation: snippet fallback when Trafilatura returns empty.

## 5. Verification

```text
Mock SearXNG response with 2 results:
  result[0]: aldpro.ru/matrix → Trafilatura returns full text
  result[1]: aldpro.ru/guide → Trafilatura fails → snippet fallback
→ Evidence[0].text contains full table
→ Evidence[1].text contains snippet
→ extraction metadata: {"trafilatura": "ok"|"fallback"}
```
