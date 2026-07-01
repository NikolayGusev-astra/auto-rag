"""
Async RAG pipeline — параллельный запуск ZVec, MCP, SearXNG через asyncio.gather().
Включает RagTrace для прозрачного трейсинга каждого этапа.

Архитектура v3:
- rusbitech: Lodestone MCP → web (Trafilatura) → empty (ZVec ВЫКЛЮЧЕН — 0% accuracy)
- devops/software-dev: ZVec → context7 → web (Trafilatura)
- LLM quality gate на всех источниках перед возвратом
- Trafilatura для полного текста веб-страниц
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

sys.path.insert(0, os.path.dirname(__file__))
from rag_config import *
from rag_mcp_client import MCPClient
from rag_trace import RagTrace

_EXECUTOR = ThreadPoolExecutor(max_workers=6)

# ── LRU cache ──────────────────────────────────────
_CACHE = OrderedDict()
_CACHE_MAX = 100


def _cache_key(query: str, domain: str) -> str:
    return hashlib.md5(f"{query}|{domain}".encode()).hexdigest()


def _cache_get(key: str) -> dict | None:
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    return None


def _cache_set(key: str, result: dict):
    _CACHE[key] = result
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


# ── Embedding ──────────────────────────────────────

def _embed(text: str) -> list[float]:
    """LM Studio embedding."""
    try:
        r = requests.post(EMBEDDING_URL, json={
            'model': EMBEDDING_MODEL,
            'input': [text[:2000]],
        }, timeout=30)
        return r.json()['data'][0]['embedding']
    except Exception:
        return [0.0] * EMBEDDING_DIM


# ── ZVec (только для non-rusbitech) ────────────────

def _blocking_zvec(query: str) -> dict:
    import zvec
    from zvec import Query as ZQ
    emb = _embed(query)
    coll = _get_zvec_collection()
    # Все категории — wiki, .autolycus, llm-wiki — все нужны для контекста
    # Но wiki приоритетнее: +0.10 к score, llm-wiki: +0.05
    doclist = coll.query(queries=[ZQ(field_name="embedding", vector=emb)], topk=10,
                         output_fields=["source", "heading", "category", "node", "content", "title"])
    chunks = []
    for d in doclist:
        txt = d.fields.get('text', '') or d.fields.get('content', '')
        if txt:
            cat = (d.fields or {}).get("category", "")
            boost = 0.10 if cat == "wiki" else (0.05 if cat == "llm-wiki" else 0.0)
            score = min(d.score + boost, 1.0)
            chunks.append({"text": txt[:500], "score": score,
                          "source": (d.fields or {}).get("source", "zvec/wiki"),
                          "category": cat})
    chunks.sort(key=lambda c: c["score"], reverse=True)
    return {"chunks": chunks[:5], "max_score": max((c["score"] for c in chunks), default=0)}


# ── MCP ────────────────────────────────────────────

def _blocking_mcp_single(name: str, query: str) -> list[dict]:
    cfg = MCP_SERVERS.get(name)
    if not cfg:
        return []
    mc = MCPClient(timeout=15)
    return mc.query(name, cfg, query, 3)


# ── Web search (SearXNG + DDG + Trafilatura) ───────

def _blocking_web(query: str, domain: str = "", collection: str = "") -> list[dict]:
    """Web search with Trafilatura full-text extraction."""
    preferred = None
    if domain and collection:
        dm = DCD_PREFERRED_WEB_SOURCE.get(domain, {})
        preferred = dm.get(collection) or dm.get('*')
    if preferred == 'skip':
        return []

    def _extract_full_text(url: str) -> str | None:
        """Trafilatura full-text extraction from URL."""
        try:
            import trafilatura
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded, output_format="txt")
                if text and len(text) > 50:
                    return text[:3000]
        except Exception:
            pass
        # Fallback: requests + trafilatura
        try:
            resp = requests.get(url, timeout=8,
                                headers={'User-Agent': 'Mozilla/5.0'})
            if resp.status_code == 200:
                import trafilatura
                text = trafilatura.extract(resp.text, output_format="txt")
                if text and len(text) > 50:
                    return text[:3000]
        except Exception:
            pass
        return None

    def _searxng(q: str) -> list[dict]:
        from urllib.parse import quote
        url = f'{SEARXNG_URL}/search?q={quote(q)}&format=json&pageno=1'
        try:
            r = requests.get(url, timeout=10,
                             headers={'User-Agent': 'HermesRAG/1.0'})
            if r.status_code == 200:
                data = r.json()
                results = data.get('results', [])[:5]
                chunks = []
                for wr in results:
                    text = wr.get('content', '') or wr.get('snippet', '')
                    url = wr.get('url', '')
                    if text or url:
                        # Try full-text extraction
                        full = _extract_full_text(url) if url else None
                        chunks.append({
                            'text': (full or text)[:3000],
                            'title': wr.get('title', ''),
                            'url': url,
                            'source': 'web/searxng',
                            'full_text': bool(full),
                        })
                return chunks
        except Exception:
            pass
        return []

    def _ddg(q: str) -> list[dict]:
        from urllib.parse import quote
        import re
        try:
            r = requests.get(
                f'https://html.duckduckgo.com/html/?q={quote(q[:200])}',
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=10,
            )
            if r.status_code == 200:
                chunks = []
                for block in re.findall(
                        r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>.*?<a[^>]+class="result__snippet"[^>]*>([^<]*)</a>',
                        r.text, re.DOTALL):
                    url, title, snippet = block
                    text = (snippet or '').strip()
                    if text:
                        full = _extract_full_text(url)
                        chunks.append({
                            'text': (full or text)[:3000],
                            'title': title.strip(),
                            'url': url,
                            'source': 'web/ddg',
                            'full_text': bool(full),
                        })
                return chunks[:5]
        except Exception:
            pass
        return []

    if preferred == 'ddg':
        return _ddg(query)
    chunks = _searxng(query)
    if not chunks:
        chunks = _ddg(query)
    return chunks


# ── LLM quality gate ───────────────────────────────

def _llm_verify(query: str, chunks: list[dict]) -> float:
    """Soft verification: returns 0.0-1.0 relevance score.
    ≥ 0.3 = pass (chunks are relevant enough).
    Uses qwen2.5-7b-instruct (non-thinking, fast).
    """
    if not chunks:
        return 0.0
    import re
    top = '\n\n'.join(
        [f'[{i}] {c["text"][:500].replace(chr(10), " ")}'
         for i, c in enumerate(chunks[:3])]
    )
    prompt = (
        f'Rate relevance 0.0-1.0. Reply ONLY a number.\n'
        f'Query: {query[:200]}\nDocuments:\n{top}'
    )
    try:
        r = requests.post(LM_STUDIO_CHAT_URL, json={
            'model': 'qwen2.5-7b-instruct',
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.0, 'max_tokens': 10,
        }, timeout=10)
        answer = r.json()['choices'][0]['message']['content'].strip()
        nums = re.findall(r'0\.\d+|1\.0', answer)
        return float(nums[0]) if nums else 0.3  # default pass on parse error
    except Exception:
        return 0.5  # pass on error (don't block)


def _blocking_llm_eval(query: str, chunks: list) -> float:
    import re
    top = '\n\n'.join(
        [f'[{i}] {c["text"][:300].replace(chr(10), " ")}'
         for i, c in enumerate(chunks[:3])])
    prompt = (f'Rate relevance 0.0-1.0. Reply ONLY a number.\n'
              f'Query: {query[:200]}\nDocuments:\n{top}')
    try:
        r = requests.post(LM_STUDIO_CHAT_URL, json={
            'model': 'qwen2.5-7b-instruct',
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.0, 'max_tokens': 20,
        }, timeout=15)
        content = r.json()['choices'][0]['message']['content'].strip()
        nums = re.findall(r'0\.\d+|1\.0', content)
        return float(nums[0]) if nums else 0.0
    except Exception:
        return 0.0


# ── Entity Match ──────────────────────────────────

_ENTITY_EXTRACTOR_CACHE = {}


def _extract_entities(query: str) -> set[str]:
    key = hash(query) % 1000000
    if key in _ENTITY_EXTRACTOR_CACHE:
        return _ENTITY_EXTRACTOR_CACHE[key]
    import re
    entities = set()
    ql = query.lower()
    urls = re.findall(r'https?://[^\s]+', query)
    entities.update(urls)
    for u in urls:
        domain = re.findall(r'://([^/]+)', u)
        if domain:
            entities.add(domain[0])
            parts = domain[0].split('.')
            if len(parts) >= 2:
                entities.add(parts[-2])
                if len(parts) >= 3:
                    entities.add(parts[0])
    known_products = [
        'terraform', 'ansible', 'docker', 'kubernetes', 'postgresql',
        'postgres', 'redis', 'nginx', 'acm', 'ald pro', 'aldpro',
        'rupest', 'alse', 'keycloak', 'workspad', 'freeipa',
        'samba', 'sssd', 'proxmox', 'patroni', 'reprepro',
        'letsencrypt', 'certbot', 'gitlab', 'prometheus', 'grafana',
        'elasticsearch', 'kafka', 'rabbitmq', 'mongodb', 'mysql',
        'pgsql', 'mariadb', 'haproxy', 'keepalived',
        'astra', 'astralinux', 'astracloud', 'laika',
    ]
    for prod in known_products:
        if prod in ql:
            entities.add(prod)
    _GENERIC_SHORT = {'ip', 'ha', 'vm', 'ac', 'dc', 'ok', 'id', 'api', 'db',
                      'os', 'ui', 'ux', 'pc', 'io', 'cl', 'fe', 'be', 'sla',
                      'tls', 'ssh', 'dns', 'dhcp', 'nat', 'vpn', 'lan', 'wan',
                      'smtp', 'pop', 'tcp', 'udp', 'http', 'html', 'xml', 'json'}
    caps = re.findall(r'\b[A-Z][a-zA-Z0-9_-]{2,}\b', query)
    for c in caps:
        low = c.lower()
        if low not in ('the', 'this', 'that', 'what', 'how', 'why',
                       'not', 'for', 'and', 'with'):
            if low not in _GENERIC_SHORT:
                entities.add(low)
    _ENTITY_EXTRACTOR_CACHE[key] = entities
    return entities


def _check_entities_in_query(query: str, chunks: list[dict]) -> bool | None:
    entities = _extract_entities(query)
    tech_entities = {e for e in entities
                     if not e.startswith('http') and len(e) > 2}
    if not entities and not tech_entities:
        return None
    chunk_text = ' '.join([c.get('text', '') for c in chunks]).lower()
    missing = []
    for ent in entities:
        if ent.startswith('http'):
            continue
        if ent not in chunk_text:
            missing.append(ent)
    checked = [e for e in entities
               if not e.startswith('http') and len(e) > 3]
    if not checked:
        return None
    missing_ratio = len(missing) / max(len(checked), 1)
    return False if missing_ratio >= 0.33 else True


# ── Fallback helper ───────────────────────────────

async def _fallback_to_mcp_web(
    query: str, domain: str, collection: str, loop,
    trace: RagTrace | None = None,
) -> dict:
    dm = DCD_COLLECTION_MCP_MAP.get(domain, {})
    primary = dm.get(collection) or dm.get('*')
    sources = []
    if primary and primary in MCP_SERVERS:
        sources.append(primary)
    if domain == 'internal' and 'confluence' in MCP_SERVERS \
            and 'confluence' not in sources:
        sources.append('confluence')
    if primary != 'lodestone' and 'lodestone' in MCP_SERVERS:
        sources.append('lodestone')
    if primary != 'jira' and 'jira' in MCP_SERVERS and domain == 'internal':
        sources.append('jira')

    if trace:
        trace.decision("mcp_source_selection",
                       choice=str(sources),
                       reason=f"primary={primary}, domain={domain}, "
                              f"collection={collection}")

    if not sources:
        if trace:
            trace.event("mcp_fallback", status="skip",
                        reason="no sources matched")
        return {'source': None, 'chunks': []}

    tasks = [loop.run_in_executor(_EXECUTOR, _blocking_mcp_single, s, query)
             for s in sources]
    all_results = await asyncio.gather(*tasks)

    best_chunks = []
    best_src = None
    for src, chunks in zip(sources, all_results):
        if trace:
            trace.event("mcp_result", source=src, chunks=len(chunks),
                        status="ok" if chunks else "empty")
        if chunks and len(chunks) > len(best_chunks):
            best_chunks = chunks
            best_src = src

    if best_chunks:
        if trace:
            trace.decision("mcp_selected", choice=best_src or "none",
                           reason=f"best of {len(sources)} sources, "
                                  f"{len(best_chunks)} chunks")
        return {'source': best_src, 'chunks': best_chunks, 'score': 0.7,
                'trace': f'MCP({best_src})'}
    if trace:
        trace.event("mcp_fallback", status="empty",
                    reason="all MCP sources returned empty")
    return {'source': None, 'chunks': []}


# ── Main entry ────────────────────────────────────

async def async_rag_search(
    query: str, dcd_result: dict,
    trace: RagTrace | None = None,
) -> dict:
    domain = dcd_result.get('domain', '')
    collection = dcd_result.get('collection', '')
    confidence = dcd_result.get('confidence', 0)
    ck = _cache_key(query, domain)

    if trace is None:
        trace = RagTrace(query, domain, collection)
    else:
        trace.domain = domain
        trace.collection = collection

    trace.event("dcd_result", domain=domain, collection=collection,
                confidence=confidence,
                fallback=dcd_result.get('fallback', False))

    cached = _cache_get(ck)
    if cached:
        trace.event("cache_hit", source=cached.get('source', '?'),
                    chunks=len(cached.get('chunks', [])))
        cached['_trace'] = trace.json()
        return cached

    loop = asyncio.get_running_loop()

    # ═══════════════════════════════════════════════
    # PATH A: rusbitech — ZVec + Lodestone + MCP + Web
    # ZVec включён, но с фильтром category="wiki"
    # ═══════════════════════════════════════════════
    if domain == 'internal':
        trace.event("rusbitech_path",
                     note="ZVec + Lodestone + MCP for rusbitech")

        # Force route для jira-target коллекций
        _FORCE_JIRA = {'rusbitech-rca', 'rusbitech-presale',
                       'rusbitech-vulnerability', 'rusbitech-security',
                       'rusbitech-customers'}
        if collection in _FORCE_JIRA:
            trace.decision("force_route", choice=f"ForceJira→{collection}",
                           reason="jira-only collection")
            with trace.stage("force_jira"):
                result = await _fallback_to_mcp_web(
                    query, domain, collection, loop, trace)
                if result['chunks']:
                    result['trace'] = 'ForceJira→' + result.get('trace', '')
                    result['_trace'] = trace.json()
                    _cache_set(ck, result)
                    return result
            # Jira empty → web
            with trace.stage("web_fallback"):
                web_chunks = await loop.run_in_executor(
                    _EXECUTOR, _blocking_web, query, domain, collection)
                trace.event("web_result", chunks=len(web_chunks))
                if web_chunks:
                    result = {'source': 'web', 'chunks': web_chunks,
                              'score': 0.6,
                              'trace': 'ForceJira→empty→Web',
                              '_trace': trace.json()}
                    _cache_set(ck, result)
                    return result
            result = {'source': 'empty', 'chunks': [], 'score': 0,
                      'trace': 'ForceJira→empty', '_trace': trace.json()}
            _cache_set(ck, result)
            return result

        # Основной путь: ZVec (wiki) + Lodestone параллельно
        trace.decision("rusbitech_path", choice="zvec+lodestone",
                       reason="parallel ZVec (category=wiki) + Lodestone MCP")

        with trace.stage("rusbitech_parallel"):
            zvec_task = loop.run_in_executor(_EXECUTOR, _blocking_zvec, query)
            lodestone_task = loop.run_in_executor(
                _EXECUTOR, _blocking_mcp_single, 'lodestone', query)
            confluence_task = loop.run_in_executor(
                _EXECUTOR, _blocking_mcp_single, 'confluence', query)
            web_task = loop.run_in_executor(_EXECUTOR, _blocking_web,
                                            query, domain, collection)

        with trace.stage("gather_rusbitech"):
            zvec_result, lodestone_chunks, confluence_chunks, web_chunks = await asyncio.gather(
                zvec_task, lodestone_task, confluence_task, web_task)

        zvec_chunks = zvec_result['chunks']
        max_score = zvec_result['max_score']
        trace.event("zvec_result", chunks=len(zvec_chunks),
                    max_score=round(max_score, 4))
        trace.event("lodestone_result", chunks=len(lodestone_chunks),
                    status="ok" if lodestone_chunks else "empty")
        trace.event("confluence_result", chunks=len(confluence_chunks),
                    status="ok" if confluence_chunks else "empty")
        trace.event("web_result", chunks=len(web_chunks))

        # Приоритет: Lodestone > Confluence > ZVec > Web
        best_source = None
        best_chunks = None

        # 1. Lodestone — всегда лучший (Confluence docs)
        if lodestone_chunks:
            with trace.stage("llm_verify_lodestone"):
                verified = await loop.run_in_executor(
                    _EXECUTOR, _llm_verify, query, lodestone_chunks)
                trace.event("llm_verify_result", source="lodestone",
                            verified=verified)
            if verified >= 0.3:
                best_source = 'lodestone'
                best_chunks = lodestone_chunks
                trace.decision("source_selection", choice="lodestone",
                               reason=f"lodestone + LLM score={verified:.2f}")

        # 2. Confluence — если Lodestone пуст
        if best_source is None and confluence_chunks:
            with trace.stage("llm_verify_confluence"):
                verified = await loop.run_in_executor(
                    _EXECUTOR, _llm_verify, query, confluence_chunks)
                trace.event("llm_verify_result", source="confluence",
                            verified=verified)
            if verified >= 0.3:
                best_source = 'confluence'
                best_chunks = confluence_chunks
                trace.decision("source_selection", choice="confluence",
                               reason=f"confluence + LLM score={verified:.2f}")

        # 3. ZVec (wiki) — если Confluence не подошёл
        if best_source is None and zvec_chunks:
            # Для wiki-контента порог ниже (0.35 вместо 0.75)
            # bge-m3 даёт 0.30-0.45 для коротких wiki-документов
            if max_score >= 0.35:
                with trace.stage("llm_verify_zvec"):
                    verified = await loop.run_in_executor(
                        _EXECUTOR, _llm_verify, query, zvec_chunks)
                    trace.event("llm_verify_result", source="zvec",
                                verified=verified, score=round(max_score, 2))
                if verified >= 0.3:
                    best_source = 'zvec'
                    best_chunks = zvec_chunks
                    trace.decision("source_selection", choice="zvec",
                                   reason=f"zvec score={max_score:.2f} + LLM score={verified:.2f}")
            else:
                trace.event("zvec_low_score", score=round(max_score, 3),
                            note="below 0.35 threshold for wiki")

        # 4. Web — если ZVec не подошёл
        if best_source is None and web_chunks:
            with trace.stage("llm_verify_web"):
                verified = await loop.run_in_executor(
                    _EXECUTOR, _llm_verify, query, web_chunks)
                trace.event("llm_verify_result", source="web",
                            verified=verified)
            if verified >= 0.3:
                best_source = 'web'
                best_chunks = web_chunks
                trace.decision("source_selection", choice="web",
                               reason=f"web + LLM score={verified:.2f}")

        if best_source:
            result = {'source': best_source, 'chunks': best_chunks,
                      'score': 0.7, '_trace': trace.json()}
            _cache_set(ck, result)
            return result

        result = {'source': 'empty', 'chunks': [], 'score': 0,
                  'trace': 'ZVec+Lodestone+Web→no_valid_answer',
                  '_trace': trace.json()}
        _cache_set(ck, result)
        return result

        result = {'source': 'empty', 'chunks': [], 'score': 0,
                  'trace': 'Lodestone→Web→no_valid_answer',
                  '_trace': trace.json()}
        _cache_set(ck, result)
        return result

    # ═══════════════════════════════════════════════
    # PATH B: devops/software-dev — ZVec → MCP → web
    # ═══════════════════════════════════════════════

    trace.event("parallel_start", sources="zvec+web")

    with trace.stage("zvec_search"):
        zvec_task = loop.run_in_executor(_EXECUTOR, _blocking_zvec, query)
        web_task = loop.run_in_executor(_EXECUTOR, _blocking_web,
                                        query, domain, collection)

    with trace.stage("gather_parallel"):
        zvec_result, web_chunks = await asyncio.gather(
            zvec_task, web_task)

    zvec_chunks = zvec_result['chunks']
    max_score = zvec_result['max_score']
    trace.event("zvec_result", chunks=len(zvec_chunks),
                max_score=round(max_score, 4))
    trace.event("web_result", chunks=len(web_chunks))

    # Entity match for ZVec
    with trace.stage("entity_match"):
        _zvec_entities_match = await loop.run_in_executor(
            _EXECUTOR, _check_entities_in_query, query, zvec_chunks)
        trace.event("entity_match_result",
                    zvec_match=str(_zvec_entities_match))

    # Entity mismatch → MCP/Web
    if _zvec_entities_match is False:
        trace.decision("source_selection", choice="mcp_fallback",
                       reason="zvec entity mismatch")
        with trace.stage("mcp_fallback"):
            result = await _fallback_to_mcp_web(
                query, domain, collection, loop, trace)
            if result['chunks']:
                result['_trace'] = trace.json()
                _cache_set(ck, result)
                return result
        if web_chunks:
            trace.decision("source_selection", choice="web",
                           reason="mcp empty")
            result = {'source': 'web', 'chunks': web_chunks,
                      'score': 0.6,
                      'trace': f'ZVec→EntityMismatch→Web',
                      '_trace': trace.json()}
            _cache_set(ck, result)
            return result

    # High score ZVec → LLM verify
    if max_score >= LLM_EVAL_HIGH_THRESHOLD and _zvec_entities_match is not False:
        with trace.stage("llm_verify_zvec"):
            verified = await loop.run_in_executor(
                _EXECUTOR, _llm_verify, query, zvec_chunks)
            trace.event("llm_verify_result", verified=verified,
                        score=round(max_score, 2))
        if verified >= 0.3:
            trace.decision("source_selection", choice="zvec",
                           reason=f"zvec score {max_score:.2f} + LLM score={verified:.2f}")
            result = {'source': 'zvec', 'chunks': zvec_chunks,
                      'score': max_score,
                      'trace': f'ZVec({max_score:.2f}→LLM{verified:.2f})',
                      '_trace': trace.json()}
            _cache_set(ck, result)
            return result
        else:
            trace.event("zvec_llm_rejected",
                        note=f"LLM rejected zvec score={max_score:.2f}")

    # Low DCD confidence → MCP/Web
    if confidence < 0.20:
        trace.decision("source_selection", choice="mcp_fallback",
                       reason=f"dcd confidence {confidence:.2f} < 0.2")
        with trace.stage("mcp_fallback_lowconf"):
            result = await _fallback_to_mcp_web(
                query, domain, collection, loop, trace)
            if result['chunks']:
                result['_trace'] = trace.json()
                _cache_set(ck, result)
                return result
        if web_chunks:
            result = {'source': 'web', 'chunks': web_chunks,
                      'score': 0.6,
                      'trace': f'DCD(conf={confidence:.2f}<0.2)→Web',
                      '_trace': trace.json()}
            _cache_set(ck, result)
            return result
        result = {'source': 'empty', 'chunks': [], 'score': 0,
                  'trace': f'DCD(conf={confidence:.2f}<0.2)→empty',
                  '_trace': trace.json()}
        _cache_set(ck, result)
        return result

    # LLM eval for borderline scores
    if max_score >= LLM_EVAL_LOW_THRESHOLD:
        with trace.stage("llm_eval"):
            llm_score = await loop.run_in_executor(
                _EXECUTOR, _blocking_llm_eval, query, zvec_chunks)
            trace.event("llm_eval_result", score=llm_score)
            if llm_score >= 0.5:
                trace.decision("source_selection", choice="zvec+llm",
                               reason=f"llm_score {llm_score:.2f} ≥ 0.5")
                result = {'source': 'zvec+llm', 'chunks': zvec_chunks,
                          'score': llm_score,
                          'trace': f'ZVec({max_score:.2f})→Qwen({llm_score:.2f})',
                          '_trace': trace.json()}
                _cache_set(ck, result)
                return result

    # Final MCP
    with trace.stage("final_mcp"):
        result = await _fallback_to_mcp_web(
            query, domain, collection, loop, trace)
        if result['chunks']:
            result['_trace'] = trace.json()
            _cache_set(ck, result)
            return result

    if web_chunks:
        trace.decision("source_selection", choice="web",
                       reason="all MCP empty, final web")
        result = {'source': 'web', 'chunks': web_chunks, 'score': 0.6,
                  'trace': 'ZVec→MCP→Web', '_trace': trace.json()}
        _cache_set(ck, result)
        return result

    trace.decision("source_selection", choice="empty",
                   reason="all sources returned empty")
    result = {'source': 'empty', 'chunks': [], 'score': 0,
              'trace': 'ZVec→MCP→Web→empty', '_trace': trace.json()}
    _cache_set(ck, result)
    return result