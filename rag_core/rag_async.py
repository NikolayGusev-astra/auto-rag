"""
Async RAG pipeline — параллельный запуск ZVec, MCP, SearXNG через asyncio.gather().
Включает RagTrace для прозрачного трейсинга каждого этапа.

Архитектура v3:
- rusbitech: ZVec early exit (score >= 0.45) → skip slow sources
- devops/software-dev: ZVec → MCP → web fallback
- LLM quality gate на quick verify (qwen2.5-7b, ~2.7s)
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

# ── Routing log ──
_ROUTING_LOG = os.path.join(os.path.dirname(__file__), "routing_log.jsonl")
_LAST_DCD = None  # set by async_rag_search for logging
def _log_routing(query: str, dcd: dict, result: dict):
    """Записать маршрутизацию запроса для DCD Learner."""
    try:
        entry = {
            "query": query[:200],
            "dcd_domain": dcd.get("domain", ""),
            "dcd_collection": dcd.get("collection", ""),
            "dcd_confidence": dcd.get("confidence", 0),
            "actual_source": result.get("source", "?"),
            "has_content": len(result.get("chunks", [])) > 0,
            "chunks_count": len(result.get("chunks", [])),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(_ROUTING_LOG, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ── LRU cache: 100 last queries ──────────────────────────────────
_CACHE = OrderedDict()
_CACHE_MAX = 100

def _cache_key(query: str, domain: str) -> str:
    return hashlib.md5(f"{query}|{domain}".encode()).hexdigest()

def _cache_get(key: str) -> dict | None:
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    return None

def _cache_set(key: str, result: dict, dcd: dict | None = None):
    _CACHE[key] = result
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    # Log routing
    d = dcd or _LAST_DCD
    if d is not None:
        query = ""
        trace_val = result.get("_trace", "")
        if isinstance(trace_val, str) and trace_val.startswith("{"):
            try:
                td = json.loads(trace_val)
                query = td.get("query", "")
            except:
                pass
        _log_routing(query[:200] if query else key[:30], d, result)


# ── Embedding ────────────────────────────────────────────────────
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


# ── ZVec (singleton) ─────────────────────────────────────────────
_ZVEC_COLLECTION = None
_ZVEC_COLLECTION_LOCK = None

def _get_zvec_collection():
    global _ZVEC_COLLECTION, _ZVEC_COLLECTION_LOCK
    if _ZVEC_COLLECTION is not None:
        return _ZVEC_COLLECTION
    import threading
    if _ZVEC_COLLECTION_LOCK is None:
        _ZVEC_COLLECTION_LOCK = threading.Lock()
    with _ZVEC_COLLECTION_LOCK:
        if _ZVEC_COLLECTION is not None:
            return _ZVEC_COLLECTION
        import zvec
        import os
        zpath = os.path.join(ZVEC_PATH, ZVEC_COLLECTION)
        # Simple file-based lock for ZVec 0.5.1
        lock_path = zpath + "/LOCK"
        try:
            with open(lock_path, 'w') as f:
                f.write(str(os.getpid()))
        except:
            pass
        _ZVEC_COLLECTION = zvec.open(zpath)
        return _ZVEC_COLLECTION


# ── Blocking helpers (thread pool) ───────────────────────────────
def _blocking_zvec(query: str) -> dict:
    """ZVec search — tries FastAPI server first, falls back to direct ZVec."""
    # Try FastAPI server (persistent, no init overhead)
    try:
        import urllib.request
        import json
        encoded = urllib.parse.quote(query[:100])
        url = f"http://127.0.0.1:8678/search?q={encoded}&topk=5"
        resp = urllib.request.urlopen(url, timeout=10)
        data = json.loads(resp.read().decode())
        if data.get("chunks"):
            return {"chunks": data["chunks"], "max_score": data["max_score"],
                    "source": "zvec_fastapi"}
    except Exception:
        pass  # FastAPI unavailable, return empty (ZVec locked by server)
    return {"chunks": [], "max_score": 0, "source": "zvec_unavailable"}


def _blocking_mcp_single(name: str, query: str) -> list[dict]:
    """Single MCP server query (blocking)."""
    cfg = MCP_SERVERS.get(name)
    if not cfg:
        return []
    mc = MCPClient(timeout=15)
    return mc.query(name, cfg, query, 3)


def _blocking_web(query: str, domain: str = "", collection: str = "") -> list[dict]:
    """Web search via SearXNG + Trafilatura."""
    import urllib.parse
    import subprocess as _sp
    preferred = DCD_PREFERRED_WEB_SOURCE.get(domain, {}).get(collection) or \
                DCD_PREFERRED_WEB_SOURCE.get(domain, {}).get("*")
    if preferred == "skip":
        return []
    encoded = urllib.parse.quote(query)
    try:
        r = _sp.run(["curl", "-s", "--max-time", "10", f"{SEARXNG_URL}/search?q={encoded}&format=json"],
                    capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout:
            data = json.loads(r.stdout)
            results = data.get("results", [])[:WEB_SEARCH_MAX_RESULTS]
            chunks = []
            for wr in results:
                text = wr.get("content", "") or wr.get("snippet", "")
                url = wr.get("url", "")
                if text and len(chunks) < 1:
                    try:
                        import urllib.request
                        import trafilatura
                        req = urllib.request.Request(url, headers={
                            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})
                        resp = urllib.request.urlopen(req, timeout=8)
                        html = resp.read().decode("utf-8", errors="replace")
                        full = trafilatura.extract(html)
                        if full:
                            text = full[:2000]
                    except:
                        pass
                if text:
                    chunks.append({"text": text[:800], "source": "web", "url": url})
            return chunks
    except Exception:
        return []
    return []


# ── LLM Verify (fast) ────────────────────────────────────────────
def _llm_verify(query: str, chunks: list[dict]) -> float:
    """Soft verification: returns 0.0-1.0 relevance score.
    ≥ 0.3 = pass (chunks are relevant enough).
    Uses local LM Studio qwen2.5-7b-instruct (~2.7s).
    """
    if not chunks or not LLM_VERIFY_ENABLED:
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
        r = requests.post(LLM_VERIFY_URL, json={
            'model': LLM_VERIFY_MODEL,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.0, 'max_tokens': 10,
        }, timeout=LLM_VERIFY_TIMEOUT)
        answer = r.json()['choices'][0]['message']['content'].strip()
        nums = re.findall(r'0\.\d+|1\.0', answer)
        return float(nums[0]) if nums else 0.3  # default pass on parse error
    except Exception:
        return 0.5  # pass on error (don't block)


# ── LLM Eval (for borderline scores) ─────────────────────────────
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
            'temperature': 0.0, 'max_tokens': 10,
        }, timeout=15)
        answer = r.json()['choices'][0]['message']['content'].strip()
        nums = re.findall(r'0\.\d+|1\.0', answer)
        return float(nums[0]) if nums else 0.0
    except Exception:
        return 0.0


# ── Entity extraction & matching ────────────────────────────────
_ENTITY_EXTRACTOR_CACHE = {}

def _extract_entities(query: str) -> set[str]:
    """Extract technical entities from query."""
    import re
    cache_key = hashlib.md5(query.encode()).hexdigest()
    if cache_key in _ENTITY_EXTRACTOR_CACHE:
        return _ENTITY_EXTRACTOR_CACHE[cache_key]
    # Technical terms
    entities = set()
    # English tech terms
    entities.update(re.findall(r'\b[A-Za-z][A-Za-z0-9_.-]{2,}\b', query))
    # Russian tech terms
    entities.update(re.findall(r'\b[А-Яа-яё][А-Яа-яё]{3,}\b', query))
    # Filter stop words
    stop = {'the', 'and', 'for', 'not', 'how', 'why', 'what', 'this', 'that',
            'это', 'что', 'как', 'все', 'если', 'котор', 'можно', 'тольк',
            'есть', 'при', 'ваш', 'меня', 'быть', 'когда', 'после', 'через'}
    entities = {e for e in entities if e.lower() not in stop and len(e) > 2}
    _ENTITY_EXTRACTOR_CACHE[cache_key] = entities
    return entities


def _check_entities_in_query(query: str, chunks: list[dict]) -> bool | None:
    """Check if extracted entities appear in chunks."""
    if not chunks:
        return None
    entities = _extract_entities(query)
    if not entities:
        return None
    chunk_text = " ".join(c.get("text", "") for c in chunks).lower()
    matches = sum(1 for e in entities if e.lower() in chunk_text)
    return matches / len(entities) >= 0.5 if entities else None


# ── MCP Fallback ────────────────────────────────────────────────
async def _fallback_to_mcp_web(
    query: str, domain: str, collection: str, loop: asyncio.AbstractEventLoop, trace: RagTrace
) -> dict:
    """Try MCP servers in priority order."""
    mcp = MCPClient(timeout=15)
    # Priority for non-rusbitech: context7 > jira > confluence > lodestone > protopack
    for name in ['context7', 'jira', 'confluence', 'lodestone', 'protopack']:
        if name in MCP_SERVERS:
            with trace.stage(f"mcp_{name}"):
                chunks = await loop.run_in_executor(_EXECUTOR, _blocking_mcp_single, name, query)
                trace.event("mcp_result", source=name, chunks=len(chunks))
            if chunks:
                trace.decision("mcp_selected", choice=name,
                               reason=f"MCP {name} returned {len(chunks)} chunks")
                return {'source': name, 'chunks': chunks, 'score': 0.7}
    # Web fallback
    web_chunks = await loop.run_in_executor(_EXECUTOR, _blocking_web, query, domain, collection)
    if web_chunks:
        trace.decision("source_selection", choice="web", reason="all MCP empty, web fallback")
        return {'source': 'web', 'chunks': web_chunks, 'score': 0.6}
    return {'source': 'empty', 'chunks': [], 'score': 0}


def _is_noise_chunk(text: str) -> bool:
    """Проверяет что чанк — не мусор (ошибка/нет результатов)."""
    if not text:
        return True
    noise_phrases = ["no results", "Error executing tool", "All configured sources failed",
                     "not found", "Lodestone Search: no results"]
    return any(p in text.lower() for p in noise_phrases)


# ── Main entry ──────────────────────────────────────────────────
async def async_rag_search(
    query: str, dcd_result: dict,
    trace: RagTrace | None = None,
) -> dict:
    global _LAST_DCD
    _LAST_DCD = dcd_result
    domain = dcd_result.get('domain', '')
    collection = dcd_result.get('collection', '')
    confidence = dcd_result.get('confidence', 0)
    ck = _cache_key(query, domain)
    loop = asyncio.get_event_loop()

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
        trace.event("cache_hit", source=cached.get('source', '?'))
        cached['_trace'] = trace.json()
        return cached

    # ── FORCE PATHS ──

    # Jira for security/rca/presale
    if collection in ('rusbitech-security', 'rusbitech-rca', 'rusbitech-presale') and 'jira' in MCP_SERVERS:
        trace.decision("rusbitech_path", choice=f"ForceJira→{collection}",
                       reason=f"DCD collection={collection} forces Jira")
        with trace.stage("force_jira"):
            mcp = MCPClient(timeout=15)
            jira_chunks = await loop.run_in_executor(_EXECUTOR, _blocking_mcp_single, 'jira', query)
            trace.event("jira_result", chunks=len(jira_chunks))
            if jira_chunks:
                trace.decision("source_selection", choice="jira", reason="Jira returned data")
                result = {'source': 'jira', 'chunks': jira_chunks,
                          'score': 0.7, 'trace': f'ForceJira→Jira',
                          '_trace': trace.json()}
                _cache_set(ck, result, dcd=dcd_result)
                return result
        # Jira empty → web
        with trace.stage("web_fallback"):
            web_chunks = await loop.run_in_executor(_EXECUTOR, _blocking_web, query, domain, collection)
            trace.event("web_result", chunks=len(web_chunks))
            if web_chunks:
                result = {'source': 'web', 'chunks': web_chunks,
                          'score': 0.6, 'trace': 'ForceJira→empty→Web',
                          '_trace': trace.json()}
                _cache_set(ck, result, dcd=dcd_result)
                return result
            result = {'source': 'empty', 'chunks': [], 'score': 0,
                      'trace': 'ForceJira→empty', '_trace': trace.json()}
            _cache_set(ck, result, dcd=dcd_result)
            return result

    # ── PATH A: rusbitech — prefer lodestone/confluence for corporate docs ──
    if domain == 'rusbitech':
        trace.event("parallel_start", sources="lodestone_first")

        # Check query for corporate product indicators
        corporate_keywords = {'ald', 'rupost', 'termidesk', 'workspad', 'alabuga', 'tatneft', 'gazprom', 'novatek', 'rusgidro', 'клиент', 'сервер'}
        query_lower = query.lower()
        is_corporate = any(kw in query_lower for kw in corporate_keywords)
        is_presale = 'presale' in collection or 'presale' in query_lower

        # For corporate queries → parallel slow sources directly, skip zvec
        if is_corporate or is_presale:
            trace.decision("rusbitech_corporate", choice="lodestone+confluence+web",
                           reason="corporate product query detected" if is_corporate else "presale query")
            with trace.stage("slow_sources_parallel"):
                lodestone_task = loop.run_in_executor(_EXECUTOR, _blocking_mcp_single, 'lodestone', query)
                confluence_task = loop.run_in_executor(_EXECUTOR, _blocking_mcp_single, 'confluence', query)
                jira_task = loop.run_in_executor(_EXECUTOR, _blocking_mcp_single, 'jira', query)
                web_task = loop.run_in_executor(_EXECUTOR, _blocking_web, query, domain, collection)

            with trace.stage("gather_slow_sources"):
                lodestone_chunks, confluence_chunks, jira_chunks, web_chunks = await asyncio.gather(
                    lodestone_task, confluence_task, jira_task, web_task)

            trace.event("lodestone_result", chunks=len(lodestone_chunks))
            trace.event("confluence_result", chunks=len(confluence_chunks))

            # Filter out noise chunks (errors, "no results")
            lodestone_chunks = [c for c in lodestone_chunks if not _is_noise_chunk(c.get('text', ''))]
            jira_chunks = [c for c in jira_chunks if not _is_noise_chunk(c.get('text', ''))]
            confluence_chunks = [c for c in confluence_chunks if not _is_noise_chunk(c.get('text', ''))]

            all_sources = {
                'lodestone': lodestone_chunks,
                'jira': jira_chunks,
                'confluence': confluence_chunks,
                'web': web_chunks,
            }
            # For presale queries, prefer Jira over lodestone
            if is_presale:
                primary = next((s for s, c in [('jira', jira_chunks), ('lodestone', lodestone_chunks),
                                                ('confluence', confluence_chunks), ('web', web_chunks)] if c), 'empty')
            else:
                primary = next((s for s, c in all_sources.items() if c), 'empty')
            trace.decision("source_selection", choice=primary,
                           reason="corporate path - main model decides")
            result = {
                'source': primary,
                'sources': all_sources,
                'chunks': all_sources.get(primary, []),
                'score': 0.7,
                '_trace': trace.json()}
            _cache_set(ck, result, dcd=dcd_result)
            return result

        # Non-corporate rusbitech → zvec first (infra topics)
        trace.event("rusbitech_noncorp", note="using zvec-first for infra/security queries")

        # 1. Fast ZVec search first
        with trace.stage("zvec_search_fast"):
            zvec_result = await loop.run_in_executor(_EXECUTOR, _blocking_zvec, query)

        zvec_chunks = zvec_result['chunks']
        max_score = zvec_result['max_score']
        trace.event("zvec_result_fast", chunks=len(zvec_chunks),
                    max_score=round(max_score, 4))

        # If ZVec has good score — return early, skip slow sources
        if max_score >= 0.60:
            trace.decision("source_selection", choice="zvec_early",
                           reason=f"zvec score {max_score:.2f} >= 0.60, skipping slow sources")
            all_sources = {'zvec': zvec_chunks, 'lodestone': [], 'confluence': [], 'web': []}
            primary = 'zvec'
            result = {'source': primary, 'sources': all_sources,
                      'chunks': zvec_chunks, 'score': max_score,
                      'trace': f'ZVec_early({max_score:.2f})',
                      '_trace': trace.json()}
            _cache_set(ck, result, dcd=dcd_result)
            return result

        # Borderline ZVec score (0.45-0.60) → LLM verify before deciding
        if max_score >= 0.45:
            trace.event("zvec_borderline", score=round(max_score, 3),
                        note="running LLM verify before source decision")
            verified = await loop.run_in_executor(_EXECUTOR, _llm_verify, query, zvec_chunks)
            trace.event("llm_verify_result", verified=verified)
            if verified >= 0.3:
                trace.decision("source_selection", choice="zvec_verified",
                               reason=f"zvec score {max_score:.2f} + LLM verify {verified:.2f}")
                all_sources = {'zvec': zvec_chunks, 'lodestone': [], 'confluence': [], 'web': []}
                primary = 'zvec'
                result = {'source': primary, 'sources': all_sources,
                          'chunks': zvec_chunks, 'score': max_score,
                          'trace': f'ZVec_verified({max_score:.2f}/{verified:.2f})',
                          '_trace': trace.json()}
                _cache_set(ck, result, dcd=dcd_result)
                return result
            trace.event("zvec_verify_failed", score=round(max_score, 2), verified=verified)

        # 2. ZVec score low — parallel search slow sources
        trace.event("zvec_low_score", score=round(max_score, 3),
                    note="launching slow sources in parallel")

        with trace.stage("slow_sources_parallel"):
            lodestone_task = loop.run_in_executor(_EXECUTOR, _blocking_mcp_single, 'lodestone', query)
            confluence_task = loop.run_in_executor(_EXECUTOR, _blocking_mcp_single, 'confluence', query)
            web_task = loop.run_in_executor(_EXECUTOR, _blocking_web, query, domain, collection)

        with trace.stage("gather_slow_sources"):
            lodestone_chunks, confluence_chunks, web_chunks = await asyncio.gather(
                lodestone_task, confluence_task, web_task)

        trace.event("lodestone_result", chunks=len(lodestone_chunks))
        trace.event("confluence_result", chunks=len(confluence_chunks))
        trace.event("web_result", chunks=len(web_chunks))

        # Return all sources, main model decides
        all_sources = {
            'lodestone': lodestone_chunks,
            'confluence': confluence_chunks,
            'zvec': zvec_chunks,
            'web': web_chunks,
        }
        primary = next((s for s, c in all_sources.items() if c), 'empty')
        trace.decision("source_selection", choice=primary,
                       reason="all sources returned, main model decides")
        result = {
            'source': primary,
            'sources': all_sources,
            'chunks': all_sources.get(primary, []),
            'score': 0.7,
            '_trace': trace.json()}
        _cache_set(ck, result, dcd=dcd_result)
        return result

    # ── PATH B: devops/software-dev — ZVec → MCP → web ──

    trace.event("parallel_start", sources="zvec+web")

    with trace.stage("zvec_search"):
        zvec_task = loop.run_in_executor(_EXECUTOR, _blocking_zvec, query)
        web_task = loop.run_in_executor(_EXECUTOR, _blocking_web, query, domain, collection)

    with trace.stage("gather_parallel"):
        zvec_result, web_chunks = await asyncio.gather(zvec_task, web_task)

    zvec_chunks = zvec_result['chunks']
    max_score = zvec_result['max_score']
    trace.event("zvec_result", chunks=len(zvec_chunks), max_score=round(max_score, 4))
    trace.event("web_result", chunks=len(web_chunks))

    # Entity match for ZVec
    with trace.stage("entity_match"):
        _zvec_entities_match = await loop.run_in_executor(
            _EXECUTOR, _check_entities_in_query, query, zvec_chunks)
        trace.event("entity_match_result", zvec_match=str(_zvec_entities_match))

    # Entity mismatch → MCP/Web
    if _zvec_entities_match is False:
        trace.decision("source_selection", choice="mcp_fallback", reason="zvec entity mismatch")
        with trace.stage("mcp_fallback"):
            result = await _fallback_to_mcp_web(query, domain, collection, loop, trace)
            if result['chunks']:
                result['_trace'] = trace.json()
                _cache_set(ck, result, dcd=dcd_result)
                return result
        if web_chunks:
            trace.decision("source_selection", choice="web", reason="mcp empty")
            result = {'source': 'web', 'chunks': web_chunks, 'score': 0.6,
                      'trace': f'ZVec→EntityMismatch→Web', '_trace': trace.json()}
            _cache_set(ck, result, dcd=dcd_result)
            return result

    # ZVec score >= threshold → return (no LLM gate)
    if max_score >= LLM_EVAL_HIGH_THRESHOLD and _zvec_entities_match is not False:
        trace.decision("source_selection", choice="zvec",
                       reason=f"zvec score {max_score:.2f} >= threshold (no LLM gate)")
        result = {'source': 'zvec', 'chunks': zvec_chunks,
                  'score': max_score, 'trace': f'ZVec({max_score:.2f})',
                  '_trace': trace.json()}
        _cache_set(ck, result, dcd=dcd_result)
        return result

    # Low DCD confidence → MCP/Web
    if confidence < 0.20:
        trace.decision("source_selection", choice="mcp_fallback",
                       reason=f"dcd confidence {confidence:.2f} < 0.2")
        with trace.stage("mcp_fallback_lowconf"):
            result = await _fallback_to_mcp_web(query, domain, collection, loop, trace)
            if result['chunks']:
                result['_trace'] = trace.json()
                _cache_set(ck, result, dcd=dcd_result)
                return result
        if web_chunks:
            result = {'source': 'web', 'chunks': web_chunks, 'score': 0.6,
                      'trace': f'DCD(conf={confidence:.2f}<0.2)→Web', '_trace': trace.json()}
            _cache_set(ck, result, dcd=dcd_result)
            return result
        result = {'source': 'empty', 'chunks': [], 'score': 0,
                  'trace': f'DCD(conf={confidence:.2f}<0.2)→empty', '_trace': trace.json()}
        _cache_set(ck, result, dcd=dcd_result)
        return result

    # Final MCP fallback
    with trace.stage("final_mcp"):
        result = await _fallback_to_mcp_web(query, domain, collection, loop, trace)
        if result['chunks']:
            result['_trace'] = trace.json()
            _cache_set(ck, result, dcd=dcd_result)
            return result

    if web_chunks:
        trace.decision("source_selection", choice="web", reason="all MCP empty, final web")
        result = {'source': 'web', 'chunks': web_chunks, 'score': 0.6,
                  'trace': 'ZVec→MCP→Web', '_trace': trace.json()}
        _cache_set(ck, result, dcd=dcd_result)
        return result

    trace.decision("source_selection", choice="empty", reason="all sources returned empty")
    result = {'source': 'empty', 'chunks': [], 'score': 0,
              'trace': 'ZVec→MCP→Web→empty', '_trace': trace.json()}
    _cache_set(ck, result, dcd=dcd_result)
    return result


# ── CLI ──

def main():
    import asyncio
    if len(sys.argv) < 2:
        print("Usage: python -m rag_async <query>")
        return

    query = " ".join(sys.argv[1:])
    result = asyncio.run(async_rag_search(query, classify(query)))

    print(f"\nSource: {result.get('source', 'empty')}")
    print(f"DCD: {result.get('dcd_domain', '')}/{result.get('dcd_collection', '')}")
    print(f"Chunks: {len(result.get('chunks', []))}")
    print(f"Trace: {result.get('trace', '?')}")

    if 'sources' in result:
        for src, chunks in result['sources'].items():
            if chunks:
                print(f"\n{src} ({len(chunks)} chunks):")
                for c in chunks[:2]:
                    txt = c.get('text', '')[:200]
                    print(f"  {txt}")


if __name__ == "__main__":
    main()