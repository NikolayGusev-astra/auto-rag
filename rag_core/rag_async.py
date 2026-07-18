"""
Async RAG pipeline — параллельный запуск ZVec, MCP, SearXNG через asyncio.gather().
Включает RagTrace для прозрачного трейсинга каждого этапа.

Архитектура v3:
- ZVec early exit (score >= 0.45) → skip slow sources
- ZVec → MCP → web fallback
- LLM quality gate на quick verify (qwen2.5-7b, ~2.7s)
- Trafilatura для полного текста веб-страниц
- Multi-source fallback: пустой источник → следующий
- Compound queries: альд+postgresql → 2 подзапроса без LLM
- Smart fusion: все чанки из всех источников → main model
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))
from rag_config import (
    EMBEDDING_URL, EMBEDDING_MODEL, EMBEDDING_DIM,
    ZVEC_PATH, ZVEC_COLLECTION,
    MCP_SERVERS, MCP_ENABLED, MCP_MAX_RESULTS,
    LLM_VERIFY_ENABLED, LLM_VERIFY_URL, LLM_VERIFY_MODEL, LLM_VERIFY_TIMEOUT,
    LLM_EVAL_HIGH_THRESHOLD,
    SEARXNG_URL, SEARXNG_ENABLED, WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_MAX_CHARS,
    DCD_PREFERRED_WEB_SOURCE,
    LOCAL_NODE_NAME,
)

from dcd_router import classify
from rag_mcp_client import MCPClient
from rag_trace import RagTrace


# ── SSRF guard ───────────────────────────────────────────────
# Delegates to rag_core.security.safe_http (TOCTOU-resistant: resolve once,
# connect by validated IP, send original Host). Kept here as thin wrappers so
# existing call sites (_blocking_web) are unchanged.
from rag_core.security.safe_http import (
    url_targets_public as _is_safe_url,
    safe_get as _safe_get_impl,
)


def _safe_get(url: str, **kwargs) -> "object | None":
    """SSRF-hardened GET. See rag_core.security.safe_http.safe_get."""
    return _safe_get_impl(url, **kwargs)

# ── Optional memvid memory layer (T3 integration) ─────────────────
try:
    from memvid_memory import Episode, MemvidMemory
    from memvid_trace import MemvidTraced
    _MEMVID_AVAILABLE = True
except Exception:
    _MEMVID_AVAILABLE = False

_memory = None

def _get_memory():
    """Lazy singleton for the memvid memory layer.

    Returns a MemvidTraced wrapper, or None when the layer is disabled
    (RAG_MEMVID_ENABLED != true) or import failed. Never raises.
    """
    global _memory
    if _memory is None and _MEMVID_AVAILABLE:
        try:
            from memvid_config_bridge import bridge_memvid_env
            bridge_memvid_env()
            _memory = MemvidTraced(MemvidMemory.for_tenant(
                os.environ.get("RAG_MEMVID_TENANT", "hermes_default")))
        except Exception:
            _memory = None
    return _memory

def _episode_answer(result: dict) -> str:
    """Снимок полезного контента RAG для episodic memory.

    Core-пайплайн возвращает ранжированные chunks, а не готовый LLM-answer.
    Сохраняем их текст, а не технический trace, чтобы recall возвращал
    пользователю знания, а не журнал маршрутизации.
    """
    if result.get("answer"):
        return str(result["answer"])
    parts = []
    for chunk in result.get("chunks", []):
        text = chunk.get("text") or chunk.get("content") or ""
        if text:
            parts.append(str(text))
    return "\n\n".join(parts)[:6000]


def _record_episode(result: dict, query: str, domain: str, trace: RagTrace) -> None:
    """Best-effort record of a RAG result as a memvid episode (guarded).

    Delegates to rag_core.memory.episode_writer, which enforces the
    poisoning guard: web/federation-only results are NOT recorded, and the
    tenant is taken explicitly rather than silently from env.
    """
    if not _MEMVID_AVAILABLE:
        return
    mem = _get_memory()
    if mem is None or not mem.active:
        return
    from rag_core.memory.episode_writer import build_episode
    tenant = os.environ.get("RAG_MEMVID_TENANT", "hermes_default")
    index_rev = os.environ.get("RAG_INDEX_REVISION", "unknown")
    try:
        ep = build_episode(result, query, domain, tenant, index_rev, trace)
        if ep is None:
            return
        mem.record(ep, trace=trace)
    except Exception as exc:
        logger.debug("memvid record failed: %s", exc)

_AsyncMCPClient = None  # lazy import to avoid circular deps

_EXECUTOR = None  # lazily bound to default_runtime() to allow DI / test isolation


def _executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        from rag_core.runtime import default_runtime
        _EXECUTOR = default_runtime().executor
    return _EXECUTOR

# ── Compound query detection ──
# Составные запросы: продукт Astra (ALD/РуПост/...) + инфраструктурная штука.
# "альд postgresql репликация" → и продукт, и инфра. Дробим на 2 подзапроса
# без LLM, гоним параллельно и сливаем чанки через smart fusion.
# (keyword tables live in rag_core.compound)

def _detect_compound(query: str, dcd: dict) -> list[dict]:
    """Detect compound queries (delegates to rag_core.compound)."""
    from rag_core.compound import detect_compound
    return detect_compound(query, dcd)


# ── Routing log ──
_ROUTING_LOG = os.path.join(os.path.dirname(__file__), "routing_log.jsonl")

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
    except Exception as exc:
        logger.debug("routing log write failed: %s", exc)


# ── LRU cache: 100 last queries (backed by RagRuntime for DI/test isolation) ──
def _rt():
    from rag_core.runtime import default_runtime
    return default_runtime()


def _cache_get(key: str) -> dict | None:
    return _rt().cache_get(key)

def _cache_set(key: str, result: dict, dcd: dict | None = None):
    _rt().cache_set(key, result)
    # Log routing
    d = dcd
    if d is not None:
        query = ""
        trace_val = result.get("_trace", "")
        if isinstance(trace_val, str) and trace_val.startswith("{"):
            try:
                td = json.loads(trace_val)
                query = td.get("query", "")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.debug("cached trace was not valid JSON: %s", exc)
        _log_routing(query[:200] if query else key[:30], d, result)


# ── Embedding ────────────────────────────────────────────────────
def _embed(text: str) -> list[float]:
    """Embedding через EmbeddingService."""
    from embedding_service import get_embedding
    return get_embedding(text)


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
        except OSError as exc:
            logger.debug("could not create ZVec lock file: %s", exc)
        _ZVEC_COLLECTION = zvec.open(zpath)
        return _ZVEC_COLLECTION


# ── Blocking helpers (thread pool) ───────────────────────────────
def _blocking_zvec(query: str) -> dict:
    """ZVec search — prefers FastAPI server, falls back to direct ZVec."""
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
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("ZVec FastAPI unavailable: %s", exc)

    # Fallback to direct ZVec
    try:
        import zvec
        from zvec import Query as ZQ
        emb = _embed(query)
        zpath = os.path.join(ZVEC_PATH, ZVEC_COLLECTION)
        coll = zvec.open(zpath)
        doclist = coll.query(
            queries=[ZQ(field_name="embedding", vector=emb)],
            topk=5,
            filter="category = 'wiki' OR category = 'llm-wiki'",
            output_fields=["source", "heading", "content", "title", "category"],
        )
        chunks = []
        for d in doclist:
            txt = d.fields.get("text", "") or d.fields.get("content", "")
            if txt:
                chunks.append({"text": txt, "score": d.score, "source": "zvec/wiki"})
        if chunks:
            return {"chunks": chunks, "max_score": max(c["score"] for c in chunks), "source": "zvec_direct"}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.debug("direct ZVec search unavailable: %s", exc)
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
    preferred = DCD_PREFERRED_WEB_SOURCE.get(domain, {}).get(collection) or \
                DCD_PREFERRED_WEB_SOURCE.get(domain, {}).get("*")
    if preferred == "skip":
        return []
    encoded = urllib.parse.quote(query)
    try:
        r = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])[:WEB_SEARCH_MAX_RESULTS]
        chunks = []
        for wr in results:
            text = wr.get("content", "") or wr.get("snippet", "")
            url = wr.get("url", "")
            if text and len(chunks) < 1:
                # SSRF guard: не фетчим приватные/локальные URL
                if not _is_safe_url(url):
                    chunks.append({"text": text[:800], "source": "web", "url": url})
                    continue
                try:
                    req = _safe_get(url, headers={
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                    })
                    if req is None:
                        continue
                    html = req.text
                    full = trafilatura.extract(html)
                    if full:
                        text = full[:2000]
                except (requests.RequestException, ValueError, TypeError) as exc:
                    logger.debug("web page enrichment failed for %s: %s", url, exc)
            if text:
                chunks.append({"text": text[:800], "source": "web", "url": url})
        return chunks
    except Exception:
        return []
    return []


# ── LLM Verify (fast) ────────────────────────────────────────────
_LLM_VERIFY_CACHE: dict[str, tuple[float, float]] = {}
_LLM_VERIFY_CACHE_MAX = 256
_LLM_VERIFY_CACHE_TTL = 120.0


def _llm_verify_cache_key(query: str, chunks: list[dict]) -> str:
    texts = "||".join((c.get("text") or "")[:180] for c in chunks[:3])
    return hashlib.md5(f"{query}||{texts}".encode()).hexdigest()


def _llm_verify(query: str, chunks: list[dict]) -> float:
    """Soft verification: returns 0.0-1.0 relevance score.

    Back-compat wrapper. Internally uses rag_core.verification which is
    fail-closed: a verifier failure yields UNAVAILABLE and this wrapper maps
    it to 0.0 (irrelevant) rather than the old 0.5 fail-open default.
    """
    from rag_core.verification import verify_relevance
    from rag_config import (
        LLM_VERIFY_ENABLED, LLM_VERIFY_URL, LLM_VERIFY_MODEL, LLM_VERIFY_TIMEOUT,
    )
    res = verify_relevance(
        query, chunks,
        enabled=LLM_VERIFY_ENABLED,
        url=LLM_VERIFY_URL,
        model=LLM_VERIFY_MODEL,
        timeout=LLM_VERIFY_TIMEOUT,
    )
    if res.status in (res.status.RELEVANT, res.status.IRRELEVANT):
        return res.score or 0.0
    # UNAVAILABLE / INVALID -> treat as "not verified relevant"
    return 0.0


# ── LLM Eval (for borderline scores) ─────────────────────────────
_ENTITY_EXTRACTOR_CACHE: dict[str, set[str]] = {}


def _extract_entities(query: str) -> set[str]:
    """Extract tech entities from query for chunk relevance check."""
    cache_key = query[:200]
    if cache_key in _ENTITY_EXTRACTOR_CACHE:
        return _ENTITY_EXTRACTOR_CACHE[cache_key]

    entities: set[str] = set()
    # English tech terms: camelCase, snake_case, CAPS acronyms
    entities.update(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', query))  # CamelCase
    entities.update(re.findall(r'\b[a-z]+_[a-z_]+\b', query))              # snake_case
    entities.update(re.findall(r'\b[A-Z]{2,}\b', query))                   # ACRONYMs
    # Version numbers / product codes
    entities.update(re.findall(r'\b\d+\.\d+(?:\.\d+)?\b', query))         # 1.2.3
    # Russian tech terms
    entities.update(re.findall(r'\b[А-Яа-яё]{3,}\b', query))
    # Filter stop words
    stop = {'the', 'and', 'for', 'not', 'how', 'why', 'what', 'this', 'that',
            'это', 'что', 'как', 'все', 'если', 'котор', 'можно', 'тольк',
            'есть', 'при', 'ваш', 'меня', 'быть', 'когда', 'после', 'через',
            'ком', 'как', 'podcast', 'для', 'или', 'ещё', 'уже', 'где', 'как',
            'так', 'вот', 'тут', 'там', 'его', 'её', 'их', 'мой', 'твой',
            'свой', 'кто', 'что', 'где', 'когда', 'почему', 'зачем', 'сколько'}
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
    for name in ['context7', 'jira', 'confluence', 'lodestone', 'protopack']:
        if name in MCP_SERVERS:
            with trace.stage(f"mcp_{name}"):
                chunks = await loop.run_in_executor(_executor(), _blocking_mcp_single, name, query)
                trace.event("mcp_result", source=name, chunks=len(chunks))
            if chunks:
                trace.decision("mcp_selected", choice=name,
                               reason=f"MCP {name} returned {len(chunks)} chunks")
                return {'source': name, 'chunks': chunks, 'score': 0.7}
    # Web fallback
    web_chunks = await loop.run_in_executor(_executor(), _blocking_web, query, domain, collection)
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
async def _async_rag_search_impl(
    query: str, dcd_result: dict,
    trace: RagTrace | None = None,
    federate: bool = True,
    max_results: int = 5,
) -> dict:
    domain = dcd_result.get('domain', '')
    collection = dcd_result.get('collection', '')
    confidence = dcd_result.get('confidence', 0)
    tenant_id = dcd_result.get('tenant_id', os.environ.get("RAG_TENANT_ID", "default"))
    acl_hash = dcd_result.get('acl_hash', os.environ.get("RAG_ACL_HASH", "none"))
    # Single cache-key source of truth: QueryContext (includes tenant/ACL/
    # collection/index-rev). Replaces the split _cache_key() helper.
    from rag_core.query_context import QueryContext
    ctx = QueryContext(
        query=query, domain=domain, collection=collection,
        max_results=max_results, tenant_id=tenant_id, principal_acl_hash=acl_hash,
    )
    ck = ctx.cache_key()
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

    chunks = []  # populated by federation fallback if all other sources empty

    # ── Generic RAG pipeline: ZVec → MCP → web ──
    # Web is started speculatively ONLY when RAG_WEB_SPECULATIVE=1 (opt-in).
    # By default web runs only after local sources prove insufficient
    # (local-first semantics; avoids leaking query to SearXNG for purely
    # local questions). See audit P1.
    _speculative_web = os.environ.get("RAG_WEB_SPECULATIVE", "0") == "1"

    trace.event("parallel_start",
                sources="zvec+web" if _speculative_web else "zvec")

    with trace.stage("zvec_search"):
        zvec_task = loop.run_in_executor(_executor(), _blocking_zvec, query)
        web_task = (loop.run_in_executor(_executor(), _blocking_web, query, domain, collection)
                    if _speculative_web else None)

    with trace.stage("gather_parallel"):
        if _speculative_web:
            zvec_result, web_chunks = await asyncio.gather(zvec_task, web_task)
        else:
            zvec_result = await zvec_task
            web_chunks = []

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

    # ── Federation fallback: опрос других RAG-инстансов ──
    if federate and not chunks and os.getenv("RAG_FEDERATED_ENABLED", "false").lower() == "true":
        with trace.stage("federation"):
            try:
                from rag_federated import query_federated_servers
                fed_results = await query_federated_servers(query, max_results=max_results, domain=domain)
            except Exception as fed_err:
                trace.error("federation", str(fed_err))
                fed_results = {}

        pool = []
        for server_name, server_chunks in fed_results.items():
            for c in server_chunks:
                text = c.get("text", "")
                score = c.get("score", 0)
                if text and score > 0 and not c.get("is_error"):
                    pool.append({
                        "text": text[:800],
                        "score": float(score),
                        "source": f"federated:{server_name}",
                        "_src": f"federated:{server_name}",
                    })

        seen = set()
        deduped = []
        for c in pool:
            key = c["text"][:200]
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        if deduped:
            deduped.sort(key=lambda x: x["score"], reverse=True)
            chunks = deduped[:max_results]
            fallback_used = "federated"
            trace.event("federation_result",
                        servers=len(fed_results),
                        chunks_merged=len(deduped))
            trace.decision("source_selection", choice="federated",
                           reason=f"federation returned {len(chunks)} chunks from {len(fed_results)} servers")
            result = {'source': 'federated', 'chunks': chunks,
                      'score': chunks[0]["score"] if chunks else 0,
                      'trace': 'ZVec→MCP→Web→Federated',
                      '_trace': trace.json()}
            _cache_set(ck, result, dcd=dcd_result)
            return result
        else:
            trace.event("federation_empty", servers=len(fed_results))

    trace.decision("source_selection", choice="empty", reason="all sources returned empty")
    result = {'source': 'empty', 'chunks': [], 'score': 0,
              'trace': 'ZVec→MCP→Web→empty', '_trace': trace.json()}
    _cache_set(ck, result, dcd=dcd_result)
    return result


# ── Public entry with memvid memory layer (T3) ───────────────────
async def async_rag_search(
    query: str, dcd_result: dict,
    trace: RagTrace | None = None,
    federate: bool = True,
    max_results: int = 5,
) -> dict:
    """Public RAG entrypoint with optional memvid episodic memory.

    Flow:
      1. recall prior episodes (if memory enabled) -> short-circuit on hit
      2. run the core pipeline (_async_rag_search_impl)
      3. record the new episode (if memory enabled, not a memory hit)
    Memory is fully opt-in (RAG_MEMVID_ENABLED) and never breaks RAG.
    """
    if trace is None:
        domain = dcd_result.get('domain', '')
        collection = dcd_result.get('collection', '')
        trace = RagTrace(query, domain, collection)

    # 1) recall (short-circuit) — store result, don't early-return
    result = None
    if _MEMVID_AVAILABLE:
        mem = _get_memory()
        if mem is not None and mem.active:
            try:
                priors = mem.recall(query, domain=trace.domain, trace=trace)
                if priors and priors[0].score >= mem.recall_threshold:
                    ep = priors[0]
                    # C6 fix: include synthetic chunk so eval/canary/CLI see
                    # consistent structure with normal pipeline (has 'chunks')
                    chunks = [{
                        "text": ep.answer or "",
                        "source": ep.sources[0].get("source", "") if ep.sources else "memory",
                        "score": ep.score,
                    }]
                    result = {
                        "answer": ep.answer,
                        "sources": ep.sources,
                        "chunks": chunks,
                        "trace": f"memvid.recall(short-circuit, score={ep.score:.3f})",
                        "from_memory": True,
                        "_trace": trace.json(),
                    }
            except Exception as exc:
                logger.debug("memvid recall failed: %s", exc)

    # 1.5) compound query split (product + infra) → parallel subqueries
    subqueries = _detect_compound(query, dcd_result)
    if subqueries:
        trace.event("compound_detected", subqueries=subqueries)
        results = await asyncio.gather(*[
            _async_rag_search_impl(
                sq["query"],
                {"domain": sq["domain"], "collection": sq["collection"],
                 "confidence": dcd_result.get("confidence", 0), "fallback": False},
                trace=RagTrace(sq["query"], sq["domain"], sq["collection"]),
                federate=federate,
                max_results=max_results,
            )
            for sq in subqueries
        ])
        fused_chunks = []
        sources_used = {}
        for r in results:
            for c in r.get("chunks", []):
                fused_chunks.append(c)
            src = r.get("source", "?")
            sources_used.setdefault(src, []).extend(r.get("chunks", []))
        if fused_chunks:
            fused_chunks = fused_chunks[:max_results]
            fused = {
                "source": "compound",
                "chunks": fused_chunks,
                "score": max((r.get("score", 0) for r in results), default=0),
                "trace": f"Compound({'+'.join(s['domain'] for s in subqueries)})",
                "_trace": trace.json(),
                "sources_used": list(sources_used.keys()),
            }
            _cache_set(ctx.cache_key(), fused, dcd=dcd_result)
            return fused

    # If memory hit, return it
    if result is not None:
        return result

    # 2) core pipeline
    result = await _async_rag_search_impl(query, dcd_result, trace=trace,
                                          federate=federate,
                                          max_results=max_results)

    # 3) record (skip if this was a memory hit)
    if not result.get("from_memory"):
        _record_episode(result, query, trace.domain, trace)

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
