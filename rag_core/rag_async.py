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
import os
import re
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

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

def _record_episode(result: dict, query: str, domain: str, trace: RagTrace) -> None:
    """Best-effort record of a RAG result as a memvid episode. Never raises."""
    if not _MEMVID_AVAILABLE:
        return
    mem = _get_memory()
    if mem is None or not mem.active:
        return
    try:
        mem.record(
            Episode(
                query=query,
                answer=result.get("trace", "") or "",
                sources=[{"source": result.get("source", "")}],
                trace=trace,
                domain=domain,
                tenant=os.environ.get("RAG_MEMVID_TENANT", "hermes_default"),
            ),
            trace=trace,
        )
    except Exception:
        pass

_AsyncMCPClient = None  # lazy import to avoid circular deps

_EXECUTOR = ThreadPoolExecutor(max_workers=6)

# ── Compound query detection ──
# Составные запросы: продукт Astra (ALD/РуПост/...) + инфраструктурная штука.
# "альд postgresql репликация" → и продукт, и инфра. Дробим на 2 подзапроса
# без LLM, гоним параллельно и сливаем чанки через smart fusion.
_COMPOUND_PRODUCT_WORDS = {"ald", "aldpro", "ald pro", "rupost", "termidesk",
                           "workspad", "ddo", "msad", "keycloak", "alse",
                           "astra linux", "astra",
                           # кириллические варианты (русскоязычные запросы)
                           "альд", "альд про", "рупост", "термидеск",
                           "воркспад", "астра линукс", "астра", "кейклок"}
_COMPOUND_INFRA_WORDS = {"postgresql", "postgres", "nginx", "redis", "docker",
                         "kubernetes", "k8s", "patroni", "etcd", "haproxy",
                         "prometheus", "grafana", "rabbitmq", "kafka", "ansible",
                         "terraform", "salt", "saltstack", "sssd", "freeipa",
                         "ipa", "msad", "ad", "active directory", "samba",
                         "kerberos", "hbac", "rbac", "zabbix", "monitoring",
                         "миграц", "доверен", "trust", "dhcp", "dns",
                         "automation", "web оснастк", "web интерфейс",
                         "web консоль",
                         # кириллические варианты инфра-терминов
                         "постгрес", "постгре", "нжинкс", "докер", "кубер",
                         "кубернетес", "патрони", "реплик", "репликация",
                         "бд", "база данных", "резервн", "бэкап"}


def _detect_compound(query: str, dcd: dict) -> list[dict]:
    """Detect compound queries with keywords from multiple domains.

    Returns list of sub-queries, or empty list if not compound.
    Each sub-query: {"query": str, "domain": str, "collection": str}
    """
    ql = query.lower()
    has_product = any(w in ql for w in _COMPOUND_PRODUCT_WORDS)
    has_infra = any(w in ql for w in _COMPOUND_INFRA_WORDS)

    if not (has_product and has_infra):
        return []

    subqueries = []
    # Product part → rusbitech domain
    if has_product:
        subqueries.append({"query": query, "domain": "rusbitech",
                           "collection": "rusbitech-products"})
    # Infra part → devops
    if has_infra:
        infra_terms = [w for w in _COMPOUND_INFRA_WORDS if w in ql]
        infra_query = f"{query} {' '.join(infra_terms[:3])}" if infra_terms else query
        subqueries.append({"query": infra_query, "domain": "devops",
                           "collection": "deployment"})
    return subqueries


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
            except Exception:
                pass
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
        except Exception:
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
        pass  # FastAPI unavailable, try direct ZVec

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
    except Exception:
        pass
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
                try:
                    req = requests.get(url, timeout=8, headers={
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                    })
                    html = req.text
                    full = trafilatura.extract(html)
                    if full:
                        text = full[:2000]
                except Exception:
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
async def _async_rag_search_impl(
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

    chunks = []  # populated by federation fallback if all other sources empty

    # ── Generic RAG pipeline: ZVec → MCP → web ──

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

    # ── Federation fallback: опрос других RAG-инстансов ──
    if not chunks and os.getenv("RAG_FEDERATED_ENABLED", "false").lower() == "true":
        with trace.stage("federation"):
            try:
                from rag_federated import query_federated_servers
                fed_results = await query_federated_servers(query, k, domain=domain)
            except Exception as fed_err:
                trace.error("federation", str(fed_err))
                fed_results = {}

        pool = []
        for server_name, server_chunks in fed_results.items():
            for c in server_chunks:
                text = c.get("text", "")
                score = c.get("score", 0)
                if text and score > 0:
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
            chunks = deduped[:k]
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

    # 1) recall (short-circuit)
    if _MEMVID_AVAILABLE:
        mem = _get_memory()
        if mem is not None and mem.active:
            try:
                priors = mem.recall(query, domain=trace.domain, trace=trace)
                if priors and priors[0].score >= mem.recall_threshold:
                    return {
                        "answer": priors[0].answer,
                        "sources": priors[0].sources,
                        "trace": f"memvid.recall(short-circuit, score={priors[0].score:.3f})",
                        "from_memory": True,
                        "_trace": trace.json(),
                    }
            except Exception:
                pass

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
            )
            for sq in subqueries
        ])
        # Smart fusion: все чанки из всех подзапросов + sources_used
        fused_chunks = []
        sources_used: dict[str, list] = {}
        for r in results:
            for c in r.get("chunks", []):
                fused_chunks.append(c)
            src = r.get("source", "?")
            sources_used.setdefault(src, []).extend(r.get("chunks", []))
        if fused_chunks:
            fused = {
                "source": "compound",
                "chunks": fused_chunks,
                "score": max((r.get("score", 0) for r in results), default=0),
                "trace": f"Compound({'+'.join(s['domain'] for s in subqueries)})",
                "_trace": trace.json(),
                "sources_used": list(sources_used.keys()),
            }
            _cache_set(_cache_key(query, dcd_result.get("domain", "")), fused, dcd=dcd_result)
            return fused

    # 2) core pipeline
    result = await _async_rag_search_impl(query, dcd_result, trace=trace)

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