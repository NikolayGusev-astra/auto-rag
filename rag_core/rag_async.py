"""
Async RAG pipeline — parallel ZVec, MCP, SearXNG via asyncio.gather().
Adapted for Autolycus: curl for embeddings, qwen3-4b, no lodestone/jira.
"""
import asyncio, hashlib, json, os, sys, time, re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_config import *
from rag_mcp_client import MCPClient

_EXECUTOR = ThreadPoolExecutor(max_workers=6)

# ── LRU cache: 100 last queries ──────────────────────────────────
_CACHE = OrderedDict()
_CACHE_MAX = 100

def _cache_key(query: str, domain: str) -> str:
    return hashlib.md5(f"{query}|{domain}".encode()).hexdigest()

def _cache_get(key: str):
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    return None

def _cache_set(key: str, result: dict):
    _CACHE[key] = result
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)

# ── Thresholds from rag_config ───────────────────────────────────
LLM_EVAL_HIGH_THRESHOLD = COSINE_THRESHOLDS.get("factual", 0.6)
LLM_EVAL_LOW_THRESHOLD = COSINE_THRESHOLDS.get("default", 0.4)

# ── Embedding via curl (requests fails on localhost:1234) ─────────
def _embed(text: str) -> list[float]:
    import subprocess as _sp
    try:
        payload = json.dumps({"model": EMBEDDING_MODEL, "input": [text]})
        r = _sp.run(
            ["curl", "-s", "--max-time", "10", EMBEDDING_URL, "-d", payload, "-H", "Content-Type: application/json"],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode == 0 and r.stdout:
            return json.loads(r.stdout)["data"][0]["embedding"]
    except: pass
    return [0.0] * EMBEDDING_DIM

# ── Blocking helpers (thread pool) ───────────────────────────────
def _blocking_zvec(query: str) -> dict:
    emb = _embed(query)
    import zvec
    from zvec import Query as ZQ
    from rag_config import ensure_zvec_lock
    zpath = os.path.join(ZVEC_PATH, ZVEC_WIKI_COLLECTION)
    ensure_zvec_lock(zpath)
    coll = zvec.open(zpath)
    doclist = coll.query(queries=[ZQ(field_name="embedding", vector=emb)], topk=5,
                         output_fields=["source", "heading", "category", "node", "content", "title"])
    chunks = []
    for d in doclist:
        txt = (d.fields or {}).get("content", "") or (d.fields or {}).get("text", "")
        if txt:
            chunks.append({"text": txt[:500], "score": d.score,
                          "source": (d.fields or {}).get("source", "zvec/wiki")})
    return {"chunks": chunks, "max_score": max((c["score"] for c in chunks), default=0)}

def _blocking_mcp_single(name: str, query: str) -> list[dict]:
    cfg = MCP_SERVERS.get(name)
    if not cfg: return []
    mc = MCPClient(timeout=15)
    return mc.query(name, cfg, query, 3)

def _blocking_mcp(query: str, domain: str, collection: str) -> dict:
    dm = DCD_COLLECTION_MCP_MAP.get(domain, {})
    primary = dm.get(collection) or dm.get("*")
    sources = []
    if primary and primary in MCP_SERVERS:
        sources.append(primary)
    all_results = {}
    for src in sources:
        chunks = _blocking_mcp_single(src, query)
        if chunks: all_results[src] = chunks
    if all_results:
        best_src = max(all_results, key=lambda s: len(all_results[s]))
        return {"source": best_src, "chunks": all_results[best_src]}
    return {"source": None, "chunks": []}

def _blocking_web(query: str) -> list[dict]:
    import subprocess as _sp, urllib.parse
    encoded = urllib.parse.quote(query)
    try:
        r = _sp.run(["curl", "-s", "--max-time", "10", f"{SEARXNG_URL}/search?q={encoded}&format=json"],
                    capture_output=True, text=True, timeout=15)
        if r.returncode != 0 or not r.stdout: return []
        data = json.loads(r.stdout)
        results = data.get("results", [])[:WEB_SEARCH_MAX_RESULTS]
        chunks = []
        for wr in results:
            text = wr.get("content", "") or wr.get("snippet", "")
            if text:
                chunks.append({"text": text[:WEB_SEARCH_MAX_CHARS],
                              "title": wr.get("title", ""), "url": wr.get("url", ""),
                              "source": "web/searxng"})
        return chunks
    except: pass
    return []

def _blocking_llm_eval(query: str, chunks: list) -> float:
    import subprocess as _sp, json, re
    top = "\n\n".join([f'[{i}] {c["text"][:300].replace(chr(10)," ")}' for i, c in enumerate(chunks[:3])])
    prompt = f"Rate relevance 0.0-1.0. Reply ONLY a number.\nQuery: {query[:200]}\nDocuments:\n{top}"
    try:
        payload = json.dumps({
            "model": LLM_REWRITE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": 20,
        })
        r = _sp.run(["curl", "-s", "--max-time", "15", LM_STUDIO_CHAT_URL, "-d", payload,
                     "-H", "Content-Type: application/json"], capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout:
            content = json.loads(r.stdout)["choices"][0]["message"]["content"].strip()
            nums = re.findall(r"0\.\d+|1\.0", content)
            if nums: return float(nums[0])
    except: pass
    return 0.0

# ── Entity Match ──────────────────────────────────────────────────
_ENTITY_CACHE = {}

def _extract_entities(query: str) -> set[str]:
    key = hashlib.md5(query.encode()).hexdigest()[:16]
    if key in _ENTITY_CACHE: return _ENTITY_CACHE[key]
    entities = set()
    ql = query.lower()
    # URLs
    urls = re.findall(r"https?://[^\s]+", query)
    entities.update(urls)
    for u in urls:
        domain = re.findall(r"://([^/]+)", u)
        if domain:
            entities.add(domain[0])
            parts = domain[0].split(".")
            if len(parts) >= 2: entities.add(parts[-2])
            if len(parts) >= 3: entities.add(parts[0])
    # Products & technologies (Autolycus)
    known = [
        "terraform", "ansible", "docker", "kubernetes", "postgresql",
        "postgres", "redis", "nginx", "xray", "openvpn", "systemd",
        "letsencrypt", "certbot", "gitlab", "prometheus", "grafana",
        "adr", "dcd", "crag", "zvec", "rag", "llm-wiki", "embedding", "hnsw",
        "astra", "astralinux", "qwen", "gemma", "bge-m3",
        "ford", "explorer", "шрус", "vin", "oem", "партномер", "carpc",
    ]
    for prod in known:
        if prod in ql: entities.add(prod)
    # OEM part numbers (F5TZ-3A427A)
    for m in re.finditer(r"\b[A-Z0-9]{3,5}-\d{3,6}[A-Z]?\b", query.upper()):
        entities.add(m.group())
    # Capitalized words (product names, ADR numbers)
    caps = re.findall(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b", query)
    for c in caps:
        low = c.lower()
        if low not in ("the","this","that","what","how","why","not","for","and","with","from","into","about"):
            entities.add(low)
    _ENTITY_CACHE[key] = entities
    return entities

def _check_entities_in_query(query: str, chunks: list[dict]) -> bool | None:
    entities = _extract_entities(query)
    tech = {e for e in entities if not e.startswith("http") and len(e) > 2}
    if not entities and not tech: return None
    chunk_text = " ".join(c.get("text", "") for c in chunks).lower()
    missing = [ent for ent in entities if not ent.startswith("http") and ent not in chunk_text]
    checked = [e for e in entities if not e.startswith("http") and len(e) > 3]
    if not checked: return None
    missing_ratio = len(missing) / max(len(checked), 1)
    return False if missing_ratio >= 0.5 else True

# ── Main async pipeline ───────────────────────────────────────────
async def async_rag_search(query: str, dcd_result: dict) -> dict:
    domain = dcd_result.get("domain", "")
    collection = dcd_result.get("collection", "")
    confidence = dcd_result.get("confidence", 0)
    ck = _cache_key(query, domain)

    cached = _cache_get(ck)
    if cached: return cached

    loop = asyncio.get_running_loop()

    # Parallel ZVec + Web
    zvec_task = loop.run_in_executor(_EXECUTOR, _blocking_zvec, query)
    web_task = loop.run_in_executor(_EXECUTOR, _blocking_web, query)

    zvec_result, web_chunks = await asyncio.gather(zvec_task, web_task)
    zvec_chunks = zvec_result["chunks"]
    max_score = zvec_result["max_score"]

    # Entity Match
    entities_ok = await loop.run_in_executor(_EXECUTOR, _check_entities_in_query, query, zvec_chunks)

    if entities_ok is False:
        # ZVec false positive → MCP/web
        result = await _fallback_to_mcp_web(query, domain, collection, loop)
        if result["chunks"]:
            _cache_set(ck, result); return result
        if web_chunks:
            result = {"source": "web", "chunks": web_chunks, "score": 0.6,
                      "trace": f"ZVec({max_score:.2f})→EntityMismatch→Web"}
            _cache_set(ck, result); return result
    elif entities_ok is True:
        if max_score >= LLM_EVAL_HIGH_THRESHOLD:
            result = {"source": "zvec", "chunks": zvec_chunks, "score": max_score,
                      "trace": f"ZVec(score={max_score:.2f}✓entities)"}
            _cache_set(ck, result); return result

    # Low DCD confidence → MCP/web
    if confidence < 0.20:
        result = await _fallback_to_mcp_web(query, domain, collection, loop)
        if result["chunks"]:
            _cache_set(ck, result); return result
        if web_chunks:
            result = {"source": "web", "chunks": web_chunks, "score": 0.6,
                      "trace": f"DCD(conf={confidence:.2f}<0.2)→Web"}
            _cache_set(ck, result); return result
        result = {"source": "empty", "chunks": zvec_chunks[:1], "score": max_score,
                  "trace": f"DCD(conf={confidence:.2f}<0.2)→empty"}
        _cache_set(ck, result); return result

    # Tiered eval
    if max_score >= LLM_EVAL_HIGH_THRESHOLD:
        result = {"source": "zvec", "chunks": zvec_chunks, "score": max_score,
                  "trace": f"ZVec(score={max_score:.2f}≥{LLM_EVAL_HIGH_THRESHOLD})"}
        _cache_set(ck, result); return result

    if max_score >= LLM_EVAL_LOW_THRESHOLD:
        llm_score = await loop.run_in_executor(_EXECUTOR, _blocking_llm_eval, query, zvec_chunks)
        if llm_score >= 0.5:
            result = {"source": "zvec+llm", "chunks": zvec_chunks, "score": llm_score,
                      "trace": f"ZVec({max_score:.2f})→Qwen({llm_score:.2f})"}
            _cache_set(ck, result); return result

    # ZVec not confident → MCP → web
    result = await _fallback_to_mcp_web(query, domain, collection, loop)
    if result["chunks"]:
        _cache_set(ck, result); return result
    if web_chunks:
        result = {"source": "web", "chunks": web_chunks, "score": 0.6,
                  "trace": "ZVec→MCP→empty→Web"}
        _cache_set(ck, result); return result

    result = {"source": "empty", "chunks": zvec_chunks[:1], "score": max_score,
              "trace": "ZVec→MCP→Web→empty"}
    _cache_set(ck, result); return result

async def _fallback_to_mcp_web(query: str, domain: str, collection: str, loop) -> dict:
    dm = DCD_COLLECTION_MCP_MAP.get(domain, {})
    primary = dm.get(collection) or dm.get("*")
    sources = []
    if primary and primary in MCP_SERVERS:
        sources.append(primary)
    if not sources:
        return {"source": None, "chunks": []}

    tasks = [loop.run_in_executor(_EXECUTOR, _blocking_mcp_single, s, query) for s in sources]
    all_results = await asyncio.gather(*tasks)

    best_chunks, best_src = [], None
    for src, chunks in zip(sources, all_results):
        if chunks and len(chunks) > len(best_chunks):
            best_chunks, best_src = chunks, src

    if best_chunks:
        return {"source": best_src, "chunks": best_chunks, "score": 0.7,
                "trace": f"MCP({best_src})"}
    return {"source": None, "chunks": []}