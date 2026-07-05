#!/usr/bin/env python3
"""
CRAG Search — Autolycus RAG v2.
Architecture: DCD classify → ZVec search → evaluate → MCP fallback → SearXNG fallback
"""
import argparse, json, logging, os, sys, time, hashlib, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_config import (
    EMBEDDING_URL, EMBEDDING_MODEL, EMBEDDING_DIM,
    RERANK_ENABLED, RERANK_MODEL, RERANK_URL,
    SEARXNG_URL, SEARXNG_ENABLED, WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_MAX_CHARS,
    MCP_ENABLED, MCP_SERVERS, MCP_MAX_RESULTS, MCP_FALLBACK_CHAIN,
    COSINE_THRESHOLDS, DEFAULT_K,
    ZVEC_WIKI_COLLECTION, ZVEC_SESSIONS_COLLECTION,
    AMBIGUOUS_RATIO, MIN_RELEVANT_COUNT, LOCAL_NODE_NAME,
)
from dcd_router import classify as dcd_classify
from zvec_adapter import ZVecSearcher
from rag_mcp_client import MCPClient

logger = logging.getLogger(__name__)
_TRAFILATURA_AVAILABLE = False
try:
    import trafilatura; _TRAFILATURA_AVAILABLE = True
except ImportError: pass


# ── Embedding ─────────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    """LM Studio embedding via curl subprocess."""
    import subprocess, json
    try:
        payload = json.dumps({"model": EMBEDDING_MODEL, "input": [text]})
        r = subprocess.run(
            ["curl", "-s", "--max-time", "10", EMBEDDING_URL, "-d", payload, "-H", "Content-Type: application/json"],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode == 0 and r.stdout:
            return json.loads(r.stdout)["data"][0]["embedding"]
    except Exception:
        pass
    return [0.0] * EMBEDDING_DIM


def get_rerank_scores(query: str, chunks: list[dict]) -> list[float]:
    """bge-reranker-v2-m3 via LM Studio (curl subprocess)."""
    if not chunks or not RERANK_ENABLED: return [c.get("score", 0) for c in chunks]
    import subprocess, json
    pairs = [{"query": query, "document": c.get("content", "")[:500]} for c in chunks]
    try:
        payload = json.dumps({"model": RERANK_MODEL, "input": pairs})
        r = subprocess.run(
            ["curl", "-s", "--max-time", "10", RERANK_URL, "-d", payload, "-H", "Content-Type: application/json"],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode == 0 and r.stdout:
            return [d["score"] for d in json.loads(r.stdout)["data"]]
    except Exception:
        pass
    return [c.get("score", 0) for c in chunks]


# ── Entity Match ────────────────────────────────────────────────
import re
_ENTITY_CACHE = {}  # query hash -> set of entities

def extract_entities(query: str) -> set[str]:
    """Extract key entities from query: URLs, products, technologies, versions."""
    q = query.lower()
    # Check cache
    qhash = hashlib.md5(q.encode()).hexdigest()
    if qhash in _ENTITY_CACHE:
        return _ENTITY_CACHE[qhash]
    
    entities = set()
    # URLs
    for m in re.finditer(r'[\w.-]+\.(com|ru|org|net|io|ai|app|tech|cloud)/?\S*', q):
        entities.add(m.group().rstrip('/'))
    # CamelCase / PascalCase / kebab-case tech terms
    for m in re.finditer(r'[A-Z][a-z]+[A-Z]\w+|[a-z]+-[a-z]+(?:\.[a-z]+)*', query):
        entities.add(m.group().lower())
    # Generic tech terms
    for m in re.finditer(r'\b(docker|kubernetes|python|rust|java|elasticsearch|redis|kafka|'
                        r'prometheus|grafana|jenkins|gitlab|github|nginx|postgresql|mysql|mongodb)\b', q):
        entities.add(m.group())
    # RAG-specific terms
    for m in re.finditer(r'\b(embedding|hnsw|fts|vector|rerank|rag|mcp)\b', q):
        entities.add(m.group())
    
    # Limit cache size
    if len(_ENTITY_CACHE) > 1_000_000:
        _ENTITY_CACHE.clear()
    _ENTITY_CACHE[qhash] = entities
    return entities

def entity_match(chunks: list[dict], query: str, threshold: float = 0.5) -> tuple[bool, set[str], set[str]]:
    """Check if ≥50% of query entities appear in at least one chunk.
    Returns: (passes, entities, matched_entities)"""
    entities = extract_entities(query)
    if not entities:
        return True, set(), set()  # no entities to check = pass
    
    matched = set()
    combined_text = ' '.join(c.get("content", c.get("text", "")).lower() for c in chunks)
    for ent in entities:
        if ent in combined_text:
            matched.add(ent)
    
    ratio = len(matched) / len(entities)
    return ratio >= threshold, entities, matched
def searxng_search(query: str, max_results: int = 5) -> list[dict]:
    if not SEARXNG_ENABLED: return []
    import urllib.parse, subprocess
    encoded = urllib.parse.quote(query)
    try:
        r = subprocess.run(["curl", "-s", f"{SEARXNG_URL}/search?q={encoded}&format=json"],
                          capture_output=True, text=True, timeout=15)
        if r.returncode != 0: return []
        data = json.loads(r.stdout)
        results = []
        for item in data.get("results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "source": "searxng",
                "score": item.get("score", 0),
            })
            # Extract full text via Trafilatura
            if _TRAFILATURA_AVAILABLE and len(results) <= 2:
                try:
                    import requests as req
                    resp = req.get(item["url"], timeout=10)
                    text = trafilatura.extract(resp.text)
                    if text: results[-1]["content"] = text[:WEB_SEARCH_MAX_CHARS]
                except Exception:
                    pass
            return results
    except Exception as e:
        logger.warning(f"SearXNG error: {e}")
        return []


# ── MCP Fallback ──────────────────────────────────────────────────
_mcp_client = MCPClient()

def mcp_search(query: str, domain: str = None) -> list[dict]:
    if not MCP_ENABLED: return []
    results = []
    for server in MCP_FALLBACK_CHAIN:
        cfg = MCP_SERVERS.get(server)
        if not cfg: continue
        hits = _mcp_client.query(server, cfg, query, MCP_MAX_RESULTS)
        for h in hits:
            results.append({
                "title": h.get("title", ""),
                "url": h.get("url", ""),
                "content": h.get("content", h.get("text", ""))[:WEB_SEARCH_MAX_CHARS],
                "source": f"mcp:{server}",
                "score": h.get("score", 0.5),
            })
    return results


# ── CRAG Search ──────────────────────────────────────────────────
class CragSearch:
    def __init__(self):
        self.zvec = ZVecSearcher(ZVEC_WIKI_COLLECTION)
        self.zvec_sessions = ZVecSearcher(ZVEC_SESSIONS_COLLECTION)

    def search(self, query: str, k: int = DEFAULT_K, include_sessions: bool = True) -> dict:
        """Full CRAG search: classify → ZVec → evaluate → MCP → web."""

        # Step 1: Classify
        dcd = dcd_classify(query)
        domain = dcd.get("domain", "general")
        confidence = dcd.get("confidence", 0)
        query_type = classify_query_type(query)

        # If DCD confidence < 0.1 — query doesn't match any domain, skip ZVec
        if confidence < 0.10:
            return {
                "context": "",
                "chunks": [],
                "quality": "incorrect",
                "domain": domain,
                "dcd_confidence": confidence,
                "query_type": query_type,
                "fallback_used": None,
                "threshold": 0,
                "max_score": 0,
                "entity_match_ratio": 0,
                "entity_match_pass": None,
                "entity_match_entities": [],
                "entity_match_found": [],
                "node": LOCAL_NODE_NAME,
            }

        # Step 2: ZVec vector search (no domain filter — categories != dcd domains)
        chunks = self.zvec.search(query, topk=k * 2, domain=None)

        # Step 2b: Sessions only if wiki returned nothing (avoid diluting)
        if not chunks and include_sessions:
            session_chunks = self.zvec_sessions.search(query, topk=k)
            chunks.extend(session_chunks)

        # Step 3: Rerank
        if chunks:
            scores = get_rerank_scores(query, chunks)
            for i, c in enumerate(chunks):
                c["score"] = scores[i] if i < len(scores) else c.get("score", 0)
            chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
            chunks = chunks[:k]

        # Step 3b: Entity Match — проверка сущностей из запроса в чанках
        em_pass, em_entities, em_matched = entity_match(chunks, query)
        entity_match_ratio = len(em_matched) / max(len(em_entities), 1)
        if not em_pass and len(em_entities) > 1:
            # ≥50% сущностей не найдены → false positive
            chunks = []

        # Step 4: Evaluate quality
        threshold = COSINE_THRESHOLDS.get(query_type, COSINE_THRESHOLDS["default"])
        max_score = max((c.get("score", 0) for c in chunks), default=0)
        quality = "correct" if max_score >= threshold else \
                  "ambiguous" if max_score >= threshold * AMBIGUOUS_RATIO else "incorrect"

        fallback_used = None
        if quality == "incorrect" and len(chunks) < MIN_RELEVANT_COUNT:
            # Step 5: MCP fallback
            mcp_results = mcp_search(query, domain)
            if mcp_results:
                chunks = mcp_results[:k]
                fallback_used = "mcp"
                quality = "correct"

        if (quality == "incorrect" or quality == "ambiguous") and not chunks:
            # Step 6: Web fallback (SearXNG)
            web_results = searxng_search(query, WEB_SEARCH_MAX_RESULTS)
            if web_results:
                chunks = web_results[:k]
                fallback_used = "web"
                quality = "correct"

        # Step 7: Format context
        context_parts = []
        for c in chunks:
            src = c.get("source", c.get("url", "?"))
            title = c.get("title", c.get("heading", ""))
            content = c.get("content", "")[:WEB_SEARCH_MAX_CHARS]
            context_parts.append(f"[{src}] {title}\n{content}")

        context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        result = {
            "context": context,
            "chunks": chunks,
            "quality": quality,
            "domain": domain,
            "dcd_confidence": confidence,
            "query_type": query_type,
            "fallback_used": fallback_used,
            "threshold": threshold,
            "max_score": max_score,
            "entity_match_ratio": entity_match_ratio,
            "entity_match_pass": em_pass,
            "entity_match_entities": list(em_entities),
            "entity_match_found": list(em_matched),
            "node": LOCAL_NODE_NAME,
        }

        # Inject into query if context available
        if context:
            result["augmented_query"] = f"""Контекст из базы знаний ({LOCAL_NODE_NAME}):
{context}

Вопрос пользователя: {query}

Ответь на вопрос, используя контекст. Если контекст не содержит ответа — скажи что не знаешь."""

        return result


def classify_query_type(q: str) -> str:
    """Simple heuristic query classification."""
    ql = q.lower()
    if any(w in ql for w in ["кто", "когда", "где", "сколько", "цена", "стоит", "партномер"]):
        return "factual"
    if any(w in ql for w in ["почему", "как", "зачем", "сравни", "разница", "оптималь"]):
        return "analytical"
    if any(w in ql for w in ["спланируй", "предложи", "создай", "напиши", "разработай"]):
        return "synthesis"
    return "default"


def main():
    p = argparse.ArgumentParser(description="CRAG Search v2")
    p.add_argument("query", nargs="?", default=None)
    p.add_argument("-k", "--topk", type=int, default=DEFAULT_K)
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-sessions", action="store_true")
    args = p.parse_args()

    if not args.query:
        # Interactive mode
        q = sys.stdin.read().strip()
        if not q: return
        args.query = q

    search = CragSearch()
    result = search.search(args.query, k=args.topk, include_sessions=not args.no_sessions)

    if args.json:
        # Clean up for JSON output
        clean = {k: v for k, v in result.items() if k != "augmented_query"}
        clean["chunks"] = [{"source": c.get("source",""), "title": c.get("title",""), 
                           "content": c.get("content","")[:200], "score": c.get("score",0)} 
                          for c in result.get("chunks", [])]
        print(json.dumps(clean, ensure_ascii=False, indent=2))
    else:
        print(f"Query: {args.query}")
        print(f"Domain: {result['domain']} (confidence: {result['dcd_confidence']})")
        print(f"Quality: {result['quality']} | Fallback: {result.get('fallback_used','none')}")
        print(f"Chunks: {len(result.get('chunks',[]))} | Max score: {result['max_score']:.3f}")
        if result.get("context"):
            print(f"\nContext ({len(result['context'])} chars):")
            print(result["context"][:1500])

if __name__ == "__main__":
    main()
