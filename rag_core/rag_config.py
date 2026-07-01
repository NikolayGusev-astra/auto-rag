"""RAG Configuration — для публичной версии. API keys вынесены в .env"""
import os

# ── ZVec ──────────────────────────────────────────────────────────
ZVEC_PATH = os.getenv("RAG_ZVEC_PATH", os.path.expanduser("~/.cache/zvec"))
ZVEC_COLLECTION = os.getenv("RAG_ZVEC_COLLECTION", "wiki")
ZVEC_SESSIONS_COLLECTION = os.getenv("RAG_ZVEC_SESSIONS", "sessions")

# ── Embedding (bge-m3 через LM Studio или sentence-transformers) ──
EMBEDDING_URL = os.getenv("RAG_EMBEDDING_URL", "http://localhost:1234/v1/embeddings")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-baai-bge-m3-568m")
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1024"))
EMBEDDING_TIMEOUT = int(os.getenv("RAG_EMBEDDING_TIMEOUT", "30"))

# ── LLM (для rewrite/decompose) ──────────────────────────────────
LM_STUDIO_CHAT_URL = os.getenv("RAG_LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")
LLM_REWRITE_ENABLED = os.getenv("RAG_REWRITE", "false").lower() == "true"
LLM_REWRITE_MODEL = os.getenv("RAG_REWRITE_MODEL", "qwen2.5-7b-instruct")

# ── Reranker (bge-reranker-v2-m3 через LM Studio) ────────────────
RERANK_ENABLED = os.getenv("RAG_RERANK", "false").lower() == "true"
RERANK_TOP_CANDIDATES = int(os.getenv("RAG_TOP_CANDIDATES", "15"))
RERANK_FINAL_K = int(os.getenv("RAG_FINAL_K", "5"))
RERANK_URL = os.getenv("RAG_RERANK_URL", EMBEDDING_URL)
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "text-embedding-bge-reranker-v2-m3")

# ── Chunking ──────────────────────────────────────────────────────
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100
MAX_CHARS_PER_INPUT = 2500
DEFAULT_K = 5

# ── Web Search (SearXNG) ─────────────────────────────────────────
WEB_SEARCH_ENABLED = os.getenv("RAG_WEB_SEARCH", "true").lower() == "true"
WEB_SEARCH_MAX_RESULTS = int(os.getenv("RAG_WEB_RESULTS", "5"))
WEB_SEARCH_TIMEOUT = int(os.getenv("RAG_WEB_TIMEOUT", "15"))
WEB_SEARCH_MAX_CHARS = int(os.getenv("RAG_WEB_MAX_CHARS", "3000"))
SEARXNG_URL = os.getenv("RAG_SEARXNG_URL", "http://localhost:8080")
SEARXNG_ENABLED = os.getenv("RAG_SEARXNG", "true").lower() == "true"

# ── MCP Servers (Context7 — доступен по API) ─────────────────────
MCP_ENABLED = os.getenv("RAG_MCP", "false").lower() == "true"
MCP_TIMEOUT = int(os.getenv("RAG_MCP_TIMEOUT", "20"))
MCP_MAX_RESULTS = int(os.getenv("RAG_MCP_RESULTS", "3"))

# Реальный ключ в .env: CONTEXT7_API_KEY=...
_CONTEXT7_KEY = os.getenv("CONTEXT7_API_KEY", "")
MCP_SERVERS = {}
if _CONTEXT7_KEY:
    MCP_SERVERS["context7"] = {
        "type": "http",
        "url": "https://mcp.context7.com/mcp",
        "headers": {"CONTEXT7_API_KEY": _CONTEXT7_KEY, "Content-Type": "application/json"},
        "query_tool": "query-docs",
    }

MCP_FALLBACK_CHAIN = list(MCP_SERVERS.keys())
DCD_COLLECTION_MCP_MAP = {"software-dev": {"*": "context7"}, "devops": {"*": "context7"}}

# ── Thresholds ────────────────────────────────────────────────────
COSINE_THRESHOLDS = {"factual": 0.60, "analytical": 0.50, "synthesis": 0.45, "default": 0.50}
MIN_RELEVANT_RATIO = 0.2
MIN_RELEVANT_COUNT = 1
AMBIGUOUS_RATIO = 0.5
SOURCE_BOOST = {"wiki/": 0.05, "adr/": 0.03, "manuals/": 0.03}

# ── Node ──────────────────────────────────────────────────────────
LOCAL_NODE_NAME = os.getenv("RAG_NODE_NAME", "local")
FEDERATED_RAG_ENABLED = False

# ── ZVec LOCK workaround ─────────────────────────────────────────
def ensure_zvec_lock(path: str) -> str:
    import os as _os
    lock_path = _os.path.join(path, "LOCK")
    if _os.path.exists(path) and not _os.path.exists(lock_path):
        try:
            fd = _os.open(lock_path, _os.O_CREAT | _os.O_WRONLY, 0o644)
            _os.close(fd)
        except OSError:
            pass
    return lock_path

# ── DCD -> preferred web search source ────────────────────────────
# Определяет какой web поиск использовать при fallback для каждого домена/коллекции
# "searxng" — SearXNG (245 движков, общий поиск)
# "ddg" — DuckDuckGo (без API ключа, быстрый)
# "skip" — не делать web fallback
# None — дефолт (SearXNG, если доступен)
DCD_PREFERRED_WEB_SOURCE = {
    "ford-club": {"*": "searxng"},
    "devops": {"*": "searxng"},
    "software-dev": {"*": "skip"},
    "manuals": {"*": "searxng"},
    "infrastructure": {"*": "ddg"},
    "publishing": {"*": "skip"},
}
