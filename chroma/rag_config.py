"""
RAG Configuration — централизованные настройки для Agentic RAG + CRAG

Модели:
- e5-large-instruct — embedding (encoder-only, 1024d, мультиязычная)
- gemma-4-e4b — классификация запросов (factual/analytical/synthesis)
- qwen2.5-7b-instruct — query rewriting/expansion (7B, инструментальная)
- cosine threshold — вместо LLM grading (ни одна локальная модель не подходит)

CRAG (Corrective RAG):
- Evaluate: cosine threshold по типам запросов
- Correct: rewrite/decompose для Ambiguous, web search для Incorrect
- Adaptive: threshold зависит от query_type

Fallback: owl-alpha через OpenRouter для сложных случаев.
"""
import os
import socket

# Embedding API
EMBEDDING_URL = os.getenv("RAG_EMBEDDING_URL", "http://localhost:1234/v1/embeddings")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-baai-bge-m3-568m")
EMBEDDING_DIM = 1024

# LLM для классификации запросов (простая задача: 3 класса, короткий ответ)
LLM_CLASSIFY_URL = os.getenv("RAG_LLM_CLASSIFY_URL", "http://localhost:1234/v1/chat/completions")
LLM_CLASSIFY_MODEL = os.getenv("RAG_LLM_CLASSIFY_MODEL", "google/gemma-4-e4b")

# LLM для query rewriting (переформулировка, требует понимания намерения)
LLM_REWRITE_URL = os.getenv("RAG_LLM_REWRITE_URL", "http://localhost:1234/v1/chat/completions")
LLM_REWRITE_MODEL = os.getenv("RAG_LLM_REWRITE_MODEL", "qwen2.5-7b-instruct")

# LLM для query decomposition (разбивка сложного запроса на подзапросы)
LLM_DECOMPOSE_URL = os.getenv("RAG_LLM_DECOMPOSE_URL", "http://localhost:1234/v1/chat/completions")
LLM_DECOMPOSE_MODEL = os.getenv("RAG_LLM_DECOMPOSE_MODEL", "qwen2.5-7b-instruct")

# ChromaDB
CHROMA_PATH = os.getenv("RAG_CHROMA_PATH", os.path.expanduser("~/.cache/chroma"))
COLLECTION_NAME = "wiki"
SESSION_COLLECTION_NAME = "sessions"

# Session chunks: индексировать в wiki collection наряду с sessions?
# True = сессии попадают в общий поиск через rag_search/rag_query
INDEX_SESSIONS_IN_WIKI = os.getenv("RAG_INDEX_SESSIONS_IN_WIKI", "true").lower() == "true"

# Chunking
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100
MAX_CHARS_PER_INPUT = 2500

# Search
DEFAULT_K = 5
RELEVANCE_THRESHOLD = 0.3
FEDERATED_RAG_ENABLED = os.getenv("RAG_FEDERATED", "true").lower() == "true"
FEDERATED_NODES = [n.strip() for n in os.getenv("RAG_FEDERATED_NODES", "autolycus,hq").split(",") if n.strip()]

def _default_node_name() -> str:
    env_node = os.getenv("RAG_NODE_NAME")
    if env_node:
        return env_node
    host = socket.gethostname().lower()
    if "hermes" in host:
        return "hq"
    if "autolycus" in host:
        return "autolycus"
    return "autolycus"

LOCAL_NODE_NAME = _default_node_name()
FEDERATED_REMOTE_TIMEOUT = int(os.getenv("RAG_REMOTE_TIMEOUT", "25"))
FEDERATED_FETCH_K = int(os.getenv("RAG_FEDERATED_FETCH_K", "5"))
FEDERATED_HQ_SSH = os.getenv("RAG_HQ_SSH", "root@hermes-agent.ru")
FEDERATED_HQ_NODE = os.getenv("RAG_HQ_NODE", "hq")
FEDERATED_REMOTE_NODES = {
    "autolycus": os.getenv("RAG_AUTO_SSH", "root@autolycus-agent.ru"),
    "hq": FEDERATED_HQ_SSH,
}
FEDERATED_REMOTE_RAG_DIR = os.getenv("RAG_REMOTE_RAG_DIR", "/root/rag-deploy")
FEDERATED_REMOTE_SCRIPT = os.getenv("RAG_REMOTE_QUERY_SCRIPT", "rag_remote_query.py")

# Reranker — bge-reranker-v2-m3 cross-encoder via LM Studio
RERANK_ENABLED = os.getenv("RAG_RERANK", "true").lower() == "true"
RERANK_TOP_CANDIDATES = int(os.getenv("RAG_RERANK_TOP", "15"))
RERANK_FINAL_K = int(os.getenv("RAG_RERANK_FINAL_K", str(DEFAULT_K)))
RERANK_URL = os.getenv("RAG_RERANK_URL", "http://localhost:1234/v1/embeddings")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "text-embedding-bge-reranker-v2-m3")
RERANK_TIMEOUT = int(os.getenv("RAG_RERANK_TIMEOUT", "10"))

# Host routing — статический (НЕ keyword-based)
# RAG_HOST определяет текущий нод (устанавливается в environ, не из текста запроса)
RAG_HOST = os.getenv("RAG_HOST", "autolycus")

# ── CRAG: Adaptive Cosine Thresholds ────────────────────────────────────
# Разные пороги в зависимости от типа запроса:
# - factual: выше — точность важнее, ответ должен быть строго по wiki
# - analytical: средний — покрытие важнее, могут быть нужны смежные темы
# - synthesis: шире — нужен максимум контекста для синтеза
# - web_fallback: порог при котором падаем на веб-поиск
COSINE_THRESHOLDS = {
    "factual":     0.75,    # точность
    "analytical":  0.65,    # покрытие
    "synthesis":   0.60,    # широта
    "default":     0.70,    # запасной
}

# Порог для определения "incorrect" (CRAG: все чанки ниже → web)
# Если доля чанков выше этого порога < MIN_RELEVANT_RATIO → incorrect
MIN_RELEVANT_RATIO = 0.2   # < 20% чанков выше threshold → incorrect
MIN_RELEVANT_COUNT = 1     # < 1 чанк выше threshold → incorrect
AMBIGUOUS_RATIO = 0.5     # < 50% чанков выше threshold → ambiguous

# Source-type boost (мягкий, без clamp до 1.0)
SOURCE_BOOST = {
    "concepts/":   0.05,
    "adr/":        0.03,
    "plans/":      0.03,
    "manuals/":    0.03,
    "sessions/":   0.02,  # session chunks in wiki collection
}

# Session chunk embedding prefix (differs from wiki passages)
SESSION_EMBED_PREFIX = "Instruct: Given a search query, retrieve relevant conversation passages\nQuery:"
WIKI_EMBED_PREFIX = "Instruct: Given a wiki search query, retrieve relevant wiki passages\nQuery:"

# Wiki paths
WIKI_PATHS = [
    os.path.expanduser("/root/wiki"),
    os.path.expanduser("/root/llm-wiki"),
    os.path.expanduser("/root/.autolycus/skills"),
]

# Exclude directories from indexing (noise reduction)
EXCLUDE_EMAIL_CACHE = os.getenv("RAG_EXCLUDE_EMAIL_CACHE", "true").lower() == "true"
EXCLUDE_DIRS = [
    ".email_cache" if EXCLUDE_EMAIL_CACHE else None,
    "queries",              # mempalace query logs
    "session-notes",        # session notes (raw logs)
    "wiki/log*",            # лог вики
    "raw/search_*",         # search result dumps
    "raw/import-digest*",   # дайджесты импортов (telegram/cli переписка)
    "raw/product/meeting*", # переписки из TG/CLI с датами
    "concepts/202605*-",    # экспортированные сессии из TG/CLI
    "concepts/202606*-",    # экспортированные сессии из TG/CLI
    "skills/references",    # skill reference templates (duplicates)
    "skills/scripts",       # executable scripts, not knowledge
    "skills/assets",        # binary assets
    "node_modules",         # npm packages
    ".git",                 # git internals
]
EXCLUDE_DIRS = [p for p in EXCLUDE_DIRS if p]

# ── Web Search (CRAG fallback) ──────────────────────────────────────────
WEB_SEARCH_ENABLED = os.getenv("RAG_WEB_SEARCH", "true").lower() == "true"
WEB_SEARCH_MAX_RESULTS = int(os.getenv("RAG_WEB_SEARCH_RESULTS", "5"))
WEB_SEARCH_TIMEOUT = int(os.getenv("RAG_WEB_SEARCH_TIMEOUT", "15"))
WEB_SEARCH_MAX_CHARS = int(os.getenv("RAG_WEB_SEARCH_MAX_CHARS", "3000"))

# ── CRAG: Query Decomposition ──────────────────────────────────────────
DECOMPOSE_ENABLED = os.getenv("RAG_DECOMPOSE", "true").lower() == "true"
DECOMPOSE_MAX_SUBQUERIES = int(os.getenv("RAG_DECOMPOSE_MAX", "3"))

# ── Indexer CRUD ────────────────────────────────────────────────────────
# Content-based dedup при индексации (пропуск точных дубликатов)
CONTENT_DEDUP_ENABLED = os.getenv("RAG_CONTENT_DEDUP", "true").lower() == "true"
