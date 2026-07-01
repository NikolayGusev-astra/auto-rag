"""RAG Configuration — generic public version.
API keys and URLs via environment variables.
"""

import os
import platform

# Node identity
RAG_HOST = os.getenv("RAG_HOST", "local")
RAG_NODE_NAME = os.getenv("RAG_NODE_NAME", "local")

# ZVec Collections
_zvec_base = os.getenv("RAG_ZVEC_PATH", os.path.expanduser("~/.cache/zvec"))
ZVEC_PATH = os.path.abspath(_zvec_base)
ZVEC_COLLECTION = os.getenv("RAG_ZVEC_COLLECTION", "wiki")
ZVEC_SESSIONS_COLLECTION = os.getenv("RAG_ZVEC_SESSIONS", "sessions")
ZVEC_WINDOWS = platform.system() == "Windows"
ZVEC_LOCK_PATH = os.path.join(ZVEC_PATH, "LOCK")

# Embedding API (local LM Studio)
EMBEDDING_URL = os.getenv("RAG_EMBEDDING_URL", "http://localhost:1234/v1/embeddings")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-baai-bge-m3-568m")
EMBEDDING_DIM = 1024
EMBEDDING_TIMEOUT = int(os.getenv("RAG_EMBEDDING_TIMEOUT", "120"))

# LM Studio API endpoints
LM_STUDIO_CHAT_URL = os.getenv("RAG_LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")

# LLM Classify
LLM_CLASSIFY_ENABLED = os.getenv("RAG_CLASSIFY", "true").lower() == "true"
LLM_CLASSIFY_URL = os.getenv("RAG_CLASSIFY_URL", LM_STUDIO_CHAT_URL)
LLM_CLASSIFY_MODEL = os.getenv("RAG_CLASSIFY_MODEL", "google/gemma-4-e4b")
LLM_CLASSIFY_TIMEOUT = int(os.getenv("RAG_CLASSIFY_TIMEOUT", "15"))

# LLM Rewrite
LLM_REWRITE_ENABLED = os.getenv("RAG_REWRITE", "true").lower() == "true"
LLM_REWRITE_URL = os.getenv("RAG_REWRITE_URL", LM_STUDIO_CHAT_URL)
LLM_REWRITE_MODEL = os.getenv("RAG_REWRITE_MODEL", "qwen2.5-7b-instruct")
LLM_REWRITE_TIMEOUT = int(os.getenv("RAG_REWRITE_TIMEOUT", "20"))

# LLM Decompose
DECOMPOSE_ENABLED = os.getenv("RAG_DECOMPOSE", "true").lower() == "true"
DECOMPOSE_MAX_SUBQUERIES = int(os.getenv("RAG_DECOMPOSE_MAX", "3"))
LLM_DECOMPOSE_URL = os.getenv("RAG_DECOMPOSE_URL", LM_STUDIO_CHAT_URL)
LLM_DECOMPOSE_MODEL = os.getenv("RAG_DECOMPOSE_MODEL", "qwen2.5-7b-instruct")
LLM_DECOMPOSE_TIMEOUT = int(os.getenv("RAG_DECOMPOSE_TIMEOUT", "20"))

# LLM Eval Scorer
RERANK_LM_STUDIO_URL = os.getenv("RAG_EVAL_URL", LM_STUDIO_CHAT_URL)
RERANK_LM_STUDIO_MODEL = os.getenv("RAG_EVAL_MODEL", "qwen3-4b")

# LLM Eval (non-thinking, for fuzzy cases)
LLM_EVAL_ENABLED = os.getenv("RAG_LLM_EVAL", "true").lower() == "true"
LLM_EVAL_MODEL = os.getenv("RAG_LLM_EVAL_MODEL", "qwen2.5-7b-instruct")
LLM_EVAL_URL = os.getenv("RAG_LLM_EVAL_URL", LM_STUDIO_CHAT_URL)
LLM_EVAL_TIMEOUT = int(os.getenv("RAG_LLM_EVAL_TIMEOUT", "15"))
LLM_EVAL_LOW_THRESHOLD = float(os.getenv("RAG_LLM_EVAL_LOW", "0.4"))
LLM_EVAL_HIGH_THRESHOLD = float(os.getenv("RAG_LLM_EVAL_HIGH", "0.75"))
RERANK_LM_STUDIO_TIMEOUT = int(os.getenv("RAG_EVAL_TIMEOUT", "30"))

# Reranker (bge-reranker-v2-m3)
RERANK_ENABLED = os.getenv("RAG_RERANK", "true").lower() == "true"
RERANK_TOP_CANDIDATES = int(os.getenv("RAG_RERANK_TOP", "15"))
RERANK_FINAL_K = int(os.getenv("RAG_RERANK_FINAL_K", "5"))
RERANK_URL = os.getenv("RAG_RERANK_URL", "http://localhost:1234/v1/embeddings")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "text-embedding-bge-reranker-v2-m3")
RERANK_TIMEOUT = int(os.getenv("RAG_RERANK_TIMEOUT", "10"))

# ChromaDB (legacy)
CHROMA_PATH = os.getenv("RAG_CHROMA_PATH", os.path.expanduser("~/.cache/chroma"))
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "rag_wiki")
SESSION_COLLECTION_NAME = os.getenv("RAG_SESSION_COLLECTION", "rag_sessions")
SESSION_EMBED_PREFIX = os.getenv("RAG_SESSION_PREFIX", "search_session: ")

# Chunking
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100
MAX_CHARS_PER_INPUT = 2500
DEFAULT_K = 5

# Web Fallback (CRAG)
WEB_SEARCH_ENABLED = os.getenv("RAG_WEB_SEARCH", "true").lower() == "true"
WEB_SEARCH_MAX_RESULTS = int(os.getenv("RAG_WEB_SEARCH_RESULTS", "5"))
WEB_SEARCH_TIMEOUT = int(os.getenv("RAG_WEB_SEARCH_TIMEOUT", "15"))
WEB_SEARCH_MAX_CHARS = int(os.getenv("RAG_WEB_SEARCH_MAX_CHARS", "3000"))
SEARXNG_URL = os.getenv("RAG_SEARXNG_URL", "http://localhost:8080")
SEARXNG_ENABLED = os.getenv("RAG_SEARXNG", "true").lower() == "true"

# MCP servers (CRAG fallback chain) — all via env vars
MCP_ENABLED = os.getenv("RAG_MCP", "true").lower() == "true"
MCP_TIMEOUT = int(os.getenv("RAG_MCP_TIMEOUT", "30"))
MCP_MAX_RESULTS = int(os.getenv("RAG_MCP_RESULTS", "3"))

# Build MCP_SERVERS from env vars (generic template)
def _build_mcp_servers():
    servers = {}
    
    # Confluence (generic)
    if os.getenv("RAG_CONFLUENCE_URL"):
        servers["confluence"] = {
            "type": "rest",
            "url": os.getenv("RAG_CONFLUENCE_URL"),
            "headers": {
                "Authorization": f"Bearer {os.getenv('RAG_CONFLUENCE_TOKEN', '')}",
                "Accept": "application/json",
            },
            "rest_query": "/rest/api/content/search?cql=text~\"{query_first3}\"&limit={max}&expand=space",
        }
    
    # Jira (generic)
    if os.getenv("RAG_JIRA_URL"):
        servers["jira"] = {
            "type": "rest",
            "url": os.getenv("RAG_JIRA_URL"),
            "headers": {
                "Authorization": f"Bearer {os.getenv('RAG_JIRA_TOKEN', '')}",
                "Accept": "application/json",
            },
            "rest_query": "/rest/api/2/search?jql=text~\"{query_and3}\"&maxResults={max}",
        }
    
    # Lodestone (generic)
    if os.getenv("RAG_LODESTONE_URL"):
        servers["lodestone"] = {
            "type": "http",
            "url": os.getenv("RAG_LODESTONE_URL") + "/mcp/",
            "headers": {
                "Authorization": f"Bearer {os.getenv('RAG_LODESTONE_TOKEN', '')}",
                "Accept": "application/json, text/event-stream",
            },
        }
    
    # Context7 (public)
    if os.getenv("RAG_CONTEXT7_URL"):
        servers["context7"] = {
            "type": "http",
            "url": os.getenv("RAG_CONTEXT7_URL", "https://context7.com/mcp/"),
            "headers": {
                "Accept": "application/json, text/event-stream",
            },
        }
    
    # Protopack (generic)
    if os.getenv("RAG_PROTOPACK_URL"):
        servers["protopack"] = {
            "type": "http",
            "url": os.getenv("RAG_PROTOPACK_URL"),
            "headers": {
                "Authorization": f"Bearer {os.getenv('RAG_PROTOPACK_TOKEN', '')}",
                "Accept": "application/json, text/event-stream",
            },
        }
    
    return servers

MCP_SERVERS = _build_mcp_servers()

# DCD preferred web source per domain/collection (override via env if needed)
DCD_PREFERRED_WEB_SOURCE = {
    "security": {"*": "context7"},
    "devops": {"*": "context7"},
    "software-dev": {"*": "context7"},
    "research": {"*": "context7"},
}

# ZVec category filters
ZVEC_CATEGORY_FILTERS = {
    "wiki": "category = 'wiki' OR category = 'llm-wiki'",
    "skills": "category = 'skill'",
}

# Cosine thresholds
COSINE_THRESHOLDS = {
    "factual": 0.35,
    "default": 0.25,
}

# Search limits
ZVEC_SEARCH_K = 5
WEB_SEARCH_MAX_RESULTS = 5
