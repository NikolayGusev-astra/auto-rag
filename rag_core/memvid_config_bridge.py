"""
memvid_config_bridge.py — Map pipeline EMBEDDING_* env vars to memvid's RAG_MEMVID_* env vars.

Our pipeline uses EMBEDDING_MODEL / EMBEDDING_URL / EMBEDDING_DIM at the top of
unified_searcher.py and rag_config.py.  Memvid expects RAG_MEMVID_EMBED_MODEL /
RAG_MEMVID_EMBED_URL / RAG_MEMVID_EMBED_API_KEY.  This module bridges the two
so that users who set the pipeline vars don't need to duplicate them.

Call `bridge_memvid_env()` before reading memvid config (e.g. at the top of
MemvidConfig.from_env()).
"""

import os

def bridge_memvid_env() -> None:
    """If memvid embed env vars are unset, inherit from our pipeline vars."""
    if not os.environ.get("RAG_MEMVID_EMBED_MODEL"):
        v = os.environ.get("EMBEDDING_MODEL")
        if v:
            os.environ["RAG_MEMVID_EMBED_MODEL"] = v
    if not os.environ.get("RAG_MEMVID_EMBED_URL"):
        v = os.environ.get("EMBEDDING_URL")
        if v:
            os.environ["RAG_MEMVID_EMBED_URL"] = v
    if not os.environ.get("RAG_MEMVID_EMBED_API_KEY"):
        # our LM Studio uses any non-empty key; default "lm-studio" is fine
        pass