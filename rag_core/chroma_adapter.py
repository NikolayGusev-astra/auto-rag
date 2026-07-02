#!/usr/bin/env python3
"""ChromaDB adapter — drop-in replacement for ZVecSearcher.

Используется на хостах без AVX2 (HQ: Intel Xeon E5-2680 v2).
Интерфейс совместим с ZVecSearcher для бесшовной подмены.

Зависимости: chromadb, requests (через _get_embedding).
"""

import json
import os
import hashlib
import subprocess
import sys
import time
from typing import Optional

# ── Config (copied from rag_config for standalone use) ──────────────
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-baai-bge-m3-568m")
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://localhost:1234/v1/embeddings")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))
CHROMA_PATH = os.environ.get("CHROMA_PATH", os.path.expanduser("~/.cache/chroma"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "wiki")


# ── Embedding (batch) ──────────────────────────────────────────────
def _get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch embedding via LM Studio API (Python HTTP, not subprocess)."""
    if not texts:
        return []
    try:
        import urllib.request
        payload = json.dumps({
            "model": EMBEDDING_MODEL,
            "input": [t[:2000] for t in texts],
        }).encode("utf-8")
        req = urllib.request.Request(
            EMBEDDING_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read().decode())
        return [d["embedding"] for d in data["data"]]
    except Exception:
        return [[0.0] * EMBEDDING_DIM for _ in texts]


def _get_embedding(text: str) -> list[float]:
    """Single embedding (fallback)."""
    embs = _get_embeddings_batch([text])
    return embs[0] if embs else [0.0] * EMBEDDING_DIM


# ── ChromaAdapter ──────────────────────────────────────────────────
class ChromaSearcher:
    """ChromaDB search — singleton per collection. Интерфейс ~ ZVecSearcher."""

    _instances: dict[str, object] = {}

    def __init__(self, collection: str = CHROMA_COLLECTION):
        self.coll_name = collection
        self._coll = None

    def _ensure_collection(self):
        if self._coll is not None:
            return self._coll
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            self._coll = client.get_collection(self.coll_name)
        except Exception:
            self._coll = client.create_collection(
                name=self.coll_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._coll

    def search(self, query: str, topk: int = 5, domain: Optional[str] = None) -> list[dict]:
        """Vector search with optional domain filter."""
        coll = self._ensure_collection()
        if coll.count() == 0:
            return []
        emb = _get_embedding(query)
        if not emb or sum(abs(v) for v in emb) == 0:
            return []
        where_filter = None
        if domain:
            where_filter = {"domain": domain}
        try:
            results = coll.query(
                query_embeddings=[emb],
                n_results=topk,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []
        formatted = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for i in range(len(ids)):
            meta = metas[i] if metas else {}
            score = 1.0 - dists[i] if dists else 0.0
            formatted.append({
                "source": meta.get("source", ""),
                "heading": meta.get("heading", ""),
                "content": docs[i] if docs else "",
                "category": meta.get("category", ""),
                "node": meta.get("node", ""),
                "title": meta.get("title", ""),
                "tags": meta.get("tags", ""),
                "score": round(score, 4),
            })
        return formatted

    def get_stats(self) -> dict:
        coll = self._ensure_collection()
        return {"doc_count": coll.count(), "index_completeness": {"embedding": 1.0}}


# ── Chroma Indexer (populate from wiki) ────────────────────────────
class ChromaIndexer:
    """Populate ChromaDB from wiki files. Skip files with problematic metadata."""

    def __init__(self, collection: str = CHROMA_COLLECTION):
        self.coll_name = collection
        self._coll = None
        self._skip_count = 0

    def _ensure_collection(self):
        if self._coll is not None:
            return self._coll
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            client.delete_collection(self.coll_name)
        except Exception:
            pass
        self._coll = client.create_collection(
            name=self.coll_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._coll

    def _safe_meta(self, val: str) -> str:
        """Ensure metadata value is safe for ChromaDB (no emoji, no encoding issues)."""
        if not isinstance(val, str):
            val = str(val)
        # Remove non-BMP characters (emoji, etc.)
        safe = ''.join(c for c in val if ord(c) < 0x10000)
        # Truncate to 200 chars
        return safe[:200]

    def index(self, wiki_paths: list[str], batch_size: int = 16):
        """Index .md files into ChromaDB with batch embedding."""
        coll = self._ensure_collection()
        files = []
        for base in wiki_paths:
            if not os.path.isdir(base):
                continue
            for root, dirs, fnames in os.walk(base):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                for fn in fnames:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in ('.md', '.txt', '.rst', '.py', '.yaml', '.yml', '.json', '.toml', '.sh', '.cfg', '.ini', '.conf', '.env'):
                        fp = os.path.join(root, fn)
                        if any(pat in fp for pat in ['.email_cache', 'node_modules', '.git', '__pycache__']):
                            continue
                        files.append(fp)
        files.sort()
        print(f"  📄 Found {len(files)} files")
        total_docs = 0
        skipped = 0
        batch_texts = []
        batch_metas = []
        batch_ids = []
        for i, fp in enumerate(files):
            try:
                with open(fp, 'r', errors='ignore') as f:
                    text = f.read()
                if len(text.strip()) < 20:
                    continue
                source = os.path.relpath(fp, os.path.commonpath(wiki_paths))
                # Ensure safe metadata
                safe_source = self._safe_meta(source)
                safe_title = self._safe_meta(os.path.splitext(os.path.basename(fp))[0])
                safe_category = self._safe_meta(source.split("/")[0] if "/" in source else "wiki")
                doc_id = f"doc_{hashlib.sha256(text.encode()).hexdigest()[:16]}_{i}"
                batch_texts.append(text[:2000])
                batch_metas.append({
                    "source": safe_source,
                    "heading": safe_title,
                    "category": safe_category,
                    "node": "chroma",
                    "title": safe_title,
                    "tags": "",
                })
                batch_ids.append(doc_id)
                total_docs += 1
                if len(batch_texts) >= batch_size or i == len(files) - 1:
                    embeddings = _get_embeddings_batch(batch_texts)
                    coll.add(
                        ids=batch_ids,
                        embeddings=embeddings,
                        documents=batch_texts,
                        metadatas=batch_metas,
                    )
                    if i % (batch_size * 10) == 0:
                        pct = (i + 1) * 100 // max(len(files), 1)
                        print(f"  📊 {pct}% ({total_docs} docs indexed)")
                    batch_texts = []
                    batch_metas = []
                    batch_ids = []
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  ⚠ {os.path.basename(fp)[:50]}: {e}")
                continue
        print(f"\n  ✅ Done: {coll.count()} docs indexed (skipped {skipped})")


if __name__ == "__main__":
    import time
    t0 = time.time()
    s = ChromaSearcher()
    r = s.search("как настроить postgresql streaming replication", topk=3)
    t1 = time.time()
    print(f"Search: {(t1-t0)*1000:.1f}ms, results: {len(r)}")
    print(f"Stats: {s.get_stats()}")