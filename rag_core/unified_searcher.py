#!/usr/bin/env python3
"""Unified vector search — auto-detects ZVec (AVX2) or ChromaDB (no AVX2).

Drop-in replacement for both ZVecSearcher and ChromaSearcher.
"""

import os
import sys
import subprocess

# ── Config ────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-baai-bge-m3-568m")
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://localhost:1234/v1/embeddings")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))
ZVEC_PATH = os.environ.get("ZVEC_PATH", os.path.expanduser("~/.cache/zvec"))
ZVEC_COLLECTION = os.environ.get("ZVEC_COLLECTION", "wiki")
CHROMA_PATH = os.environ.get("CHROMA_PATH", os.path.expanduser("~/.cache/chroma"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "wiki")


def _has_avx2() -> bool:
    """Check if CPU supports AVX2."""
    try:
        with open("/proc/cpuinfo") as f:
            return "avx2" in f.read().lower()
    except Exception:
        return False


# ── Embedding ─────────────────────────────────────────────────────
def _get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch embedding via LM Studio."""
    import urllib.request
    import json
    if not texts:
        return []
    payload = json.dumps({
        "model": EMBEDDING_MODEL,
        "input": [t[:2000] for t in texts],
    }).encode("utf-8")
    req = urllib.request.Request(EMBEDDING_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read().decode())
        return [d["embedding"] for d in data["data"]]
    except Exception:
        return [[0.0] * EMBEDDING_DIM for _ in texts]


def _get_embedding(text: str) -> list[float]:
    return _get_embeddings_batch([text])[0]


# ── Unified Searcher ──────────────────────────────────────────────
class UnifiedSearcher:
    """Auto-detects backend: ZVec if AVX2, else ChromaDB."""

    def __init__(self, collection: str = None):
        self._zvec = None
        self._chroma = None
        self._collection = collection or ZVEC_COLLECTION
        self._backend = None  # "zvec" or "chroma"
        self._detect_backend()

    def _detect_backend(self):
        """Auto-detect: ZVec (AVX2) preferred, Chroma fallback."""
        if _has_avx2():
            # Check ZVec collection exists
            zpath = os.path.join(ZVEC_PATH, self._collection)
            if os.path.exists(zpath):
                try:
                    import zvec
                    zvec.init()
                    zvec.open(zpath)
                    self._backend = "zvec"
                    return
                except Exception:
                    pass
        # Fallback to Chroma
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            client.get_collection(self._collection)
            self._backend = "chroma"
        except Exception:
            self._backend = "chroma"  # Will fail on search, but at least defined

    def _ensure_zvec(self):
        if self._zvec is None and self._backend == "zvec":
            import zvec
            import threading
            zvec.init()
            zpath = os.path.join(ZVEC_PATH, self._collection)
            lock_path = zpath + "/LOCK"
            try:
                with open(lock_path, "w") as f:
                    f.write(str(os.getpid()))
            except Exception:
                pass
            self._zvec = zvec.open(zpath)
        return self._zvec

    def _ensure_chroma(self):
        if self._chroma is None and self._backend == "chroma":
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            self._chroma = client.get_collection(self._collection)
        return self._chroma

    def search(self, query: str, topk: int = 5, domain: str = None) -> list[dict]:
        """Unified search: same interface as ZVecSearcher/ChromaSearcher."""
        emb = _get_embedding(query)
        if not emb or sum(abs(v) for v in emb) == 0:
            return []

        if self._backend == "zvec":
            coll = self._ensure_zvec()
            if coll is None:
                return []
            from zvec import Query
            filter_expr = f'category = "{domain}"' if domain else None
            try:
                results = coll.query(
                    queries=[Query(field_name="embedding", vector=emb)],
                    topk=topk,
                    filter=filter_expr,
                    output_fields=["source", "heading", "category", "node", "content", "title", "tags"],
                )
            except Exception:
                return []
            formatted = []
            if isinstance(results, list):
                for r in results:
                    f = r.fields or {}
                    formatted.append({
                        "source": f.get("source", ""),
                        "heading": f.get("heading", ""),
                        "content": f.get("content", f.get("text", "")),
                        "category": f.get("category", ""),
                        "node": f.get("node", ""),
                        "title": f.get("title", ""),
                        "tags": f.get("tags", ""),
                        "score": round(r.score, 4),
                    })
            return formatted

        elif self._backend == "chroma":
            coll = self._ensure_chroma()
            if coll is None or coll.count() == 0:
                return []
            # B2: фильтруем по обоим полям (category И domain пишутся в metadata
            # chroma-индексатором), т.к. ZVec-ветка использует category, а DCD
            # передаёт domain — иначе доменная фильтрация на Chroma тихо пуста.
            where_filter = (
                {"$or": [{"category": domain}, {"domain": domain}]}
                if domain else None
            )
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
                m = metas[i] if metas else {}
                score = 1.0 - dists[i] if dists else 0.0
                formatted.append({
                    "source": m.get("source", ""),
                    "heading": m.get("heading", ""),
                    "content": docs[i] if docs else "",
                    "category": m.get("category", ""),
                    "node": m.get("node", ""),
                    "title": m.get("title", ""),
                    "tags": m.get("tags", ""),
                    "score": round(score, 4),
                })
            return formatted

        return []

    def get_stats(self) -> dict:
        """Return stats compatible with both backends."""
        if self._backend == "zvec":
            coll = self._ensure_zvec()
            if coll:
                return {"doc_count": coll.stats.doc_count, "index_completeness": coll.stats.index_completeness}
        elif self._backend == "chroma":
            coll = self._ensure_chroma()
            if coll:
                return {"doc_count": coll.count(), "index_completeness": {"embedding": 1.0}}
        return {"doc_count": 0, "index_completeness": {"embedding": 0.0}}

    @property
    def backend(self) -> str:
        return self._backend


# ── Convenience function for rag_async ────────────────────────────
_UNIFIED_SEARCHER = None


def get_unified_searcher(collection: str = None) -> UnifiedSearcher:
    """Singleton getter for rag_async integration."""
    global _UNIFIED_SEARCHER
    if _UNIFIED_SEARCHER is None:
        _UNIFIED_SEARCHER = UnifiedSearcher(collection)
    return _UNIFIED_SEARCHER


if __name__ == "__main__":
    import time
    s = UnifiedSearcher()
    print(f"Backend: {s.backend}")
    print(f"Stats: {s.get_stats()}")
    if s.get_stats()["doc_count"] > 0:
        t0 = time.time()
        r = s.search("nginx reverse proxy", topk=3)
        t1 = time.time()
        print(f"Search took {(t1-t0)*1000:.1f}ms")
        for res in r:
            print(f"  {res['score']:.4f} | {res['source'][:60]}")
    else:
        print("Collection empty - run indexer first")