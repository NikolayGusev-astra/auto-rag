"""
ZVec Adapter for RAG v2 — заменяет ChromaDB на ZVec в CRAG-пайплайне.

Использование:
    from zvec_adapter import ZVecSearcher
    searcher = ZVecSearcher()
    results = searcher.search(query, topk=5)
"""
import os
import sys
import json
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_config import ZVEC_PATH, ZVEC_WIKI_COLLECTION, ZVEC_SESSIONS_COLLECTION
from rag_config import EMBEDDING_URL, EMBEDDING_MODEL, EMBEDDING_DIM
from rag_config import ensure_zvec_lock


class ZVecSearcher:
    """ZVec search with LM Studio embeddings. Singleton per collection."""

    _instances: dict[str, tuple] = {}  # collection_path -> (collection, lock_path)

    def __init__(self, collection: str = ZVEC_WIKI_COLLECTION):
        self.coll_path = os.path.join(ZVEC_PATH, collection)
        self._coll = None
        self._lock_path = None

    def _ensure_collection(self):
        if self._coll is not None:
            return self._coll
        import zvec
        zvec.init()
        # LOCK workaround
        self._lock_path = ensure_zvec_lock(self.coll_path)
        if os.path.exists(self.coll_path):
            try:
                self._coll = zvec.open(self.coll_path)
            except Exception as e:
                from rag_config import ZVEC_WIKI_COLLECTION as coll_name
                from zvec import CollectionSchema, FieldSchema, VectorSchema, DataType
                from zvec import FtsIndexParam, HnswIndexParam, InvertIndexParam, CollectionOption, MetricType
                import shutil
                shutil.rmtree(self.coll_path, ignore_errors=True)
                coll_name_clean = os.path.basename(self.coll_path)
                schema = CollectionSchema(
                    name=coll_name_clean or "wiki",
                    fields=[
                        FieldSchema("source", DataType.STRING, nullable=False, index_param=InvertIndexParam()),
                        FieldSchema("heading", DataType.STRING, nullable=True),
                        FieldSchema("category", DataType.STRING, nullable=False, index_param=InvertIndexParam()),
                        FieldSchema("node", DataType.STRING, nullable=False),
                        FieldSchema("content_hash", DataType.STRING, nullable=True),
                        FieldSchema("char_count", DataType.INT32, nullable=True),
                        FieldSchema("title", DataType.STRING, nullable=True),
                        FieldSchema("tags", DataType.STRING, nullable=True),
                        FieldSchema("content", DataType.STRING, nullable=False,
                                    index_param=FtsIndexParam(tokenizer_name="standard", filters=["lowercase"])),
                    ],
                    vectors=[
                        VectorSchema("embedding", DataType.VECTOR_FP32, dimension=EMBEDDING_DIM,
                                    index_param=HnswIndexParam(metric_type=MetricType.COSINE)),
                    ],
                )
                self._coll = zvec.create_and_open(
                    self.coll_path, schema,
                    CollectionOption(read_only=False, enable_mmap=True),
                )
        else:
            self._coll = zvec.open(self.coll_path)
        return self._coll

    def _get_embedding(self, text: str) -> list[float]:
        """LM Studio embedding via curl subprocess (requests fails on localhost:1234)."""
        import subprocess, json
        try:
            payload = json.dumps({"model": EMBEDDING_MODEL, "input": [text]})
            r = subprocess.run(
                ["curl", "-s", "--max-time", "10", EMBEDDING_URL, "-d", payload, "-H", "Content-Type: application/json"],
                capture_output=True, text=True, timeout=15
            )
            if r.returncode == 0 and r.stdout:
                data = json.loads(r.stdout)
                return data["data"][0]["embedding"]
        except Exception:
            pass
        return [0.0] * EMBEDDING_DIM

    def search(self, query: str, topk: int = 5, domain: str = None) -> list[dict]:
        """Vector search with domain filter."""
        coll = self._ensure_collection()
        emb = self._get_embedding(query)

        if emb is None or sum(abs(v) for v in emb) == 0:
            # Zero embedding → can't search
            return []

        from zvec import Query

        filter_expr = None
        if domain:
            filter_expr = f'category = "{domain}"'

        try:
            results = coll.query(
                queries=[Query(field_name="embedding", vector=emb)],
                topk=topk,
                filter=filter_expr,
                output_fields=["source", "heading", "category", "node", "content", "title", "tags"],
            )
        except Exception:
            return []

        # Format results
        formatted = []
        if isinstance(results, list):
            for r in results:
                f = r.fields or {}
                formatted.append({
                    "id": r.id if hasattr(r, 'id') else "",
                    "source": f.get("source", ""),
                    "heading": f.get("heading", ""),
                    "category": f.get("category", ""),
                    "node": f.get("node", ""),
                    "content": f.get("content", "")[:500],
                    "title": f.get("title", ""),
                    "score": r.score if hasattr(r, 'score') else 0.0,
                })
        return formatted[:topk]
