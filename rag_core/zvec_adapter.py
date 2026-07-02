"""
ZVec Adapter for RAG v2 — hybrid FTS + Vector search with RRF fusion.

Based on kanban production pipeline. Replaces vector-only search.

Usage:
    from zvec_adapter import ZVecSearcher
    searcher = ZVecSearcher()
    results = searcher.search(query, topk=5)
    results_hybrid = searcher.search_hybrid(query, topk=5)
"""
import os
import sys
import json
import time
import logging
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_config import ZVEC_PATH, ZVEC_COLLECTION, ZVEC_SESSIONS_COLLECTION
from rag_config import EMBEDDING_URL, EMBEDDING_MODEL, EMBEDDING_DIM
from rag_config import ensure_zvec_lock

logger = logging.getLogger(__name__)


class ZVecSearcher:
    """ZVec search with LM Studio embeddings. Supports hybrid FTS+Vector."""

    _instances: dict[str, tuple] = {}  # collection_path -> (collection, lock_path)

    def __init__(self, collection: str = ZVEC_COLLECTION):
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
                from rag_config import ZVEC_COLLECTION as coll_name
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
        """Vector search with domain filter (original API, preserved)."""
        coll = self._ensure_collection()
        emb = self._get_embedding(query)

        if emb is None or sum(abs(v) for v in emb) == 0:
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

        return self._format_results(results, topk)

    def search_hybrid(
        self,
        query: str,
        topk: int = 5,
        domain: str = None,
        recall_topk: int = 20,
        rrf_constant: int = 60,
    ) -> list[dict]:
        """Hybrid search: FTS (BM25) + Vector (Cosine) → RRF fusion.

        Significantly better recall on exact matches (document numbers,
        error messages, config keys). Requires FTS index on 'content' field.

        Args:
            query: search query text
            topk: final number of results to return
            domain: optional category filter (e.g. "devops")
            recall_topk: over-fetch factor before RRF fusion (default 20)
            rrf_constant: RRF rank constant (default 60)

        Returns:
            List of dicts with score, content, metadata.
        """
        coll = self._ensure_collection()
        emb = self._get_embedding(query)

        if emb is None or sum(abs(v) for v in emb) == 0:
            # Zero embedding — fall back to FTS only
            logger.warning("Zero embedding for query, falling back to FTS only")
            return self._fts_only(query, topk, domain)

        from zvec import Query, Fts, RrfReRanker

        filter_expr = None
        if domain:
            filter_expr = f'category = "{domain}"'

        try:
            # Two separate queries fused by RRF
            # Zvec requires: "A single Query should not set both fts and vector"
            results = coll.query(
                queries=[
                    Query(
                        field_name="content",
                        fts=Fts(match_string=query),
                    ),
                    Query(
                        field_name="embedding",
                        vector=emb,
                    ),
                ],
                topk=recall_topk,
                reranker=RrfReRanker(rank_constant=rrf_constant),
                filter=filter_expr,
                output_fields=["source", "heading", "category", "node", "content", "title", "tags"],
            )
        except Exception as e:
            logger.warning("Hybrid search failed (%s), falling back to vector only", e)
            return self.search(query, topk, domain)

        return self._format_results(results, topk)

    def _fts_only(self, query: str, topk: int, domain: str = None) -> list[dict]:
        """FTS-only fallback when embedding is unavailable."""
        coll = self._ensure_collection()

        from zvec import Query, Fts

        filter_expr = None
        if domain:
            filter_expr = f'category = "{domain}"'

        try:
            results = coll.query(
                queries=[
                    Query(field_name="content", fts=Fts(match_string=query)),
                ],
                topk=topk,
                filter=filter_expr,
                output_fields=["source", "heading", "category", "node", "content", "title", "tags"],
            )
        except Exception:
            return []

        return self._format_results(results, topk)

    @staticmethod
    def _format_results(results, topk: int) -> list[dict]:
        """Format Zvec results into list of dicts."""
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
                    "text": f.get("content", "")[:500],  # alias for fuser compatibility
                    "title": f.get("title", ""),
                    "score": r.score if hasattr(r, 'score') else 0.0,
                })
        return formatted[:topk]
