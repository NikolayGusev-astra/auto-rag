"""
memvid_memory.py — Episodic memory layer for Hermes agent.

Wraps memvid (memvid-sdk) as an append-only, versioned, time-travelable
memory capsule AROUND the existing auto-rag pipeline. It is NOT a
replacement for ZVec/Chroma — it remembers what the agent already did,
so RAG retrieves knowledge from the world, memvid remembers the agent's
own past episodes.

------------------------------------------------------------------------------
USAGE (in rag_search.py / Hermes agent loop)
------------------------------------------------------------------------------

    from memvid_memory import MemvidMemory, Episode

    memory = MemvidMemory.for_tenant("hermes_default")

    # 1) recall prior episodes BEFORE running RAG
    priors = memory.recall(query, domain=domain)
    if priors and priors[0].score >= memory.recall_threshold:
        # high-confidence prior — short-circuit or augment prompt
        return priors[0].answer, priors[0].sources, priors[0].trace

    # 2) run your normal auto-rag pipeline
    answer, sources, trace = rag_search_orig(query, ...)

    # 3) record the episode (query, answer, sources, RagTrace)
    memory.record(Episode(
        query=query, answer=answer, sources=sources, trace=trace,
        domain=domain, tenant="hermes_default",
    ))

    # Optional: time-travel recall
    priors = memory.recall(query, when="last tuesday")

------------------------------------------------------------------------------
DESIGN PRINCIPLES
------------------------------------------------------------------------------
- Dependency-optional: if `memvid` SDK is missing or disabled via env,
  recall() returns [] and record() is a no-op. Safe to A/B test.
- One capsule per tenant: memory_{tenant}.mv2 (or .mv2e if encrypted).
- Thread-safe via a per-tenant RLock (Hermes streams SSE in parallel).
- Embeddings reused from existing LM Studio endpoint (bge-m3) via
  memvid's api_embed feature. Falls back to manual embed+cosine if the
  SDK does not expose precomputed-vector search.
- All failures degrade gracefully: a memory failure must NEVER break
  the main RAG flow.

------------------------------------------------------------------------------
ENV VARS (override defaults; see MemvidConfig)
------------------------------------------------------------------------------
  RAG_MEMVID_ENABLED          default: false (opt-in)
  RAG_MEMVID_MODE             off | recall | record | both (default: both)
  RAG_MEMVID_DIR              default: ./memvid_capsules
  RAG_MEMVID_TENANT           default: hermes_default
  RAG_MEMVID_RECALL_TOPK      default: 5
  RAG_MEMVID_RECALL_THRESHOLD default: 0.75
  RAG_MEMVID_EMBED_URL        default: $RAG_EMBEDDING_URL (LM Studio)
  RAG_MEMVID_EMBED_MODEL      default: bge-m3
  RAG_MEMVID_EMBED_API_KEY    default: lm-studio
  RAG_MEMVID_ENCRYPTION_KEY   optional -> enables .mv2e encrypted capsules
  RAG_MEMVID_TEMPORAL         true/false (default: true) -> temporal_track
  RAG_MEMVID_LOG_LEVEL        default: INFO
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("hermes.memvid")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] hermes.memvid: %(message)s"))
    log.addHandler(_h)
log.setLevel(os.environ.get("RAG_MEMVID_LOG_LEVEL", "INFO"))


# ---------------------------------------------------------------------------
# Episode data model
# ---------------------------------------------------------------------------
@dataclass
class Episode:
    """One agent episode: query → answer + provenance + trace.

    Serialized to JSON and stored as a single memvid Smart Frame.
    The `payload` bytes stored in memvid == json.dumps(episode_dict).
    """
    query: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    trace: Optional[Dict[str, Any]] = None      # RagTrace from auto-rag
    domain: Optional[str] = None
    tenant: Optional[str] = None
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # filled by MemvidMemory.recall()
    score: float = 0.0
    frame_id: Optional[str] = None
    # optional user correction / feedback hooks
    feedback: Optional[str] = None      # "correct" | "wrong" | None

    def to_payload(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False).encode("utf-8")

    @classmethod
    def from_payload(cls, raw: bytes | str) -> "Episode":
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        d = json.loads(raw)
        # tolerate missing fields
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class MemvidConfig:
    enabled: bool = False
    mode: str = "both"            # off | recall | record | both
    dir: Path = Path("./memvid_capsules")
    tenant: str = "hermes_default"
    recall_topk: int = 5
    recall_threshold: float = 0.75
    embed_url: str = "http://localhost:1234/v1/embeddings"
    embed_model: str = "text-embedding-multilingual-e5-large-instruct"
    embed_api_key: str = "lm-studio"
    encryption_key: Optional[str] = None
    temporal: bool = True

    @property
    def do_recall(self) -> bool:
        return self.enabled and self.mode in ("recall", "both")

    @property
    def do_record(self) -> bool:
        return self.enabled and self.mode in ("record", "both")

    @property
    def capsule_path(self) -> Path:
        ext = ".mv2e" if self.encryption_key else ".mv2"
        return self.dir / f"memory_{self.tenant}{ext}"

    @classmethod
    def from_env(cls) -> "MemvidConfig":
        def b(v: Optional[str], default: bool) -> bool:
            if v is None:
                return default
            return v.strip().lower() in ("1", "true", "yes", "on")

        return cls(
            enabled=b(os.environ.get("RAG_MEMVID_ENABLED"), False),
            mode=os.environ.get("RAG_MEMVID_MODE", "both").strip().lower(),
            dir=Path(os.environ.get("RAG_MEMVID_DIR", "./memvid_capsules")),
            tenant=os.environ.get("RAG_MEMVID_TENANT", "hermes_default"),
            recall_topk=int(os.environ.get("RAG_MEMVID_RECALL_TOPK", "5")),
            recall_threshold=float(
                os.environ.get("RAG_MEMVID_RECALL_THRESHOLD", "0.75")),
            embed_url=os.environ.get(
                "RAG_MEMVID_EMBED_URL",
                os.environ.get("RAG_EMBEDDING_URL",
                               "http://localhost:1234/v1/embeddings")),
            embed_model=os.environ.get("RAG_MEMVID_EMBED_MODEL", "text-embedding-multilingual-e5-large-instruct"),
            embed_api_key=os.environ.get(
                "RAG_MEMVID_EMBED_API_KEY", "lm-studio"),
            encryption_key=os.environ.get("RAG_MEMVID_ENCRYPTION_KEY") or None,
            temporal=b(os.environ.get("RAG_MEMVID_TEMPORAL"), True),
        )


# ---------------------------------------------------------------------------
# Backend protocol — isolates memvid-sdk version differences
# ---------------------------------------------------------------------------
class _MemvidBackend(Protocol):
    """Thin adapter around the memvid-sdk. Two implementations:
    - _RealMemvidBackend  : uses memvid-sdk (best-effort API probing)
    - _NoopMemvidBackend  : disabled / import-failed fallback
    """
    def put(self, payload: bytes, title: str, tags: Dict[str, str],
            uri: Optional[str] = None) -> Optional[str]: ...
    def commit(self) -> bool: ...
    def search(self, query: str, top_k: int,
               when: Optional[str] = None) -> List[Dict[str, Any]]: ...
    def close(self) -> None: ...


class _NoopMemvidBackend:
    """Used when memvid is disabled or import failed. Always empty."""
    def put(self, *a, **kw): return None
    def commit(self): return False
    def search(self, *a, **kw): return []
    def close(self): pass


class _RealMemvidBackend:
    """Native single-file memvid backend with external LM Studio embeddings."""
    def __init__(self, cfg: MemvidConfig):
        self.cfg = cfg
        self._embed = _Embedder(cfg)
        self._migrate_legacy_sidecar(cfg)
        self._mem = self._open_capsule(cfg)
        log.info("memvid backend ready: %s (tenant=%s, mode=%s, native_vec=true)",
                 cfg.capsule_path, cfg.tenant, cfg.mode)

    def _sdk(self):
        try:
            import memvid_sdk as memvid
            return memvid
        except ImportError:
            import memvid  # type: ignore[no-redef]
            return memvid

    def _migrate_legacy_sidecar(self, cfg: MemvidConfig) -> None:
        """Move legacy JSONL vectors into a native, single-file MV2 capsule.

        The first implementation stored embeddings in `<capsule>.vecidx.jsonl`
        because `add_memory_cards()` does not populate the SDK vector index.
        The SDK's documented `put_many(..., embeddings=...)` does, so migrate
        atomically and retain `.legacy.bak` files for rollback.
        """
        import json as _json
        legacy_idx = cfg.capsule_path.with_suffix(cfg.capsule_path.suffix + ".vecidx.jsonl")
        if not legacy_idx.exists():
            return
        rows = []
        try:
            for line in legacy_idx.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = _json.loads(line)
                if row.get("payload") and row.get("vec"):
                    rows.append(row)
        except Exception as e:
            log.warning("legacy memvid index unreadable; keeping it untouched: %s", e)
            return
        if not rows:
            return

        cfg.dir.mkdir(parents=True, exist_ok=True)
        temp = cfg.capsule_path.with_suffix(cfg.capsule_path.suffix + ".native.tmp")
        for stale in (temp, temp.with_suffix(temp.suffix + ".wal")):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        memvid = self._sdk()
        native = None
        try:
            native = memvid.create(str(temp), kind="basic", enable_vec=True, enable_lex=True)
            documents, vectors = [], []
            for row in rows:
                ep = row["payload"]
                documents.append({
                    "title": str(ep.get("query") or row.get("entity") or "episode")[:120],
                    "label": str(ep.get("domain") or "general"),
                    "text": _json.dumps(ep, ensure_ascii=False),
                    "metadata": {
                        "domain": str(ep.get("domain") or ""),
                        "tenant": str(ep.get("tenant") or cfg.tenant),
                    },
                })
                vectors.append(row["vec"])
            native.put_many(
                documents,
                embeddings=vectors,
                embedding_identity={
                    "kind": "manual",
                    "model": cfg.embed_model,
                    "dimension": len(vectors[0]),
                },
            )
            native.commit()
            stats = native.stats()
            if not stats.get("has_vec_index") or not stats.get("vec_index_bytes"):
                raise RuntimeError("native vector index was not created")
            native.close()
            native = None
            if cfg.capsule_path.exists():
                cfg.capsule_path.replace(cfg.capsule_path.with_suffix(cfg.capsule_path.suffix + ".legacy.bak"))
            temp.replace(cfg.capsule_path)
            legacy_idx.replace(legacy_idx.with_suffix(legacy_idx.suffix + ".legacy.bak"))
            log.info("migrated %d episodic vectors into native MV2 index", len(rows))
        except Exception as e:
            log.warning("native memvid migration failed; retaining legacy sidecar: %s", e)
            try:
                if native is not None:
                    native.close()
            except Exception:
                pass
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    # -- open / create capsule ----------------------------------------------
    def _open_capsule(self, cfg: MemvidConfig):
        cfg.dir.mkdir(parents=True, exist_ok=True)
        path = str(cfg.capsule_path)
        memvid = self._sdk()
        try:
            if cfg.capsule_path.exists() and hasattr(memvid, "use"):
                return memvid.use("basic", path, enable_vec=True, enable_lex=True)
            return memvid.create(path, kind="basic", apikey=None,
                                 enable_vec=True, enable_lex=True)
        except Exception as e:
            log.error("memvid capsule open failed (%s): %s", path, e)
            raise

    # -- put ---------------------------------------------------------------
    def put(self, payload: bytes, title: str, tags: Dict[str, str],
            uri: Optional[str] = None) -> Optional[str]:
        """Store an episode and its embedding inside the native MV2 index."""
        try:
            import json as _json
            ep = _json.loads(payload.decode("utf-8", "replace"))
            answer = str(ep.get("answer") or ep.get("query") or "")
            vector = self._embed.embed(answer)
            if not vector:
                log.warning("memvid put skipped: embedding unavailable")
                return None
            frame_ids = self._mem.put_many(
                [{
                    "title": str(ep.get("query") or title or "episode")[:120],
                    "label": str(ep.get("domain") or "general"),
                    "text": _json.dumps(ep, ensure_ascii=False),
                    "metadata": {
                        "domain": str(ep.get("domain") or ""),
                        "tenant": str(ep.get("tenant") or tags.get("tenant") or self.cfg.tenant),
                    },
                }],
                embeddings=[vector],
                embedding_identity={
                    "kind": "manual",
                    "model": self.cfg.embed_model,
                    "dimension": len(vector),
                },
            )
            return str(frame_ids[0]) if frame_ids else None
        except Exception as e:
            log.warning("memvid put failed: %s", e)
            return None

    # -- commit -------------------------------------------------------------
    def commit(self) -> bool:
        try:
            if hasattr(self._mem, "commit"):
                self._mem.commit()
                return True
            return False
        except Exception as e:
            log.warning("memvid commit failed: %s", e)
            return False

    # -- search -------------------------------------------------------------
    def search(self, query: str, top_k: int,
               when: Optional[str] = None) -> List[Dict[str, Any]]:
        """Semantic recall from vectors embedded directly in the MV2 capsule."""
        q_vec = self._embed.embed(query)
        if not q_vec:
            return []
        try:
            result = self._mem.ask(
                query,
                k=top_k,
                mode="semantic",
                context_only=True,
                return_sources=True,
                query_embedding=q_vec,
                query_embedding_model=self.cfg.embed_model,
            )
            hits = result.get("hits", []) if isinstance(result, dict) else []
            out = []
            decoder = json.JSONDecoder()
            for hit in hits:
                text = _attr(hit, "text") or ""
                if not text:
                    continue
                try:
                    payload, _ = decoder.raw_decode(text)
                    out.append({
                        "text": json.dumps(payload, ensure_ascii=False),
                        "title": _attr(hit, "title"),
                        "score": float(_attr(hit, "score") or 0.0),
                        "frame_id": _attr(hit, "frame_id") or _attr(hit, "id"),
                    })
                except (ValueError, TypeError):
                    log.debug("skip malformed native memvid hit")
            return out
        except Exception as e:
            log.debug("native semantic search failed: %s", e)
            return []

    def close(self):
        try:
            if hasattr(self._mem, "close"):
                self._mem.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Embedder — LM Studio (OpenAI-compatible) /v1/embeddings
# ---------------------------------------------------------------------------
class _Embedder:
    def __init__(self, cfg: MemvidConfig):
        self.cfg = cfg
        self._cache: Dict[str, List[float]] = {}
        self._client = None

    def embed(self, text: str) -> Optional[List[float]]:
        if not text:
            return None
        if text in self._cache:
            return self._cache[text]
        try:
            vec = self._embed_http(text)
            if vec:
                self._cache[text] = vec
            return vec
        except Exception as e:
            log.debug("embed failed: %s", e)
            return None

    def _embed_http(self, text: str) -> Optional[List[float]]:
        # Prefer `requests` if available; fall back to urllib.
        # IMPORTANT: LM Studio listens on localhost:1234. If an HTTP(S)
        # proxy is set in the env (common when an LLM proxy runs on
        # 127.0.0.1:12334), requests/urllib route loopback
        # traffic through it and silently fail. Force NO_PROXY for
        # the embed endpoint (matches work-branch curl-embedding fix).
        try:
            import requests  # type: ignore
            r = requests.post(
                self.cfg.embed_url,
                headers={"Authorization": f"Bearer {self.cfg.embed_api_key}",
                         "Content-Type": "application/json"},
                json={"model": self.cfg.embed_model, "input": text},
                timeout=10,
                proxies={"http": None, "https": None},
            )
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]
        except ImportError:
            pass
        import urllib.request
        no_proxy = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(no_proxy)
        req = urllib.request.Request(
            self.cfg.embed_url,
            data=json.dumps(
                {"model": self.cfg.embed_model, "input": text}).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.cfg.embed_api_key}",
                     "Content-Type": "application/json"},
        )
        with opener.open(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["data"][0]["embedding"]


class LMStudioEmbedder:
    """Adapter implementing memvid_sdk.embeddings.EmbeddingProvider.

    Proxies embeddings to the local LM Studio OpenAI-compatible
    /v1/embeddings endpoint (bge-m3, 1024d) so memvid's vec
    search works without a managed embedding backend.
    """

    def __init__(self, cfg: MemvidConfig):
        self._cfg = cfg
        self._http = _Embedder(cfg)
        self._dim: Optional[int] = None

    @property
    def model_name(self) -> str:
        return self._cfg.embed_model

    @property
    def dimension(self) -> int:
        if self._dim is None:
            try:
                v = self._http.embed("dimension-probe")
                self._dim = len(v) if v else 1024
            except Exception:
                self._dim = 1024
        return self._dim

    def embed_query(self, text: str) -> List[float]:
        return self._http.embed(text) or [0.0] * self.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


def _cosine(a: List[float], b: List[float]) -> float:
    import math
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _attr(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------
class MemvidMemory:
    """Episodic memory facade for Hermes.

    One instance per process is fine — capsule handles are cached per tenant
    and guarded by per-tenant RLocks.
    """
    _instances: Dict[str, "MemvidMemory"] = {}
    _instances_lock = threading.Lock()

    def __init__(self, cfg: MemvidConfig):
        self.cfg = cfg
        self._lock = threading.RLock()
        self._backend: _MemvidBackend = self._make_backend(cfg)

    # -- factory ------------------------------------------------------------
    @classmethod
    def for_tenant(cls, tenant: str, **overrides) -> "MemvidMemory":
        """Get or create the singleton MemvidMemory for a tenant."""
        key = tenant
        with cls._instances_lock:
            inst = cls._instances.get(key)
            if inst is None:
                cfg = MemvidConfig.from_env()
                cfg.tenant = tenant
                for k, v in overrides.items():
                    setattr(cfg, k, v)
                inst = cls(cfg)
                cls._instances[key] = inst
            return inst

    @classmethod
    def reset(cls):
        """Drop cached instances (used by tests / canary switches)."""
        with cls._instances_lock:
            for inst in cls._instances.values():
                try:
                    inst._backend.close()
                except Exception:
                    pass
            cls._instances.clear()

    # -- backend wiring -----------------------------------------------------
    def _make_backend(self, cfg: MemvidConfig) -> _MemvidBackend:
        if not cfg.enabled:
            log.info("memvid disabled (RAG_MEMVID_ENABLED=false) -> noop")
            return _NoopMemvidBackend()
        # memvid-sdk 2.x exposes the module as `memvid_sdk`; the
        # legacy package is just `memvid`. Try the SDK first.
        try:
            import memvid_sdk  # noqa: F401  (primary)
        except ImportError:
            try:
                import memvid  # noqa: F401  (legacy fallback)
            except ImportError:
                log.warning("memvid not installed (`pip install memvid-sdk`); "
                            "running in noop mode. RAG flow unaffected.")
                return _NoopMemvidBackend()
        try:
            return _RealMemvidBackend(cfg)
        except Exception as e:
            log.error("memvid backend init failed: %s — using noop", e)
            return _NoopMemvidBackend()

    # -- public API ---------------------------------------------------------
    @property
    def recall_threshold(self) -> float:
        return self.cfg.recall_threshold

    @property
    def active(self) -> bool:
        return not isinstance(self._backend, _NoopMemvidBackend)

    def recall(self, query: str, *, domain: Optional[str] = None,
               top_k: Optional[int] = None,
               when: Optional[str] = None) -> List[Episode]:
        """Recall prior episodes similar to `query`.

        Args:
            query:  the user query / current question.
            domain: optional DCD domain filter (matched against tags).
            top_k:  override RAG_MEMVID_RECALL_TOPK.
            when:   natural-language time filter, e.g. "last tuesday",
                    "yesterday", "before 2024-12-01". Requires
                    RAG_MEMVID_TEMPORAL=true and memvid temporal_track.
        Returns:
            List of Episode sorted by score desc. Empty if disabled /
            no hits / on any error.
        """
        if not self.cfg.do_recall or not query:
            return []
        k = top_k or self.cfg.recall_topk
        t0 = time.perf_counter()
        try:
            with self._lock:
                hits = self._backend.search(query, k, when=when)
        except Exception as e:
            log.warning("recall failed: %s", e)
            return []
        eps: List[Episode] = []
        for h in hits or []:
            try:
                text = _attr(h, "text") or (h.get("text") if isinstance(h, dict) else "")
                if not text:
                    continue
                ep = Episode.from_payload(text)
                ep.score = float(_attr(h, "score") or 0.0)
                ep.frame_id = _attr(h, "frame_id") or _attr(h, "id")
                # domain tag filter (post-filter, memvid tags vary by version)
                if domain and ep.domain and ep.domain != domain:
                    continue
                eps.append(ep)
            except Exception as e:
                log.debug("skip malformed hit: %s", e)
                continue
        eps.sort(key=lambda e: e.score, reverse=True)
        dt = (time.perf_counter() - t0) * 1000
        log.debug("recall q=%r domain=%s when=%s -> %d hits in %.1fms",
                  query[:60], domain, when, len(eps), dt)
        return eps

    def record(self, episode: Episode) -> bool:
        """Persist an episode as a Smart Frame + commit."""
        if not self.cfg.do_record:
            return False
        if not episode or not episode.query:
            return False
        t0 = time.perf_counter()
        try:
            with self._lock:
                tags = {"tenant": episode.tenant or self.cfg.tenant}
                if episode.domain:
                    tags["domain"] = episode.domain
                if episode.feedback:
                    tags["feedback"] = episode.feedback
                uri = f"mv2://hermes/{episode.episode_id}"
                fid = self._backend.put(
                    payload=episode.to_payload(),
                    title=episode.query[:120],
                    tags=tags,
                    uri=uri,
                )
                ok = self._backend.commit()
            dt = (time.perf_counter() - t0) * 1000
            log.debug("record episode=%s frame=%s commit=%s in %.1fms",
                      episode.episode_id, fid, ok, dt)
            return bool(ok)
        except Exception as e:
            log.warning("record failed: %s", e)
            return False

    def recall_as_context(self, query: str, *, domain: Optional[str] = None,
                          top_k: Optional[int] = None,
                          max_chars: int = 1200) -> str:
        """Convenience: format recalled episodes as a prompt prefix.

        Returns empty string if nothing recalled — so the caller can do
        `prefix = memory.recall_as_context(q); prompt = prefix + user_msg`
        without conditional branches.
        """
        eps = self.recall(query, domain=domain, top_k=top_k)
        if not eps:
            return ""
        lines = ["[PRIOR EPISODES — what Hermes already answered before]"]
        total = 0
        for i, ep in enumerate(eps, 1):
            if ep.score < self.cfg.recall_threshold:
                continue
            block = (f"#{i} (score={ep.score:.2f}, {ep.created_at[:10]}, "
                     f"domain={ep.domain or '-'})\n"
                     f"Q: {ep.query}\nA: {ep.answer}")
            if total + len(block) > max_chars:
                break
            lines.append(block)
            total += len(block)
        if len(lines) == 1:
            return ""
        return "\n\n".join(lines) + "\n\n"

    def close(self):
        with self._lock:
            try:
                self._backend.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Smoke test:  python3 memvid_memory.py
# ---------------------------------------------------------------------------
def _smoke():
    logging.basicConfig(level=logging.DEBUG)
    cfg = MemvidConfig.from_env()
    print("config:", cfg)
    m = MemvidMemory(cfg)
    print("active:", m.active, "mode:", cfg.mode)

    ep = Episode(
        query="Как сбросить пароль администратора в Astra Linux?",
        answer="Используйте sudo passwd root в recovery mode...",
        sources=[{"uri": "confluence://AL/123"}],
        trace={"dcd": "astra", "latency_ms": 320},
        domain="astra",
        tenant=cfg.tenant,
    )
    m.record(ep)
    hits = m.recall("сброс пароля astra", domain="astra")
    print(f"recalled {len(hits)} episodes")
    for h in hits:
        print(f"  - score={h.score:.3f}  q={h.query!r}")
    print("context:\n" + m.recall_as_context("сброс пароля astra"))


if __name__ == "__main__":
    _smoke()
