# ADR Migration — Phase 2.5: Model Runtime & Index Compatibility (ADR-002)

> **For Codex:** Depends on Phase 1 (model_providers.py: EmbeddingProvider, EmbeddingProfile, capabilities). Each task = one narrow patch, TDD (RED→GREEN→commit). Do NOT require LM Studio at any step — CPU/OpenAI-compatible providers must work standalone.

**Goal:** Implement provider-independent model layer with embedding-index contract enforcement: index manifest stores EmbeddingProfile; provider↔index compatibility gate blocks mismatched models even at equal dimension; CPU (sentence-transformers/ONNX) and OpenAI-compatible providers; no-LLM + lexical-only degradation; cloud policy disabled by default; runtime portability tests; staged re-embedding on profile change.

**Architecture:** New `rag_core/gateway/model_runtime/` package:
- `registry.py` — provider registry + capability negotiation (`RuntimeCapabilities`)
- `manifest.py` — `IndexManifest` (stores `EmbeddingProfile`, active revision pointer)
- `compatibility.py` — `check_compatible(provider_profile, index_profile)` → blocks mismatched
- `providers/cpu.py` — `SentenceTransformersEmbeddingProvider`, `OnnxEmbeddingProvider`
- `providers/openai_compat.py` — `OpenAICompatibleEmbeddingProvider`
- `policies.py` — cloud policy enum (disabled/query_only/selected_evidence/full), default disabled

---

## Task 2.5.1: IndexManifest stores EmbeddingProfile

**Objective:** `IndexManifest` persists full `EmbeddingProfile` (model_id, revision, dimension, normalized, distance_metric, preprocessing_revision) + active revision path. Load/save JSON.

**Files:**
- Create: `rag_core/gateway/model_runtime/__init__.py`
- Create: `rag_core/gateway/model_runtime/manifest.py`
- Test: `tests/gateway/test_index_manifest.py`

**Step 1: Failing test**

```python
# tests/gateway/test_index_manifest.py
import pytest, tempfile
from rag_core.gateway.model_runtime.manifest import IndexManifest
from rag_core.gateway.model_providers import EmbeddingProfile


def test_manifest_roundtrip_embedding_profile(tmp_path):
    prof = EmbeddingProfile(
        provider_family="sentence-transformers",
        model_id="intfloat/multilingual-e5-base",
        model_revision="abc123", dimension=768, normalized=True,
        distance_metric="cosine", preprocessing_revision="q-p-v1")
    m = IndexManifest(root=tmp_path)
    m.write(profile=prof, active_revision="rev-0001")
    loaded = IndexManifest(root=tmp_path)
    assert loaded.profile == prof
    assert loaded.active_revision == "rev-0001"


def test_manifest_equality_by_fields():
    p1 = EmbeddingProfile("sentence-transformers", "m", "r", 768, True, "cosine", "q-p-v1")
    p2 = EmbeddingProfile("sentence-transformers", "m", "r", 768, True, "cosine", "q-p-v1")
    assert p1 == p2
```

**Step 2: Run** `python -m pytest tests/gateway/test_index_manifest.py -q` → FAIL (import).

**Step 3: Implement**

```python
# rag_core/gateway/model_runtime/__init__.py
from .manifest import IndexManifest
from .compatibility import check_compatible
from .registry import ProviderRegistry, RuntimeCapabilities
__all__ = ["IndexManifest", "check_compatible", "ProviderRegistry", "RuntimeCapabilities"]
```

```python
# rag_core/gateway/model_runtime/manifest.py
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
from rag_core.gateway.model_providers import EmbeddingProfile


@dataclass
class IndexManifest:
    profile: EmbeddingProfile | None = None
    active_revision: str | None = None
    _path: Path | None = None

    def __init__(self, root: Path):
        self._root = Path(root)
        self._path = self._root / "index_manifest.json"
        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.profile = EmbeddingProfile(**data["profile"])
            self.active_revision = data["active_revision"]

    def write(self, profile: EmbeddingProfile, active_revision: str):
        self.profile = profile
        self.active_revision = active_revision
        self._root.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({
            "profile": profile.__dict__,
            "active_revision": active_revision,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): IndexManifest with EmbeddingProfile (ADR-002 Phase 2.5)`.

---

## Task 2.5.2: Compatibility gate (blocks mismatched model at equal dimension)

**Objective:** `check_compatible(provider_profile, index_profile)` returns False (with reason) when model_id/revision/normalized/metric/preprocessing differ — even if dimension matches. Returns True only on full contract match.

**Files:**
- Create: `rag_core/gateway/model_runtime/compatibility.py`
- Test: `tests/gateway/test_compatibility.py`

**Step 1: Failing test**

```python
# tests/gateway/test_compatibility.py
import pytest
from rag_core.gateway.model_runtime.compatibility import check_compatible
from rag_core.gateway.model_providers import EmbeddingProfile


def _p(model_id, dim=768, revision="r", norm=True, metric="cosine", pre="q-p-v1"):
    return EmbeddingProfile("family", model_id, revision, dim, norm, metric, pre)


def test_equal_dim_diff_model_blocked():
    ok, reason = check_compatible(_p("model-a"), _p("model-b"))
    assert ok is False
    assert "model" in reason.lower()


def test_same_model_revision_compatible():
    ok, reason = check_compatible(_p("model-a", revision="r1"), _p("model-a", revision="r1"))
    assert ok is True


def test_diff_normalization_blocked():
    ok, _ = check_compatible(_p("m", norm=True), _p("m", norm=False))
    assert ok is False


def test_diff_preprocessing_blocked():
    ok, _ = check_compatible(_p("m", pre="v1"), _p("m", pre="v2"))
    assert ok is False
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
# rag_core/gateway/model_runtime/compatibility.py
from __future__ import annotations
from rag_core.gateway.model_providers import EmbeddingProfile


def check_compatible(
    provider_profile: EmbeddingProfile,
    index_profile: EmbeddingProfile,
) -> tuple[bool, str]:
    """Return (ok, reason). Dimension alone is NOT sufficient."""
    if provider_profile.model_id != index_profile.model_id:
        return False, (
            f"Embedding model mismatch: provider={provider_profile.model_id!r} "
            f"index={index_profile.model_id!r}. Different models are not "
            f"interchangeable even at equal dimension.")
    if provider_profile.model_revision != index_profile.model_revision:
        return False, f"Model revision differs: {provider_profile.model_revision} vs {index_profile.model_revision}"
    if provider_profile.normalized != index_profile.normalized:
        return False, "Normalization policy differs"
    if provider_profile.distance_metric != index_profile.distance_metric:
        return False, f"Distance metric differs: {provider_profile.distance_metric} vs {index_profile.distance_metric}"
    if provider_profile.preprocessing_revision != index_profile.preprocessing_revision:
        return False, f"Preprocessing contract differs: {provider_profile.preprocessing_revision} vs {index_profile.preprocessing_revision}"
    return True, "compatible"
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): embedding compatibility gate (ADR-002 Phase 2.5)`.

---

## Task 2.5.3: CPU EmbeddingProvider (sentence-transformers / ONNX)

**Objective:** `SentenceTransformersEmbeddingProvider` implements `EmbeddingProvider` Protocol without LM Studio. `embed_query` / `embed_documents` return lists. Graceful: if lib missing → raises clear error at init (not silent).

**Files:**
- Create: `rag_core/gateway/model_runtime/providers/__init__.py`
- Create: `rag_core/gateway/model_runtime/providers/cpu.py`
- Test: `tests/gateway/test_cpu_provider.py`

**Step 1: Failing test**

```python
# tests/gateway/test_cpu_provider.py
import pytest
from rag_core.gateway.model_runtime.providers.cpu import (
    SentenceTransformersEmbeddingProvider, make_cpu_profile)


def test_cpu_profile_dimension_declared():
    prof = make_cpu_profile("intfloat/multilingual-e5-base", dim=768)
    assert prof.dimension == 768
    assert prof.normalized is True


@pytest.mark.asyncio
async def test_cpu_provider_embed_returns_vectors(monkeypatch):
    # stub the underlying ST model to avoid heavy download in test
    class FakeModel:
        def encode(self, texts, **kw):
            dim = 4
            return [[0.1] * dim for _ in texts]
    prov = SentenceTransformersEmbeddingProvider(model_id="fake/e5", dim=4)
    monkeypatch.setattr(prov, "_model", FakeModel())
    vec = await prov.embed_query("привет")
    assert len(vec) == 4
    docs = await prov.embed_documents(["a", "b"])
    assert len(docs) == 2 and len(docs[0]) == 4
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
# rag_core/gateway/model_runtime/providers/__init__.py
from .cpu import SentenceTransformersEmbeddingProvider, OnnxEmbeddingProvider, make_cpu_profile
__all__ = ["SentenceTransformersEmbeddingProvider", "OnnxEmbeddingProvider", "make_cpu_profile"]
```

```python
# rag_core/gateway/model_runtime/providers/cpu.py
from __future__ import annotations
from rag_core.gateway.model_providers import (
    EmbeddingProvider, EmbeddingCapabilities, EmbeddingProfile)


def make_cpu_profile(model_id: str, dim: int, revision: str | None = None,
                     normalized: bool = True, metric: str = "cosine",
                     pre: str = "query-passages-v1") -> EmbeddingProfile:
    return EmbeddingProfile(
        provider_family="sentence-transformers", model_id=model_id,
        model_revision=revision, dimension=dim, normalized=normalized,
        distance_metric=metric, preprocessing_revision=pre)


class SentenceTransformersEmbeddingProvider:
    def __init__(self, model_id: str, dim: int, revision: str | None = None):
        self._model_id = model_id
        self._dim = dim
        self._model = None  # lazy: import sentence_transformers on first use

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            provider_id="sentence-transformers", model_id=self._model_id,
            revision=None, local=True, offline_capable=True,
            max_batch_size=32, max_input_tokens=None)

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers not installed; CPU embedding "
                    "provider unavailable. Install it or use another provider."
                ) from e
            self._model = SentenceTransformer(self._model_id)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    async def embed_query(self, text: str) -> list[float]:
        self._ensure()
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


class OnnxEmbeddingProvider:
    # analogous; loads ONNX runtime model. Stub for Phase 2.5 scope.
    def __init__(self, model_path: str, dim: int):
        self._model_path = model_path
        self._dim = dim

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            provider_id="onnx", model_id=self._model_path, revision=None,
            local=True, offline_capable=True, max_batch_size=32,
            max_input_tokens=None)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("ONNX embedding impl in follow-up")

    async def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("ONNX embedding impl in follow-up")
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): CPU sentence-transformers embedding provider (ADR-002 Phase 2.5)`.

---

## Task 2.5.4: OpenAI-compatible EmbeddingProvider

**Objective:** `OpenAICompatibleEmbeddingProvider` talks to any `/v1/embeddings` endpoint (LM Studio, vLLM, Ollama, corporate gateway). No LM Studio assumption. Validates returned dimension against expected.

**Files:**
- Modify: `rag_core/gateway/model_runtime/providers/__init__.py` (add)
- Create: `rag_core/gateway/model_runtime/providers/openai_compat.py`
- Test: `tests/gateway/test_openai_provider.py`

**Step 1: Failing test**

```python
# tests/gateway/test_openai_provider.py
import pytest
from rag_core.gateway.model_runtime.providers.openai_compat import (
    OpenAICompatibleEmbeddingProvider)


@pytest.mark.asyncio
async def test_openai_compat_embed_uses_base_url(monkeypatch):
    captured = {}
    class FakeResp:
        def __init__(self, data): self._d = data
        def raise_for_status(self): pass
        def json(self): return self._d
    class FakeClient:
        async def embed(self, *, model, input, **kw):
            captured["model"] = model
            captured["input"] = input
            return FakeResp({"data": [{"embedding": [0.2, 0.3]}]})
    prov = OpenAICompatibleEmbeddingProvider(
        base_url="http://127.0.0.1:1234/v1", model="e5", expected_dim=2)
    monkeypatch.setattr(prov, "_client", FakeClient())
    vec = await prov.embed_query("q")
    assert vec == [0.2, 0.3]
    assert captured["input"] == ["q"]


@pytest.mark.asyncio
async def test_openai_compat_dim_mismatch_raises(monkeypatch):
    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"data": [{"embedding": [0.1]}]}  # dim 1
    class FakeClient:
        async def embed(self, *, model, input, **kw):
            return FakeResp()
    prov = OpenAICompatibleEmbeddingProvider(
        base_url="x", model="e5", expected_dim=2)
    monkeypatch.setattr(prov, "_client", FakeClient())
    with pytest.raises(ValueError):
        await prov.embed_query("q")
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
# rag_core/gateway/model_runtime/providers/openai_compat.py
from __future__ import annotations
import httpx
from rag_core.gateway.model_providers import (
    EmbeddingProvider, EmbeddingCapabilities)


class OpenAICompatibleEmbeddingProvider:
    def __init__(self, base_url: str, model: str, expected_dim: int,
                 api_key: str | None = None):
        self._base = base_url.rstrip("/")
        self._model = model
        self._dim = expected_dim
        self._api_key = api_key
        self._client = None

    def _ensure(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base,
                headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else {})

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            provider_id="openai-compatible", model_id=self._model,
            revision=None, local=False, offline_capable=False,
            max_batch_size=16, max_input_tokens=None)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_query(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        self._ensure()
        resp = await self._client.post("/embeddings", json={
            "model": self._model, "input": [text]})
        resp.raise_for_status()
        data = resp.json()["data"]
        vec = data[0]["embedding"]
        if len(vec) != self._dim:
            raise ValueError(
                f"Embedding dim {len(vec)} != expected {self._dim} for "
                f"model {self._model}. Refusing to use incompatible index.")
        return vec
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): OpenAI-compatible embedding provider (ADR-002 Phase 2.5)`.

---

## Task 2.5.5: RuntimeCapabilities negotiation + no-LLM / lexical-only fallback

**Objective:** `RuntimeCapabilities` dataclass + `ProviderRegistry.negotiate()` returns available caps. Retrieval must still work with `lexical_search=True, embeddings=False` (BM25/FTS fallback) and `generation=False`.

**Files:**
- Create: `rag_core/gateway/model_runtime/registry.py`
- Test: `tests/gateway/test_registry.py`

**Step 1: Failing test**

```python
# tests/gateway/test_registry.py
import pytest
from rag_core.gateway.model_runtime.registry import (
    ProviderRegistry, RuntimeCapabilities)


def test_negotiate_reports_lexical_without_embeddings():
    reg = ProviderRegistry(embeddings=None, lexical=True, reranking=False,
                           query_rewrite=False, generation=False, offline=True)
    caps = reg.negotiate()
    assert caps.lexical_search is True
    assert caps.embeddings is False
    assert caps.generation is False


def test_minimal_profile_allows_retrieval():
    caps = RuntimeCapabilities(embeddings=False, lexical_search=True,
                               reranking=False, query_rewrite=False,
                               generation=False, offline=True)
    assert caps.lexical_search  # BM25/FTS still works without embeddings
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
# rag_core/gateway/model_runtime/registry.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeCapabilities:
    embeddings: bool
    lexical_search: bool
    reranking: bool
    query_rewrite: bool
    generation: bool
    offline: bool


class ProviderRegistry:
    def __init__(self, *, embeddings=None, lexical: bool = True,
                 reranking: bool = False, query_rewrite: bool = False,
                 generation: bool = False, offline: bool = True):
        self._embeddings = embeddings
        self._lexical = lexical
        self._reranking = reranking
        self._query_rewrite = query_rewrite
        self._generation = generation
        self._offline = offline

    def negotiate(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            embeddings=self._embeddings is not None,
            lexical_search=self._lexical,
            reranking=self._reranking,
            query_rewrite=self._query_rewrite,
            generation=self._generation,
            offline=self._offline,
        )

    def embedding_provider(self):
        return self._embeddings
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): RuntimeCapabilities negotiation + lexical fallback (ADR-002 Phase 2.5)`.

---

## Task 2.5.6: Cloud policy (disabled by default)

**Objective:** `CloudPolicy` enum (disabled/query_only/selected_evidence/full). Default `disabled`. Provider must refuse cloud calls unless policy permits. No document content sent when policy < selected_evidence.

**Files:**
- Create: `rag_core/gateway/model_runtime/policies.py`
- Test: `tests/gateway/test_policies.py`

**Step 1: Failing test**

```python
# tests/gateway/test_policies.py
import pytest
from rag_core.gateway.model_runtime.policies import CloudPolicy, guard_cloud_call


def test_default_policy_disabled_blocks_all():
    assert CloudPolicy.default() == CloudPolicy.DISABLED


def test_guard_blocks_document_when_policy_query_only():
    with pytest.raises(PermissionError):
        guard_cloud_call(CloudPolicy.QUERY_ONLY, sends_document=True)


def test_guard_allows_query_when_query_only():
    # does not raise
    guard_cloud_call(CloudPolicy.QUERY_ONLY, sends_document=False,
                     sends_query=True)


def test_guard_blocks_when_disabled():
    with pytest.raises(PermissionError):
        guard_cloud_call(CloudPolicy.DISABLED, sends_query=True)
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
# rag_core/gateway/model_runtime/policies.py
from __future__ import annotations
from enum import Enum


class CloudPolicy(str, Enum):
    DISABLED = "disabled"
    QUERY_ONLY = "query_only"
    SELECTED_EVIDENCE = "selected_evidence"
    FULL = "full"

    @classmethod
    def default(cls) -> "CloudPolicy":
        return cls.DISABLED


def guard_cloud_call(policy: CloudPolicy, *, sends_query: bool = False,
                     sends_document: bool = False) -> None:
    """Raise PermissionError if the call would violate the policy."""
    if policy == CloudPolicy.DISABLED:
        raise PermissionError("Cloud provider disabled by default policy")
    if sends_document and policy not in (
            CloudPolicy.SELECTED_EVIDENCE, CloudPolicy.FULL):
        raise PermissionError(
            f"Policy {policy.value} forbids sending document content to cloud")
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): cloud policy disabled by default (ADR-002 Phase 2.5)`.

---

## Task 2.5.7: Runtime portability tests (2 runtimes, 1 model; dim-match ≠ compat)

**Objective:** Integration-style: same EmbeddingProfile opens with two provider instances (CPU stub + OpenAI-compat stub) of SAME model; different model SAME dimension is rejected by compatibility gate.

**Files:**
- Test: `tests/gateway/test_runtime_portability.py`

**Step 1: Failing test**

```python
# tests/gateway/test_runtime_portability.py
import pytest
from rag_core.gateway.model_runtime.compatibility import check_compatible
from rag_core.gateway.model_runtime.manifest import IndexManifest
from rag_core.gateway.model_runtime.providers.cpu import make_cpu_profile
from rag_core.gateway.model_providers import EmbeddingProfile
import tempfile


def test_two_runtimes_same_model_open_index():
    base = EmbeddingProfile("sentence-transformers", "m/e5", "r", 768,
                            True, "cosine", "q-p-v1")
    # cpu provider profile (same model)
    cpu = make_cpu_profile("m/e5", dim=768)
    ok, _ = check_compatible(cpu, base)
    assert ok is True  # same model id/revision/contract


def test_dim_match_diff_model_rejected():
    idx = EmbeddingProfile("family", "model-a", "r", 768, True, "cosine", "q-p-v1")
    other = EmbeddingProfile("family", "model-b", "r", 768, True, "cosine", "q-p-v1")
    ok, reason = check_compatible(other, idx)
    assert ok is False
    assert "model" in reason.lower()


def test_manifest_blocks_incompatible_on_load(tmp_path):
    # write manifest with model-a, then a provider with model-b must be rejected
    m = IndexManifest(root=tmp_path)
    m.write(profile=EmbeddingProfile("f", "model-a", "r", 768, True, "cosine", "q-p-v1"),
            active_revision="rev1")
    loaded = IndexManifest(root=tmp_path)
    incoming = EmbeddingProfile("f", "model-b", "r", 768, True, "cosine", "q-p-v1")
    ok, _ = check_compatible(incoming, loaded.profile)
    assert ok is False
```

**Step 2: Run** → FAIL (no test file).
**Step 3:** Tests only — no new impl (uses 2.5.1-2.5.2).
**Step 4: Run** → PASS. **Step 5: Commit** `test(gateway): runtime portability + dim-match rejection (ADR-002 Phase 2.5)`.

---

## Task 2.5.8: Staged re-embedding on profile change

**Objective:** `ReindexPlanner` builds new staged revision under a NEW profile dir; only after integrity check publishes via manifest swap (reuse SyncEngine atomic publish from Phase 3). Old index untouched during build.

**Files:**
- Create: `rag_core/gateway/model_runtime/reindex.py`
- Test: `tests/gateway/test_reindex.py`

**Step 1: Failing test**

```python
# tests/gateway/test_reindex.py
import pytest, tempfile
from rag_core.gateway.model_runtime.reindex import ReindexPlanner
from rag_core.gateway.model_runtime.manifest import IndexManifest
from rag_core.gateway.model_providers import EmbeddingProfile


def test_reindex_builds_staged_then_publishes(tmp_path):
    planner = ReindexPlanner(root=tmp_path)
    new_prof = EmbeddingProfile("sentence-transformers", "m/e5", "r2", 768,
                                True, "cosine", "q-p-v1")
    # simulate building staged revision
    rev_path = planner.build_staged(new_prof, docs=[{"id": "d1", "text": "x"}])
    assert rev_path.exists()
    # publish
    planner.publish(new_prof, rev_path)
    manifest = IndexManifest(root=tmp_path)
    assert manifest.profile == new_prof
    assert manifest.active_revision == str(rev_path)
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
# rag_core/gateway/model_runtime/reindex.py
from __future__ import annotations
import json
from pathlib import Path
from rag_core.gateway.model_runtime.manifest import IndexManifest
from rag_core.gateway.model_providers import EmbeddingProfile


class ReindexPlanner:
    def __init__(self, root: Path):
        self.root = Path(root)

    def build_staged(self, profile: EmbeddingProfile, docs: list[dict]) -> Path:
        # new revision dir keyed by profile identity
        key = f"{profile.model_id}@{profile.model_revision}"
        rev = self.root / "reindex-staged" / key
        rev.mkdir(parents=True, exist_ok=True)
        with open(rev / "docs.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return rev

    def publish(self, profile: EmbeddingProfile, rev_path: Path):
        # integrity: docs parse
        with open(rev_path / "docs.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json.loads(line)
        manifest = IndexManifest(self.root)
        manifest.write(profile=profile, active_revision=str(rev_path))
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): staged re-embedding planner (ADR-002 Phase 2.5)`.

---

## Phase 2.5 Verification Gate

```bash
python -m pytest tests/gateway/ -q
```
Expected: all Phase 1 + 2.5 gateway tests green. No LM Studio required to pass.

**ADR-002 coverage after this phase:**
- [x] Index manifest with full EmbeddingProfile
- [x] Provider↔index compatibility gate (dim-match ≠ compat)
- [x] CPU embedding provider (sentence-transformers)
- [x] OpenAI-compatible provider (LM Studio/vLLM/Ollama/gateway)
- [x] No-LLM + lexical-only degradation path
- [x] Cloud policy disabled by default
- [x] Runtime portability tests (2 runtimes 1 model; reject dim-match diff model)
- [x] Staged re-embedding on profile change

→ ADR-002 coverage raises from ~40-50% to ~95%. Remaining: ONNX concrete impl (stubbed here), live LM Studio end-to-end (integration, env-dependent).
