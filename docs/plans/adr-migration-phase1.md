# ADR Migration — Phase 1: Foundation & Contracts

> **For Codex:** Execute task-by-task. Each task = one narrow patch (one defect class / one module). Write the failing test first (RED), run to confirm failure, implement minimal code (GREEN), run to confirm pass, then commit. Do NOT combine multiple tasks into one patch. Windows: use `codex exec --sandbox danger-full-access` if sandbox blocks writes.

**Goal:** Establish the agent-gateway domain models, protocols, and MCP schema docs without altering the existing `rag_async.py` legacy pipeline. All existing tests must stay green.

**Architecture:** New modules under `rag_core/gateway/` (fresh package, no imports from `rag_async`). Domain models are frozen dataclasses. `SourceConnector` is a Protocol; existing ZVec/MCP/Jira adapters get thin wrapper adapters in Phase 2. Model providers are Protocols (ADR-002) but concrete implementations are out of scope for Phase 1 (only the Protocol + capability dataclasses).

**Tech Stack:** Python 3.11+, `pydantic` (already in requirements), `pytest`, `pytest-asyncio`.

**Source of truth:** `docs/ADR-001-knowledge-gateway.md`, `docs/ADR-002-model-runtime.md`, `docs/MIGRATION-PLAN.md`.

---

## Task 1.1: Create `rag_core/gateway/` package + domain models `Document` / `DocumentRef`

**Objective:** Define the immutable `Document` and `DocumentRef` dataclasses from ADR-001 §Data model.

**Files:**
- Create: `rag_core/gateway/__init__.py`
- Create: `rag_core/gateway/models.py`
- Test: `tests/gateway/test_models.py`

**Step 1: Write failing test**

```python
# tests/gateway/test_models.py
import pytest
from dataclasses import FrozenInstanceError
from rag_core.gateway.models import Document, DocumentRef


def test_document_is_frozen_and_has_required_fields():
    doc = Document(
        id="confluence:12345",
        source="confluence",
        source_instance="wiki-prod",
        title="Обновление кластера",
        text="...",
        uri="https://wiki.example/pages/12345",
        version="v3",
        updated_at=None,
        content_hash="abc123",
        metadata={},
    )
    assert doc.id == "confluence:12345"
    assert doc.source == "confluence"
    # frozen -> mutation raises
    with pytest.raises(FrozenInstanceError):
        doc.title = "x"


def test_documentref_identifies_chunk():
    ref = DocumentRef(document_id="confluence:12345", chunk_id="chunk-4")
    assert ref.document_id == "confluence:12345"
    assert str(ref) == "confluence:12345#chunk-4"
```

**Step 2: Run test to verify failure**
Run: `python -m pytest tests/gateway/test_models.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag_core.gateway'`

**Step 3: Write minimal implementation**

```python
# rag_core/gateway/__init__.py
from .models import Document, DocumentRef, SyncBatch, Evidence

__all__ = ["Document", "DocumentRef", "SyncBatch", "Evidence"]
```

```python
# rag_core/gateway/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(frozen=True)
class Document:
    id: str
    source: str
    source_instance: str
    title: str
    text: str
    uri: str | None = None
    version: str | None = None
    updated_at: datetime | None = None
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentRef:
    document_id: str
    chunk_id: str | None = None

    def __str__(self) -> str:
        if self.chunk_id:
            return f"{self.document_id}#{self.chunk_id}"
        return self.document_id
```

**Step 4: Run test to verify pass**
Run: `python -m pytest tests/gateway/test_models.py -q`
Expected: PASS (2 passed)

**Step 5: Commit**
```bash
git add rag_core/gateway/__init__.py rag_core/gateway/models.py tests/gateway/test_models.py
git commit -m "feat(gateway): add Document/DocumentRef frozen models (ADR-001 Phase 1)"
```

---

## Task 1.2: Add `Evidence` (gateway variant) with `origin` enum

**Objective:** Gateway `Evidence` carries `document_id`, `origin` (Literal), `retrieval_score`, `reranker_score`, `updated_at`, `synced_at` — distinct from legacy `rag_core.evidence.Evidence`.

**Files:**
- Modify: `rag_core/gateway/models.py` (add `Evidence`, `EvidenceOrigin`)
- Test: `tests/gateway/test_models.py` (append)

**Step 1: Write failing test** (append to `tests/gateway/test_models.py`)

```python
from rag_core.gateway.models import Evidence, EvidenceOrigin


def test_evidence_has_origin_and_scores():
    ev = Evidence(
        id="confluence:12345#chunk-4",
        document_id="confluence:12345",
        title="Обновление кластера",
        text="...",
        source="confluence",
        uri="https://wiki.example/pages/12345",
        origin=EvidenceOrigin.LOCAL_SNAPSHOT,
        retrieval_score=0.81,
        reranker_score=0.92,
        updated_at=None,
        synced_at=None,
        metadata={},
    )
    assert ev.origin == "local_snapshot"
    assert ev.retrieval_score == 0.81
```

**Step 2: Run to verify failure**
Run: `python -m pytest tests/gateway/test_models.py::test_evidence_has_origin_and_scores -q`
Expected: FAIL — `ImportError: cannot import name 'Evidence'`

**Step 3: Implement** (add to `rag_core/gateway/models.py`)

```python
EvidenceOrigin = Literal["local_snapshot", "live_corporate", "public_web", "agent_memory"]


@dataclass(frozen=True)
class Evidence:
    id: str
    document_id: str
    title: str
    text: str
    source: str
    uri: str | None = None
    origin: EvidenceOrigin = "local_snapshot"
    retrieval_score: float = 0.0
    reranker_score: float | None = None
    updated_at: datetime | None = None
    synced_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Step 4: Run to verify pass**
Run: `python -m pytest tests/gateway/test_models.py -q`
Expected: PASS (3 passed)

**Step 5: Commit**
```bash
git add rag_core/gateway/models.py tests/gateway/test_models.py
git commit -m "feat(gateway): add Evidence + EvidenceOrigin (ADR-001 Phase 1)"
```

---

## Task 1.3: Add `SyncBatch` and `SourceHealth`

**Objective:** Define `SyncBatch` (added/changed/deleted docs, cursor, warnings, stats) and `SourceHealth`.

**Files:**
- Modify: `rag_core/gateway/models.py`
- Test: `tests/gateway/test_models.py` (append)

**Step 1: Write failing test**

```python
from rag_core.gateway.models import SyncBatch, SourceHealth, Document


def test_syncbatch_carries_cursor_and_lists():
    docs = [
        Document(id="jira:1", source="jira", source_instance="jira-prod",
                 title="t", text="x", content_hash="h1"),
    ]
    batch = SyncBatch(
        added=docs, changed=[], deleted=["jira:0"],
        cursor="cur-42", warnings=[], stats={"added": 1},
    )
    assert batch.cursor == "cur-42"
    assert len(batch.added) == 1
    assert batch.deleted == ["jira:0"]


def test_source_health_available_flag():
    h = SourceHealth(source="jira", available=True, detail="ok")
    assert h.available is True
```

**Step 2: Run to verify failure**
Expected: FAIL — import error

**Step 3: Implement** (add to `models.py`)

```python
@dataclass(frozen=True)
class SyncBatch:
    added: list[Document] = field(default_factory=list)
    changed: list[Document] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    cursor: str | None = None
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceHealth:
    source: str
    available: bool
    detail: str = ""
```

**Step 4: Run to verify pass**
Run: `python -m pytest tests/gateway/test_models.py -q`
Expected: PASS (5 passed)

**Step 5: Commit**
```bash
git add rag_core/gateway/models.py tests/gateway/test_models.py
git commit -m "feat(gateway): add SyncBatch + SourceHealth (ADR-001 Phase 1)"
```

---

## Task 1.4: Define `SourceConnector` Protocol

**Objective:** Protocol with `search_live`, `fetch`, `sync_changes`, `health` (async).

**Files:**
- Create: `rag_core/gateway/connector.py`
- Test: `tests/gateway/test_connector.py`

**Step 1: Write failing test**

```python
# tests/gateway/test_connector.py
from rag_core.gateway.connector import SourceConnector, SearchRequest


def test_searchrequest_has_query_and_defaults():
    req = SearchRequest(query="кластер", topk=5)
    assert req.query == "кластер"
    assert req.topk == 5
    assert req.include_web is False


def test_sourceconnector_is_protocol():
    import inspect
    assert hasattr(SourceConnector, "search_live")
    assert hasattr(SourceConnector, "fetch")
    assert hasattr(SourceConnector, "sync_changes")
    assert hasattr(SourceConnector, "health")
```

**Step 2: Run to verify failure**
Expected: FAIL — module not found

**Step 3: Implement**

```python
# rag_core/gateway/connector.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SearchRequest:
    query: str
    topk: int = 5
    domain: str | None = None
    collection: str | None = None
    include_web: bool = False
    continuation_token: str | None = None


@runtime_checkable
class SourceConnector(Protocol):
    source: str

    async def search_live(self, request: SearchRequest) -> list:
        ...

    async def fetch(self, ref) -> object:
        ...

    async def sync_changes(self, cursor: str | None) -> object:
        ...

    async def health(self) -> object:
        ...
```

**Step 4: Run to verify pass**
Run: `python -m pytest tests/gateway/test_connector.py -q`
Expected: PASS (2 passed)

**Step 5: Commit**
```bash
git add rag_core/gateway/connector.py tests/gateway/test_connector.py
git commit -m "feat(gateway): add SourceConnector Protocol + SearchRequest (ADR-001 Phase 1)"
```

---

## Task 1.5: Define Model Provider Protocols (ADR-002)

**Objective:** `EmbeddingProvider`, `RerankerProvider`, `LanguageModelProvider` Protocols + capability dataclasses + `EmbeddingProfile`.

**Files:**
- Create: `rag_core/gateway/model_providers.py`
- Test: `tests/gateway/test_model_providers.py`

**Step 1: Write failing test**

```python
# tests/gateway/test_model_providers.py
from rag_core.gateway.model_providers import (
    EmbeddingProvider, RerankerProvider, LanguageModelProvider,
    EmbeddingProfile, EmbeddingCapabilities,
)


def test_embedding_profile_is_frozen():
    p = EmbeddingProfile(
        provider_family="sentence-transformers",
        model_id="intfloat/multilingual-e5-base",
        model_revision="abc123",
        dimension=768, normalized=True,
        distance_metric="cosine", preprocessing_revision="query-passages-v1",
    )
    assert p.dimension == 768
    assert p.normalized is True


def test_providers_are_runtime_checkable():
    assert hasattr(EmbeddingProvider, "embed_query")
    assert hasattr(RerankerProvider, "rerank")
    assert hasattr(LanguageModelProvider, "complete")
```

**Step 2: Run to verify failure**
Expected: FAIL — module not found

**Step 3: Implement**

```python
# rag_core/gateway/model_providers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class ModelCapabilities:
    provider_id: str
    model_id: str
    revision: str | None
    local: bool
    offline_capable: bool
    max_batch_size: int
    max_input_tokens: int | None = None


@dataclass(frozen=True)
class EmbeddingCapabilities(ModelCapabilities):
    dimension: int
    normalized: bool
    similarity_metric: str


@dataclass(frozen=True)
class EmbeddingProfile:
    provider_family: str
    model_id: str
    model_revision: str | None
    dimension: int
    normalized: bool
    distance_metric: str
    preprocessing_revision: str


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def capabilities(self) -> EmbeddingCapabilities: ...
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class RerankerProvider(Protocol):
    async def rerank(self, query: str, evidence: Sequence[Any], limit: int) -> list[Any]: ...


@runtime_checkable
class LanguageModelProvider(Protocol):
    async def complete(self, request: Any) -> Any: ...
```

**Step 4: Run to verify pass**
Run: `python -m pytest tests/gateway/test_model_providers.py -q`
Expected: PASS (2 passed)

**Step 5: Commit**
```bash
git add rag_core/gateway/model_providers.py tests/gateway/test_model_providers.py
git commit -m "feat(gateway): add model provider Protocols + EmbeddingProfile (ADR-002 Phase 1)"
```

---

## Task 1.6: Document MCP schemas (`docs/mcp-schema.md`)

**Objective:** Write the MCP tool schemas for `search`, `fetch`, `sync`, `sync_status`, `list_sources`, `source_status` as JSON Schema / request-response examples. This is docs only — no code.

**Files:**
- Create: `docs/mcp-schema.md`

**Content:** For each tool, specify:
- transport: MCP stdio
- request params (JSON)
- response shape (matches `Evidence` / `SyncBatch` from Task 1.2-1.3)
- error shape (diagnostic `tool_error` with `code` + `message`, no credentials leaked)
- `search` response MUST include `runtime` observability block (ADR-002 §Observability)

**Step 1-4:** N/A (docs). Write the file, then:
```bash
git add docs/mcp-schema.md
git commit -m "docs(gateway): MCP tool schemas for agent gateway (ADR-001 Phase 1)"
```

---

## Phase 1 Verification Gate

Run full suite to confirm no regression:
```bash
python -m pytest tests -q
```
Expected: **187 passed, 4 skipped, 1 xfailed** (baseline) + new gateway tests green.

If any pre-existing test breaks, STOP and report — do not modify legacy tests to force green.

**Phase 1 exit criteria (from ADR-001 §Критерии принятия, subset):**
- [ ] Domain models importable
- [ ] Protocols documented & importable
- [ ] MCP schemas written
- [ ] Existing pipeline untouched, all legacy tests green

→ Proceed to Phase 2 only after this gate is green.
