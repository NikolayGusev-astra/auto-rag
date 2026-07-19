# ADR Migration — Phase 3: Sync Engine

> **For Codex:** Depends on Phase 1-2. Each task = one narrow patch, TDD. Sync Engine writes to a
**staged** index revision; publish is atomic; failure leaves active index untouched.

**Goal:** Incremental source→local sync with cursor, tombstones, staged atomic publish, resume,
integrity check.

**Architecture:** `rag_core/gateway/sync.py` (SyncEngine: `sync_source(connector, cursor)` → builds
`SyncBatch` → writes staged revision → validates → publishes). Staged revision stored under
`indexes/<profile>/revision-XXXX/`; manifest `indexes/manifest.json` points to active. Active index
read-only during sync.

---

## Task 3.1: SyncBatch persistence to staged revision

**Objective:** SyncEngine writes `SyncBatch` docs to a new staged dir; does not touch active.

**Files:**
- Create: `rag_core/gateway/sync.py`
- Test: `tests/gateway/test_sync.py`

**Step 1: Failing test**

```python
# tests/gateway/test_sync.py
import pytest, tempfile, os
from rag_core.gateway.sync import SyncEngine
from rag_core.gateway.models import Document, SyncBatch


def test_sync_writes_staged_not_active(tmp_path):
    engine = SyncEngine(root=tmp_path)
    docs = [Document(id="jira:1", source="jira", source_instance="p",
                     title="t", text="x", content_hash="h1")]
    batch = SyncBatch(added=docs, cursor="c1")
    rev = engine.stage_sync("jira", batch)
    # staged dir exists, manifest NOT yet pointing to it
    assert os.path.isdir(rev.path)
    assert engine.active_revision("jira") is None
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
# rag_core/gateway/sync.py
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
from rag_core.gateway.models import SyncBatch


class SyncEngine:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def stage_sync(self, source: str, batch: SyncBatch) -> "Revision":
        rev_dir = self.root / source / "staged"
        rev_dir.mkdir(parents=True, exist_ok=True)
        docs_file = rev_dir / "docs.jsonl"
        with open(docs_file, "w", encoding="utf-8") as f:
            for d in batch.added:
                f.write(json.dumps(d.__dict__, default=str) + "\n")
        rev = Revision(path=rev_dir, cursor=batch.cursor)
        return rev

    def active_revision(self, source: str):
        return None  # manifest wiring in 3.4


class Revision:
    def __init__(self, path: Path, cursor: str | None):
        self.path = path
        self.cursor = cursor
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): SyncEngine staged write (ADR-001 Phase
3)`.

---

## Task 3.2: Tombstones / delete propagation

**Objective:** `SyncBatch.deleted` ids written to `tombstones.jsonl`; active query excludes
tombstoned.

**Files:**
- Modify: `rag_core/gateway/sync.py`
- Test: `tests/gateway/test_sync.py` (append)

**Step 1: Failing test**

```python
def test_tombstones_written(tmp_path):
    engine = SyncEngine(root=tmp_path)
    batch = SyncBatch(deleted=["jira:0"], cursor="c2")
    rev = engine.stage_sync("jira", batch)
    tfile = rev.path / "tombstones.jsonl"
    assert tfile.exists()
    lines = tfile.read_text(encoding="utf-8").strip().splitlines()
    assert "jira:0" in lines[0]
```

**Step 2: Run** → FAIL.
**Step 3: Implement** (in `stage_sync`, after docs write):

```python
        if batch.deleted:
            with open(rev_dir / "tombstones.jsonl", "w", encoding="utf-8") as f:
                for did in batch.deleted:
                    f.write(did + "\n")
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): tombstone persistence (ADR-001 Phase 3)`.

---

## Task 3.3: Integrity check + atomic publish

**Objective:** `publish(revision)` validates (docs parse, no corruption) then atomically swaps
manifest pointer. On validation failure → raises, active unchanged.

**Files:**
- Modify: `rag_core/gateway/sync.py`
- Test: `tests/gateway/test_sync.py` (append)

**Step 1: Failing test**

```python
def test_publish_atomic_and_reversible_on_bad(tmp_path):
    engine = SyncEngine(root=tmp_path)
    good = SyncBatch(added=[Document(id="jira:1", source="jira",
                     source_instance="p", title="t", text="x",
                     content_hash="h1")], cursor="c1")
    rev = engine.stage_sync("jira", good)
    engine.publish("jira", rev)
    assert engine.active_revision("jira") == str(rev.path)
    # bad revision: corrupt docs file
    bad = SyncBatch(added=[Document(id="jira:2", source="jira",
                    source_instance="p", title="t", text="x",
                    content_hash="h2")], cursor="c2")
    rev2 = engine.stage_sync("jira", bad)
    (rev2.path / "docs.jsonl").write_text("{broken", encoding="utf-8")
    try:
        engine.publish("jira", rev2)
        assert False, "should have raised"
    except ValueError:
        pass
    # active still points to good
    assert engine.active_revision("jira") == str(rev.path)
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
    def publish(self, source: str, revision: Revision):
        # validate
        docs_file = revision.path / "docs.jsonl"
        if docs_file.exists():
            with open(docs_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        json.loads(line)  # raises on corrupt
        # atomic swap via manifest
        manifest = self.root / source / "manifest.json"
        tmp = self.root / source / "manifest.tmp.json"
        tmp.write_text(json.dumps({"active_index": str(revision.path),
                                   "cursor": revision.cursor}),
                       encoding="utf-8")
        os.replace(tmp, manifest)

    def active_revision(self, source: str):
        manifest = self.root / source / "manifest.json"
        if not manifest.exists():
            return None
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return data.get("active_index")
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): atomic publish + integrity check (ADR-001
Phase 3)`.

---

## Task 3.4: Resume cursor + `sync_status`

**Objective:** `sync_status(source)` returns last cursor + health; `sync_source` accepts `cursor`
and passes to connector.

**Files:**
- Modify: `rag_core/gateway/sync.py`
- Test: `tests/gateway/test_sync.py` (append)

**Step 1: Failing test**

```python
def test_sync_status_returns_cursor(tmp_path):
    engine = SyncEngine(root=tmp_path)
    good = SyncBatch(added=[Document(id="jira:1", source="jira",
                     source_instance="p", title="t", text="x",
                     content_hash="h1")], cursor="c1")
    rev = engine.stage_sync("jira", good)
    engine.publish("jira", rev)
    st = engine.sync_status("jira")
    assert st["cursor"] == "c1"
    assert st["available"] is True
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
    def sync_status(self, source: str) -> dict:
        manifest = self.root / source / "manifest.json"
        if not manifest.exists():
            return {"source": source, "available": False, "cursor": None}
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return {"source": source, "available": True,
                "cursor": data.get("cursor")}
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): sync_status + resume cursor (ADR-001
Phase 3)`.

---

## Task 3.5: Unified `RevisionPublisher` (reindex → atomic publish)

**Objective:** Single atomic-publish entrypoint that consumes a verified staged revision
from `ReindexPlanner` (Phase 2.5 Task 2.5.8) and publishes it via the same manifest-swap
used by `SyncEngine`. This is the ONLY publish path — reindex does not implement its own.

**Depends on:** Phase 2.5 Task 2.5.8 (`ReindexPlanner.build_staged` + `check_integrity`),
Phase 3 Task 3.3 (`SyncEngine.publish` atomic manifest swap).

**Files:**
- Create: `rag_core/gateway/model_runtime/publisher.py`
- Test: `tests/gateway/test_publisher.py`

**Step 1: Failing test**

```python
# tests/gateway/test_publisher.py
import pytest, tempfile
from rag_core.gateway.model_runtime.reindex import ReindexPlanner
from rag_core.gateway.model_runtime.publisher import RevisionPublisher
from rag_core.gateway.model_runtime.manifest import IndexManifest
from rag_core.gateway.model_providers import EmbeddingProfile


def test_publish_verified_staged_revision(tmp_path):
    planner = ReindexPlanner(root=tmp_path)
    prof = EmbeddingProfile("sentence-transformers", "m/e5", "r2", 768,
                            True, "cosine", "q-p-v1")
    rev = planner.build_staged(prof, docs=[{"id": "d1", "text": "x"}])
    assert planner.check_integrity(rev) is True
    pub = RevisionPublisher(root=tmp_path)
    pub.publish(prof, rev)
    manifest = IndexManifest(root=tmp_path)
    assert manifest.profile == prof
    assert manifest.active_revision == str(rev)


def test_publish_rejects_unverified_revision(tmp_path):
    planner = ReindexPlanner(root=tmp_path)
    prof = EmbeddingProfile("sentence-transformers", "m/e5", "r2", 768,
                            True, "cosine", "q-p-v1")
    rev = planner.build_staged(prof, docs=[{"id": "d1", "text": "x"}])
    (rev / "docs.jsonl").write_text("{broken", encoding="utf-8")
    pub = RevisionPublisher(root=tmp_path)
    with pytest.raises(ValueError):
        pub.publish(prof, rev)  # integrity must fail before swap
```

**Step 2: Run** → FAIL (no publisher module).
**Step 3: Implement**

```python
# rag_core/gateway/model_runtime/publisher.py
from __future__ import annotations
from pathlib import Path
from rag_core.gateway.model_runtime.reindex import ReindexPlanner
from rag_core.gateway.model_runtime.manifest import IndexManifest
from rag_core.gateway.model_providers import EmbeddingProfile


class RevisionPublisher:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._planner = ReindexPlanner(root)

    def publish(self, profile: EmbeddingProfile, rev_path: Path):
        # single gate: integrity must pass before any manifest swap
        if not self._planner.check_integrity(rev_path):
            raise ValueError("staged revision failed integrity check; not published")
        manifest = IndexManifest(self.root)
        manifest.write(profile=profile, active_revision=str(rev_path))
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): unified RevisionPublisher (reuses SyncEngine atomic swap, ADR-002 Phase 2.5→3)`.

---

## Phase 3 Verification Gate

```bash
python -m pytest tests/gateway/test_sync.py -q
```
Expected: all sync tests PASS. Active index never corrupted on bad publish.

**Exit criteria (ADR-001 §Критерии, subset):**
- [ ] incremental sync add/update/delete
- [ ] failed sync does not damage active index
- [ ] `sync_status` shows cursor/health
