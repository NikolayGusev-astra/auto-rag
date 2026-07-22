# ADR-007: Canonical Dedup in RetrievalCoordinator.fuse()

**Status:** Accepted — implemented in commit @90b463c
**Date:** 2026-07-22
**Extends:** ADR-006 (4.1 — Canonical Document Identity)

## 1. Context

ADR-006 introduced `canonical_document_id` in `Evidence.__post_init__` via `rag_core.gateway.identity.canonical_document_id(source, document_id, metadata)`. The field is computed and populated automatically. Tests exist in `tests/gateway/test_canonical_identity.py` (10 tests).

However, `RetrievalCoordinator.fuse()` deduplicates by `Evidence.document_id`, not by `Evidence.canonical_id`. This means:

- `local_snapshot` returns `Evidence(document_id="sirius-195479")`
- `jira` live connector returns `Evidence(document_id="SIRIUS-195479")`
- Both are the same logical document but occupy **two slots** in top-K

The producer side computes `canonical_id` (ADR-006 §4.1) but the consumer ignores it.

## 2. Problem

When live and snapshot sources return the same document, the fusion layer fails to collapse duplicates. This wastes top-K slots and creates a misleading "more evidence" appearance.

**Audit finding (2026-07-22):** canonical_id exists in the dataclass but fuse() does not read it.

## 3. Decision

**Use `canonical_id` as the dedup key in `RetrievalCoordinator.fuse()`.**

```python
# Before: best_by_document[scored.document_id]
# After:  best_by_canonical[scored.canonical_id or scored.document_id]
```

Fallback to `document_id` when `canonical_id` is None (defensive — should never happen with `__post_init__`).

No other changes: source balancing, score sorting, and `deduplicate_evidence()` remain as-is.

## 4. Consequences

- **Positive:** one logical document = one slot in top-K. Works for all source pairs where `canonical_document_id()` normalizes IDs.
- **Neutral:** provenance is preserved (the higher-scored variant wins, both have `source`/`origin` metadata).
- **Negative:** if `canonical_document_id()` is wrong for a corner case, dedup could collapse *different* documents. Mitigation: the existing 10 tests in `test_canonical_identity.py` verify the function; a single integration test (snapshot + live → same canonical → one output) gates the change.

## 5. Verification

One integration test:
```text
local_snapshot Evidence(document_id="sirius-195479", source="snapshot")
+
jira Evidence(document_id="SIRIUS-195479", source="jira")
→
fuse() outputs 1 item, not 2
```

Existing tests (`test_canonical_identity.py`, `test_coordinator.py`) must still pass.
