# ADR-006: Stabilization Before Expansion
* **Status:** Proposed
* **Date:** 2026-07-21
* **Extends:** ADR-004 + ADR-005

## Decision
Auto-RAG enters **stabilization before expansion**.
No new subsystems unless driven by real defects.
Only 5 workstreams permitted:

1. Canonical document deduplication
2. Exact ID/title/slug/entity boosting
3. Stage-level latency metrics
4. auto-rag doctor
5. Reproducible demo scenarios

## Implementation order
1. Identity & Ranking (canonical ID → dedup → boost)
2. Measurements (stage timers → eval latency)
3. Operations (doctor + CLI)
4. Demos (corporate, offline, web)
5. Usage (2 weeks / 100 real queries)

## Evaluation gate
CitationHitRate@5 >= 0.79
Recall@5 >= 0.79
MRR >= 0.69
Empty rate <= 2.0%

nDCG@5: null when not computed (not 0.0)

## Stop conditions
1. No canonical duplicates in top-5
2. Known ID/slug/title queries hit top-3
3. Latency breakdown per request
4. auto-rag doctor checks minimal profile
5. 3 demo scenarios reproducible
6. Golden metrics not degraded
7. 2 weeks / 100 real queries
8. No P0/P1 in reference flow

Full text: docs/ADR-006-stabilization-before-expansion.md