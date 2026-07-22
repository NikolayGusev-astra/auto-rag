# ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](ADR-001-knowledge-gateway.md) | Auto-RAG как локальный offline-capable knowledge gateway для AI-агентов | Proposed | 2026-07-19 |
| [ADR-002](ADR-002-model-runtime.md) | Независимый от LM Studio модельный слой и переносимость индекса | Proposed | 2026-07-19 |
| [ADR-003](ADR-003-adaptive-loop.md) | Сохранение adaptive retrieval loop как optional профиль вокруг unified retrieval core | Proposed | 2026-07-19 |
| [ADR-004](ADR-004-local-workstation-rag.md) | Локальный offline-capable RAG для одного инженера (продуктовая граница после Phase 6) | Proposed | 2026-07-20 |
| [ADR-005](ADR-005-trusted-local-execution-domain.md) | Trusted local execution domain | Proposed | 2026-07-20 |
| [ADR-006](ADR-006-stabilization-before-expansion.md) | Stabilization Before Expansion | Accepted | 2026-07-21 |
| [ADR-007](ADR-007-canonical-dedup.md) | Canonical Dedup in RetrievalCoordinator.fuse() | Accepted | 2026-07-22 |
| [ADR-008](ADR-008-offline-sync.md) | Offline Sync for Live Corporate Connectors | Implemented | 2026-07-22 |
| [ADR-009](ADR-009-allowlisted-fulltext.md) | Full-Text Extraction for Allowlisted Web | Accepted | 2026-07-22 |

## Связанные документы
- [Migration Plan](MIGRATION-PLAN.md) — поэтапный переход от legacy `rag_async` к agent gateway (Фазы 1-5)
- [Migration Plans Index](../plans/adr-migration-index.md) — детальные TDD-планы по фазам (вкл. Phase A-F для ADR-003)
- [ARCHITECTURE.md](ARCHITECTURE.md) — текущее техническое описание (legacy full-RAG pipeline)
- [OPERATIONS.md](OPERATIONS.md) — эксплуатация
- [COMPATIBILITY.md](COMPATIBILITY.md) — матрица совместимости
- [ROLLBACK.md](ROLLBACK.md) — процедура отката

## Статус архитектуры

Reference core (Agent Gateway + Source Connectors + Sync Engine + Retrieval/Fusion) operational:
- **468 тестов, 5 skipped, 1 xfailed**
- 8 live connectors (Jira, Confluence, Lodestone, Allowlisted Web, Hub, ZVec, SearXNG, Local Snapshot)
- Canonical dedup (cross-source), parallel fan-out, BGE-M3 embedding reranker, PDF extraction, Trafilatura full-text
- Offline sync: Jira + Confluence incremental sync_changes with pagination, cursor, 429 backoff
- CI workflow (pytest + ruff + build + wheel install)
- Latency benchmark (p50/p95, parallel fan-out proof)
- MCP stdio gateway для Hermes Agent
