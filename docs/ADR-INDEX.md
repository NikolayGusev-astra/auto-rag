# ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](ADR-001-knowledge-gateway.md) | Auto-RAG как локальный offline-capable knowledge gateway для AI-агентов | Proposed | 2026-07-19 |
| [ADR-002](ADR-002-model-runtime.md) | Независимый от LM Studio модельный слой и переносимость индекса | Proposed | 2026-07-19 |
| [ADR-003](ADR-003-adaptive-loop.md) | Сохранение adaptive retrieval loop как optional профиль вокруг unified retrieval core | Proposed | 2026-07-19 |
| [ADR-004](ADR-004-local-workstation-rag.md) | Локальный offline-capable RAG для одного инженера (продуктовая граница после Phase 6) | Proposed | 2026-07-20 |

## Связанные документы
- [Migration Plan](MIGRATION-PLAN.md) — поэтапный переход от legacy `rag_async` к agent gateway (Фазы 1-5)
- [Migration Plans Index](../plans/adr-migration-index.md) — детальные TDD-планы по фазам (вкл. Phase A-F для ADR-003)
- [ARCHITECTURE.md](ARCHITECTURE.md) — текущее техническое описание (legacy full-RAG pipeline)
- [OPERATIONS.md](OPERATIONS.md) — эксплуатация

## Статус архитектуры
Текущий код (`rag_async.py`) — legacy/full-RAG profile. Новый reference core (Agent Gateway + Source Connectors + Sync Engine + Local Knowledge Store + Retrieval/Fusion) находится в стадии миграции согласно ADR-001 / ADR-002.
