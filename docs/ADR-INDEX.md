# ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](ADR-001-knowledge-gateway.md) | Auto-RAG как локальный offline-capable knowledge gateway для AI-агентов | Proposed | 2026-07-19 |
| [ADR-002](ADR-002-model-runtime.md) | Независимый от LM Studio модельный слой и переносимость индекса | Proposed | 2026-07-19 |

## Связанные документы
- [Migration Plan](MIGRATION-PLAN.md) — поэтапный переход от legacy `rag_async` к agent gateway (Фазы 1-5)
- [ARCHITECTURE.md](ARCHITECTURE.md) — текущее техническое описание (legacy full-RAG pipeline)
- [OPERATIONS.md](OPERATIONS.md) — эксплуатация

## Статус архитектуры
Текущий код (`rag_async.py`) — legacy/full-RAG profile. Новый reference core (Agent Gateway + Source Connectors + Sync Engine + Local Knowledge Store + Retrieval/Fusion) находится в стадии миграции согласно ADR-001 / ADR-002.
