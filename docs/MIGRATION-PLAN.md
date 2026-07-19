# Migration Plan: Auto-RAG → Agent Knowledge Gateway

**Статус:** Draft (следует из ADR-001, ADR-002)
**Цель:** переориентация с "local-first full RAG pipeline" на "local offline-capable knowledge gateway for AI agents".

Текущий `rag_async` pipeline сохраняется как **legacy/full-RAG profile** до завершения миграции. Новый reference core строится рядом, не ломая существующий.

---

## Фаза 1 — Foundation & Contracts

**Цель:** зафиксировать протоколы и domain models, не меняя существующий pipeline.

- [ ] Зафиксировать ADR-001, ADR-002 (✅ сделано: `docs/ADR-001-knowledge-gateway.md`, `docs/ADR-002-model-runtime.md`)
- [ ] Описать MCP schemas (`search`, `fetch`, `sync`, `sync_status`, `list_sources`, `source_status`) в `docs/mcp-schema.md`
- [ ] Добавить domain models:
  - [ ] `Document` (id, source, source_instance, title, text, uri, version, updated_at, content_hash, metadata)
  - [ ] `DocumentRef`
  - [ ] `SyncBatch` (added/changed/deleted docs, cursor, warnings, stats)
  - [ ] `Evidence` (id, document_id, title, text, source, uri, origin, retrieval_score, reranker_score, updated_at, synced_at, metadata)
  - [ ] `SearchRequest`, `CompletionRequest`, `CompletionResponse`
- [ ] Создать `SourceConnector` Protocol (`search_live`, `fetch`, `sync_changes`, `health`)
- [ ] Создать `EmbeddingProvider` / `RerankerProvider` / `LanguageModelProvider` Protocols (ADR-002)
- [ ] Обернуть существующие Jira/MCP/ZVec реализации адаптерами (adapter, не fork)

**Acceptance:** доменные модели импортируются; протоколы задокументированы; существующие тесты зелёные.

---

## Фаза 2 — Agent Gateway MVP

**Цель:** первый работающий MCP-facing retrieval без генерации ответа.

- [ ] Реализовать MCP server (stdio) с `search` и `fetch`
- [ ] `search` возвращает structured `Evidence[]` (не LLM-ответ)
- [ ] Отключить генерацию финального ответа в agent mode
- [ ] Добавить source availability detection (per-source live/local status)
- [ ] Hybrid lexical+vector retrieval в новом coordinator (не в `rag_async`)
- [ ] Deterministic filters (version/entity), dedup по source ID + content hash
- [ ] Reranker (optional, capability-gated)
- [ ] Agent-friendly pagination / continuation token

**Acceptance:** агент вызывает `search` → получает evidence + URI + freshness; offline mode работает по snapshot; отказ reranker/LLM не блокирует поиск.

---

## Фаза 3 — Sync Engine

**Цель:** жизненный цикл локального снимка.

- [ ] Реализовать `sync` и `sync_status` MCP tools
- [ ] Incremental sync cursor (per source)
- [ ] Tombstones / delete propagation
- [ ] Staged atomic index publish (old index активен до валидации нового)
- [ ] Resume после прерывания
- [ ] Schema version + migration support
- [ ] Integrity check + diagnostics
- [ ] Full rebuild capability

**Acceptance:** sync поддерживает add/update/delete; неуспешная sync не повреждает активный индекс; `sync_status` показывает cursor/health.

---

## Фаза 4 — Scope Reduction (legacy → extension)

**Цель:** убрать не-core механизмы из reference path.

- [ ] Перенести episodic memory в optional `MemoryConnector` (НЕ short-circuit документальный retrieval)
- [ ] Убрать federation из reference pipeline (оставить experimental extension)
- [ ] Web — только explicit opt-in (`include_web=true` или отдельный tool)
- [ ] LLM generation — вне основного pipeline (только query-rewrite/rerank/eval/debug)
- [ ] Пометить `rag_async` как legacy/full-RAG mode
- [ ] Убрать tenant/ACL из product terminology (сохранить поля для обратной совместимости, не делать их обязательными)

**Acceptance:** federation отсутствует в default path; memory не short-circuit'ит; web не запускается без явного флага; reference pipeline не требует tenant/ACL.

---

## Фаза 5 — Decomposition & Agent Integration

**Цель:** разнести монолит и провести интеграционные тесты.

- [ ] Разбить `rag_async.py` (~865 строк) на: gateway / retrieval / fusion / connector-execution
- [ ] Новый gateway = reference implementation
- [ ] Agent integration tests минимум с 2 клиентами (например, Hermes + Codex)
- [ ] CPU-bounded scheduler (приоритет interactive search > fetch > sync > rebuild)
- [ ] Observability block в ответе (runtime profile)

**Acceptance:** `rag_async` больше не обязательный entrypoint для agent mode; интеграционный тест проходит с ≥2 агентами; CPU scheduler отдаёт приоритет интерактивному поиску.

---

## Связь с ADR-002 (Model Runtime)

Фазы 1-5 используют provider-independent model layer:
- EmbeddingProvider обязателен для vector retrieval
- RerankerProvider / LanguageModelProvider — optional capability
- Index manifest хранит `EmbeddingProfile` (dimension, normalized, distance_metric, preprocessing_revision)
- Несовместимая embedding model блокирует vector search с понятным сообщением
- Cloud provider выключен по умолчанию (`disabled`)

## Risks

- Существующий `rag_async` и новый gateway сосуществуют → дублирование логики на время миграции
- Sync adapters сложнее search adapters (cursor/tombstone/staged publish)
- Миграция metadata схемы индекса требует rebuild для старых индексов
- Документация должна чётко разделять legacy и gateway режимы
