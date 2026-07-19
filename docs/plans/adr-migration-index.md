# ADR Migration Plans Index

Планы поэтапной миграции Auto-RAG из legacy "local-first full RAG pipeline" в "local
offline-capable knowledge gateway for AI agents" (ADR-001, ADR-002, ADR-003).

Каждый план написан в формате `writing-plans`: bite-sized субтаски (2-5 мин), TDD
(RED→GREEN), explicit file paths, точные команды. Предназначен для исполнения через Codex
(узкие патчи, один defect class за раз).

| Фаза | Файл | Содержание | Зависит от |
|------|------|-----------|-----------|
| 1 | [adr-migration-phase1.md](adr-migration-phase1.md) | Foundation & Contracts: domain models (Document, Evidence, SyncBatch), SourceConnector Protocol, Model Provider Protocols, MCP schema docs | — |
| 2 | [adr-migration-phase2.md](adr-migration-phase2.md) | Agent Gateway MVP: ZvecConnector adapter, RetrievalCoordinator (dedup/filter/rerank), MCP `search` handler, source availability | Phase 1 |
| 2.5 | [adr-migration-phase2.5.md](adr-migration-phase2.5.md) | Model Runtime & Index Compatibility (ADR-002): index manifest, compatibility gate, CPU/OpenAI providers, no-LLM/lexical fallback, cloud policy, portability tests, staged re-embedding | Phase 1 |
| 3 | [adr-migration-phase3.md](adr-migration-phase3.md) | Sync Engine: staged write, tombstones, atomic publish + integrity, resume cursor, sync_status (incl. Task 3.5 RevisionPublisher) | Phase 1-2 |
| 4 | [adr-migration-phase4.md](adr-migration-phase4.md) | Scope Reduction: MemoryConnector (no short-circuit), web opt-in, federation experimental, rag_async legacy mark | Phase 1-3 |
| 5 | [adr-migration-phase5.md](adr-migration-phase5.md) | Decomposition & Integration: split rag_async, full MCP stdio server, CLI, agent integration (Hermes+Codex), CPU scheduler | Phase 1-4 |
| A–F | [adr-migration-phaseA-F.md](adr-migration-phaseA-F.md) | Adaptive Retrieval Loop (ADR-003): QueryPlan/RoutingFeedback/MemoryEpisode contracts, DcdPlanner, fusion final_score, FeedbackStore, MemvidEnricher, AdaptiveLoop (reference path preserved) | Phase 1, 2, 4 |
| 6 | [adr-migration-phase6-audit-fixes.md](adr-migration-phase6-audit-fixes.md) | **Audit remediation:** P0-1 non-destructive incremental sync, P0-2 real MCP transport, P1-1 plan-driven coordinator + origin/availability/topk, P1-2 unified manifest store, P1-3 sync builds index (chunk/embed), P1-4 persistent feedback + memvid store | Phase 1-5, A-F |

## Как исполнять (Codex)

Для каждой фазы — отдельный `codex exec` вызов с узким промптом на 1-2 субтаска:
- Передай путь к плану + номер Task (например: "Execute docs/plans/adr-migration-phase1.md Task 1.1")
- Codex пишет failing test → RED → implement → GREEN → commit (НЕ push)
- После каждой фазы — локальный `python -m pytest tests -q`, сверить с baseline (187 passed, 4 skipped, 1 xfailed)
- Push в main только по явному "ок" пользователя (standing rule из cursor skill)

## Рекомендованный порядок исполнения

Phase 2.5 зависит только от Phase 1 (типы/Protocol из 1.5). Оптимальный порядок,
уменьшающий переделки (capability negotiation + index compatibility готовы до retrieval):

```
Phase 1 → Phase 2.5 (Tasks 1–8) → Phase 2 → Phase 3 (вкл. Task 3.5) → Phase 4 → Phase 5
```

Task 2.5.8 (staged re-embedding) готовит staged dir + integrity в Phase 2.5; публикация
выполняется единым `RevisionPublisher` (Phase 3 Task 3.5), переиспользующим atomic swap
SyncEngine. Второго механизма публикации нет.

## ADR-002 coverage

До Phase 2.5: ~40-50% (только protocols + CPU scheduler stub). После Phase 2.5: ~95% (manifest,
compatibility gate, CPU + OpenAI-compatible providers, no-LLM/lexical fallback, cloud policy,
portability tests, staged re-embedding prep). Остаток: ONNX concrete impl (stubbed), live LM
Studio end-to-end (integration, env-dependent).

## Verification gates

Каждая фаза заканчивается gate (см. конец файла фазы). Не переходить к следующей, пока gate не green.

## Статус

- [ ] Phase 1 — planned
- [ ] Phase 2.5 — planned (Tasks 1–8, publication deferred to Phase 3 Task 3.5)
- [ ] Phase 2 — planned
- [ ] Phase 3 — planned (incl. Task 3.5 RevisionPublisher)
- [ ] Phase 4 — planned
- [ ] Phase 5 — planned
- [ ] Phase A–F (ADR-003 adaptive loop) — planned (depends on Phase 1, 2, 4)
- [ ] Phase 6 (audit remediation P0/P1) — planned (depends on Phase 1-5, A-F)
