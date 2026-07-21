# ADR-006: Stabilization Before Expansion

**Status:** Accepted — core retrieval fixes implemented (commits 40ac885 through 65b9804)
**Date:** 2026-07-21
**Extends:** ADR-004 + ADR-005

## 1. Context

Auto-RAG достиг состояния рабочей локальной RAG-платформы для инженера. Ключевые свойства:

* Memvid больше не short-circuit'ит corporate retrieval.
* Jira exact issue fetch включает comments и linked issues.
* Confluence извлекает PDF attachments.
* Web research через allowlisted/policy-controlled pipeline.
* Local snapshot работает offline.
* MCP gateway готов для Hermes Agent.
* Retrieval отделён от generation.
* Система рассчитана на одного инженера / trusted local execution domain.

## 2. Problem

Система достаточно функциональна. Главная задача — доказать ежедневную пользу: точность top-k, отсутствие дублей, скорость, простота запуска, диагностика, демо-сценарии, реальная эксплуатация.

## 3. Decision

**Stabilization Before Expansion.** До завершения стабилизации запрещается добавлять новые крупные архитектурные компоненты. Разрешены только улучшения quality/latency/operability.

## 4. Workstreams

### 4.1 Canonical Document Identity
Один документ может приходить из нескольких источников (jira, local_snapshot, zvec, memvid). Добавить `canonical_document_id` к Evidence. Дубли объединяются, provenance сохраняется.

### 4.2 Exact ID / Title / Slug Boost
Запросы вида `INT-6515`, `639737224`, `ald-pro ALD Pro продукт` должны находить точную сущность. Deterministic boost: exact ID +1.00, slug +0.70, title +0.60, alias +0.40.

### 4.3 Stage-Level Latency
Общее время 7.8s не говорит кто тормозит. Добавить stage timing в diagnostics: planning, memvid, jira, confluence, hub, wiki, local_snapshot, zvec, searxng, trafilatura, camoufox, fusion, dedup, reranking, context_assembly, generation, total.

### 4.4 auto-rag doctor
`auto-rag doctor` / `auto-rag doctor --json`. Проверяет config schema, secret refs, knowledge root, snapshot, manifest, retrieval smoke, models, MCP transport, web policy. Exit codes: 0=ready, 1=config error, 2=snapshot unavailable, 3=degraded. Read-only, не чинит.

### 4.5 Demo Scenarios
- **A — Corporate exact**: `Что известно по INT-6515?` → Jira top-3, comments, linked, no dupes, citations.
- **B — Offline degradation**: Live corporate отключены → gateway стартует, local_snapshot работает, Memvid помогает.
- **C — Web enrichment**: SearXNG discovery → Trafilatura → Camoufox fallback → PUBLIC_WEB origin.

## 5. Retrieval vs Generation

Retrieval метрики: CitationHitRate@5, CitationPrecision@5, Recall@5, MRR, nDCG@5 (null when not computed), Empty rate, duplicate rate, source recall, latency.

Generation метрики: groundedness, answer completeness, citation usage, unsupported claim rate. Базовый локальный generation: 7B instruct (qwen2.5-7b-instruct reference).

## 6. Evaluation Gate

Ranking/fusion changes accepted only if:
- CitationHitRate@5 >= 0.79
- Recall@5 >= 0.79
- MRR >= 0.69
- Empty rate <= 2.0%
- nDCG@5 never displayed as 0.0 when uncomputed.

## 7. Web Policy

- `include_web=False` — per-request kill switch обязателен.
- DCD отключает web для exact corporate identifiers: SIRIUS-*, INT-*, PROJECT-*, ACM-*, AA-*.
- Web — только публичные/внешние вопросы через allowlisted pipeline.

## 8. Memvid Policy

Memvid = episodic hints, offline fallback, personal memory. Не является source of truth, replacement for Jira/Confluence, reason to skip corporate retrieval. Enrichment только после подтверждённых Evidence.

## 9. DCD Policy

DCD auto-learning — ручной. Automatic routing policy update запрещён до конца stabilization period.

## 10. Implementation Order

1. Canonical ID + deduplication + exact ID/title/slug boost + regression eval
2. Stage timers + diagnostics + eval latency report
3. auto-rag doctor + JSON output + exit codes
4. Three demos: corporate exact, offline, web
5. 2 weeks usage / 100 real queries → review

## 11. Stop Conditions

Stabilization ends when: canonical duplicates don't occupy top-5, exact queries stable, stage latency visible, doctor works for minimal profile, 3 demos reproducible, metrics not regressed, 2 weeks/100 queries, no P0/P1 in reference flow.

## 12. Implementation Status (2026-07-21)

| Workstream | Status |
|---|---|
| Jira full-content (comments+linked) | ✅ 40ac885 + 615a0f3 |
| Confluence PDF extraction | ✅ 3161c85 |
| Allowlisted public web | ✅ 65b9804 |
| Lodestone corporate KB | ✅ 615a0f3 |
| Corporate-first routing | ✅ gateway.toml |
| Web disabled by default | ✅ |
| Canonical deduplication | ⬜ Workstream 4.1 |
| Exact ID / slug boost | ⬜ Workstream 4.2 |
| Stage-level latency | ⬜ Workstream 4.3 |
| auto-rag doctor | ⬜ Workstream 4.4 |
| Demo scenarios | ⬜ Workstream 4.5 |
| Chunking before LLM | ⬜ only if proven necessary |
