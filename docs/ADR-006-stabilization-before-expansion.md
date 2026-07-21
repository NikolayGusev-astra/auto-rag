# ADR-006: Stabilization Before Expansion

**Status:** Accepted — core retrieval fixes implemented (commits 40ac885 through 36a0ab5)
**Date:** 2026-07-21
**Extends:** ADR-004 + ADR-005

## 1. Context

Auto-RAG достиг состояния рабочей локальной RAG-платформы для инженера.

Ключевые свойства:

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

### 4.1 Canonical Document Identity + Exact ID/Slug Boost
Один документ может приходить из нескольких источников. `canonical_document_id` дедуплицирует дубли, provenance сохраняется. Exact ID/slug/title получают deterministic boost.

**Tests:** `tests/gateway/test_canonical_identity.py` (10 tests)
**Commit:** `25a8c9d`

### 4.2 Stage-Level Latency
`StageTimer` в coordinator записывает timing per connector + fusion + reranking. Видно в diagnostics.

**Tests:** coordinator и adaptive_loop тесты
**Commit:** `8580c50`

### 4.3 auto-rag doctor
`doctor.py` — read-only проверка: config, secrets, knowledge root, snapshot, retrieval smoke, models. JSON output, exit codes (0/1/2/3).

**Tests:** `tests/gateway/test_doctor.py` (6 tests: offline, config error, snapshot missing, Jira unavailable, JSON, readonly)
**Commit:** `fb67d6a`

### 4.4 Demo Scenarios
- **A — Corporate exact**: `demo.py` — Jira exact key → full comments + linked
- **B — Offline degradation**: `demo.py` — gateway with live corporate disabled
- **C — Web enrichment**: SearXNG → Trafilatura → Camoufox fallback → PUBLIC_WEB

**Commit:** `e49cbf8`

### 4.5 Core Retrieval Fixes (post-decision implementation)
Дополнительные исправления, обнаруженные при живой регрессии SIRIUS-195479:

- Jira paginated comments + linked issue content + enrichment diagnostics (`40ac885`, `615a0f3`)
- Confluence PDF attachment extraction (`3161c85`)
- Allowlisted public web retrieval (`65b9804`)
- Lodestone corporate KB integration (`615a0f3`)
- Corporate-first routing policy (`gateway.toml`)

## 5. Implementation Status

| Capability          |     Decision | Implementation |                  Validation | Commit   |
| ------------------- | ------------ | -------------- | --------------------------- | -------- |
| Canonical dedup     |     Accepted |    Implemented |            10 automated tests | `25a8c9d` |
| Exact ID/slug boost |     Accepted |    Implemented |             4 boost variants | `25a8c9d` |
| Stage latency       |     Accepted |    Implemented |                  coordinator | `8580c50` |
| `auto-rag doctor`   |     Accepted |    Implemented |     6 tests, JSON + exit codes | `fb67d6a` |
| Demo scenarios      |     Accepted |    Implemented |                A/B/C scripts | `e49cbf8` |
| Jira full-content   |     Accepted |    Implemented |  Live SIRIUS-195479 regression | `40ac885`+`615a0f3` |
| Confluence PDF      |     Accepted |    Implemented |               4 content_status tests | `3161c85` |
| Allowlisted web     |     Accepted |    Implemented |            5 internal/public tests | `65b9804` |
| Lodestone           |     Accepted |    Implemented |        Graceful degradation | `615a0f3` |
| Corporate-first     |     Accepted |    Implemented |          gateway.toml policy | — |
| Chunking before LLM |  Conditional | Not implemented |   Только при потере фактов | — |

**Total tests:** 435 passed, 5 skipped, 1 xfailed *(verified at commit `e59ae59`)*

## 6. Retrieval vs Generation

Retrieval метрики: CitationHitRate@5, CitationPrecision@5, Recall@5, MRR, nDCG@5 (null when not computed), Empty rate, duplicate rate, source recall, latency.

Generation метрики: groundedness, answer completeness, citation usage, unsupported claim rate. Базовый локальный generation: 7B instruct (qwen2.5-7b-instruct reference).

## 7. Evaluation Gate

Ranking/fusion changes accepted only if:
- CitationHitRate@5 >= 0.79, Recall@5 >= 0.79, MRR >= 0.69, Empty rate <= 2.0%
- nDCG@5 never displayed as 0.0 when uncomputed

## 8. Web Policy

`include_web=False` — per-request kill switch. DCD отключает web для SIRIUS-*/INT-*/PROJECT-*/ACM-*/AA-*. Allowlisted public retrieval через SearXNG с domain filter (`aldpro.ru|astralinux.ru`).

## 9. Rollout Readiness

| Layer | Status |
|-------|--------|
| Architecture decision | **Accepted** |
| Single-user reference implementation | **Operational** — 435 tests, SIRIUS-195479 regression passed |
| Organization-wide rollout | **Not yet approved** |

**Критерии перехода к пилоту:**
- `auto-rag doctor` работает на всех профилях
- Корпоративные источники дают full content (Jira comments, Confluence PDF)
- Web generic disabled, allowlist управляется централизованно
- Секреты не входят в дистрибутив (credential_ref, не plaintext)
- Есть install/update/rollback процедура
- Есть smoke test для приёмки
- Source/latency diagnostics доступны

**Пилотная раскатка:** 10 → 50 → 200 → 1000 инженеров.
Организационная модель — отдельное решение (ADR-007: Managed Distribution).

## 10. Stop Conditions

Stabilization ends when: canonical duplicates don't occupy top-5, exact queries stable, stage latency visible, doctor works for minimal profile, 3 demos reproducible, metrics not regressed, 2 weeks/100 queries, no P0/P1 in reference flow.
