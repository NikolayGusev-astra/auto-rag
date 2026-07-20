# ADR-005: Trusted Local Execution Domain

* **Status:** Proposed
* **Date:** 2026-07-21
* **Related:** ADR-001, ADR-002, ADR-003, ADR-004
* **Extends:** ADR-004
* **Decision scope:** размещение и доверие к компонентам Auto-RAG

## Context

ADR-004 определяет Auto-RAG как локальную offline-capable RAG-платформу одного инженера. Термин «локальная» не должен трактоваться как обязательное размещение всех компонентов в одном процессе, на одной рабочей станции или только на localhost.

Целевая система включает ресурсоёмкие компоненты: Hermes Agent, Auto-RAG MCP gateway, локальные snapshots, ZVec/Chroma, embedding/reranker providers, LM Studio/vLLM, SearXNG, Trafilatura, Camoufox, DCD learning, golden-set evaluation, Memvid.

Часть сервисов может располагаться на доверенном вычислительном узле, доступном напрямую или через SSH-туннель. Архитектурная граница определяется не физическим расположением, а единым пользователем, доверенным контуром исполнения, политиками передачи данных и сохранением offline degradation.

## Decision

Auto-RAG разворачивается в рамках **Trusted Local Execution Domain** — доверенного вычислительного контура одного инженера. Контур может состоять из рабочей станции, локальных контейнеров и опциональных доверенных узлов.

Система остаётся локальной персональной RAG-платформой, даже если отдельные тяжёлые сервисы вынесены с рабочей станции.

## Deployment classes

| Класс | Описание | Примеры |
|-------|----------|--------|
| `local_process` | На той же ОС, в пользовательском контексте | Gateway, LocalSnapshot, CPU embeddings |
| `local_container` | Локальный контейнер на рабочей станции | SearXNG, ZVec server, Camoufox |
| `trusted_node` | Отдельный доверенный узел, прямая сеть | GPU-сервер LM Studio, ZVec |
| `ssh_tunneled` | Доверенный узел через SSH-туннель | Camoufox, SearXNG на autolycus-agent.ru |
| `external_cloud` | Вне доверенного контура, deny-by-default | Внешний LLM API |

## Reference topologies

### Topology B: Local gateway with trusted compute node (рекомендуемая)

```
Рабочая станция:
  Hermes / Codex
  Auto-RAG gateway
  LocalSnapshotConnector

Доверенный узел (autolycus-agent.ru):
  SearXNG
  Camoufox
  LM Studio
  ZVec
  Trafilatura
```

При недоступности узла: gateway продолжает работать, local snapshot доступен, web/remote помечаются unavailable.

## Web research architecture

```
QueryPlan → web policy gate → SearXNG discovery
  → URL filtering → HTTP fetch → Trafilatura extraction
  → quality check → Camoufox fallback (при необходимости)
  → sanitization → provenance → Evidence[]
```

- SearXNG: discovery (URL, title, snippet)
- Trafilatura: основной extractor (статические HTML)
- Camoufox: дорогой fallback (JS-рендер, динамические страницы)
- Web retrieval выключен по умолчанию, включается QueryPlan + policy

## Availability

Обязательные для reference profile: gateway + knowledge root + LocalSnapshotConnector + MCP transport. Отказ любого опционального компонента не блокирует локальный retrieval.

## Acceptance criteria

1. Config schema поддерживает `deployment_class`
2. Для remote services заданы policy и timeout
3. `trusted_node`/`ssh_tunneled` ≠ cloud endpoints
4. `external_cloud` → deny-by-default
5. Отказ доверенного узла не блокирует LocalSnapshot
6. Web retrieval выключен по умолчанию
7. Trafilatura → Camoufox fallback
8. Все web Evidence содержат URL и origin
9. SSH tunnel управляется операционным слоем
10. Минимальный offline reference profile работает без удалённых узлов
