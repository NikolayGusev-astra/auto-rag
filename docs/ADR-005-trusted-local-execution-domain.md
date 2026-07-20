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

ADR считается реализованным, когда:

1. Config schema поддерживает `deployment_class`.
2. Для каждого remote service задаются policy и timeout.
3. `trusted_node` и `ssh_tunneled` endpoints не считаются cloud endpoints.
4. `external_cloud` использует deny-by-default.
5. Отказ доверенного узла не блокирует LocalSnapshot retrieval.
6. Health различает configured, reachable и ready.
7. Web retrieval выключен по умолчанию.
8. Trafilatura используется до Camoufox (quality-gated fallback).
9. Camoufox имеет resource и security isolation:
   - SSRF-фильтр (запрет localhost/private/link-local);
   - валидация схемы `http/https`;
   - ограничение redirects;
   - без `--no-sandbox` без доказанной внешней контейнеризации.
10. Все web Evidence содержат URL и origin.
11. SSH tunnel управляется операционным слоем, а не gateway.
12. Архитектурная документация маркирует `CURRENT / AVAILABLE / TARGET`.
13. Один пользователь и один персональный trust domain остаются продуктовой границей.
14. Ни один новый endpoint не получает document content без явной policy.
15. Минимальный offline reference profile работает без удалённых узлов.

## Product boundaries

ADR-005 разрешает сложную и распределённую архитектуру, но не разрешает:

* multi-tenant control plane;
* общий индекс для произвольного числа пользователей;
* централизованное управление персональными credentials;
* неявную передачу корпоративных документов во внешний cloud;
* обязательную зависимость от доверенного удалённого узла;
* превращение SSH transport в собственный orchestration subsystem;
* автоматическое включение web retrieval без policy.

### Positive

* Тяжёлые сервисы можно вынести с ноутбука.
* Сохраняется единая локальная персональная RAG-система.
* Offline retrieval продолжает работать без доверенного узла.
* Появляется единая policy-модель для local, trusted и external endpoints.

### Accepted trade-off

Сложность оправдана: Auto-RAG — полноценная RAG-платформа инженера, а не минимальный файловый поиск. Распределение компонентов внутри доверенного контура сохраняет автономность и безопасность.<｜end▁of▁thinking｜>Любое расширение к shared enterprise platform требует отдельного ADR.
