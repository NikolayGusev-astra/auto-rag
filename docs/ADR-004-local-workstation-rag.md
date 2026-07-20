# ADR-004: Local Workstation RAG for a Single Engineer

* **Status:** Proposed
* **Date:** 2026-07-20
* **Related:** ADR-001, ADR-002, ADR-003
* **Supersedes:** любые трактовки Auto-RAG как централизованной multi-user RAG-платформы

## Context

Auto-RAG создаётся как локальный knowledge gateway для одного инженера, работающего на защищённой рабочей станции Astra Linux.

Инженер использует Codex, Cursor, Claude или другой coding agent и должен получать релевантный контекст из:

* локально синхронизированных Jira и Wiki;
* корпоративных MCP/RAG-источников, когда сеть доступна;
* локальной памяти и накопленных episodes;
* опционального web retrieval, только при явном разрешении.

Система должна продолжать работать вне корпоративной сети, используя локальный snapshot.

Phase 6 закрыла основные архитектурные дефекты:

* destructive incremental sync;
* fail-open обработку повреждённых ревизий;
* несовместимые manifest-схемы;
* отсутствие реального индексного pipeline;
* connector bypass в AdaptiveLoop;
* фиктивную availability;
* memory short-circuit;
* непостоянные feedback и enrichment;
* собственный JSON-протокол вместо MCP.

После этих исправлений возник риск продуктового дрейфа: дальнейшая разработка connector factory, configuration runtime и MCP interoperability может превратить локальный инструмент инженера в универсальную серверную платформу.

Этот ADR фиксирует продуктовую границу.

## Decision

Auto-RAG остаётся **локальным offline-capable RAG и knowledge gateway для одного инженера**.

Система не является:

* multi-tenant RAG-платформой;
* централизованным корпоративным поисковым сервисом;
* общим индексом для команды;
* системой управления секретами;
* удалённой control plane;
* обязательным HTTP-сервисом.

Основной deployment target:

```text
один инженер
→ одна локальная рабочая станция
→ один локальный knowledge root
→ один writer process
→ локальный MCP stdio gateway
→ coding agent
```

## Primary user flow

Целевой пользовательский путь:

```text
auto-rag init
→ auto-rag sync
→ auto-rag serve
→ coding agent вызывает MCP search
→ Auto-RAG возвращает Evidence[]
```

При недоступной корпоративной сети:

```text
corporate connectors unavailable
→ LocalSnapshotConnector остаётся healthy
→ поиск выполняется по локальному индексу
→ gateway продолжает работать
```

## Architecture

### 1. Local workstation process

Auto-RAG запускается как локальный пользовательский процесс.

По умолчанию используется MCP stdio transport.

HTTP transport может существовать только как опциональный localhost-интерфейс и не является обязательной частью reference deployment.

### 2. Local knowledge root

Все локальные данные хранятся в пользовательском knowledge root, например:

```text
~/.local/share/auto-rag
```

Knowledge root содержит:

* source snapshots;
* chunks;
* lexical index;
* vector artifacts;
* revision manifests;
* feedback journal;
* memory episodes;
* runtime diagnostics.

Один knowledge root обслуживается одним writer process.

### 3. Source systems remain authoritative

Auto-RAG не реализует собственную модель корпоративных ACL.

Jira, Wiki, почта, MCP и корпоративные RAG-системы выполняют авторизацию с использованием прав текущего инженера.

Локально сохраняются только данные, которые инженер уже имел право получить.

Auto-RAG не расширяет права доступа и не создаёт общую копию данных для других пользователей.

### 4. Local snapshot is the reference source

`LocalSnapshotConnector` является обязательным connector-ом reference profile.

Он должен запускаться без:

* корпоративной сети;
* LM Studio;
* cloud API;
* Memvid;
* DCD learning;
* reranker.

Live connectors являются опциональными дополнениями.

### 5. Connector configuration

Следующая продуктовая фаза реализует только локальный bootstrap и узкий connector factory.

Configuration loader должен отвечать на вопросы:

* где расположен knowledge root;
* включён ли локальный snapshot;
* какие live sources включены;
* где получить ссылку на credentials;
* разрешён ли web;
* включён ли adaptive profile.

Config не должен содержать секреты в открытом виде.

Разрешены только ссылки на секреты, например:

```toml
credential_env = "JIRA_TOKEN"
```

или ссылки на системное хранилище учётных данных.

### 6. No mandatory network dependency

Startup gateway не должен блокироваться из-за недоступности Jira, Wiki, MCP, LM Studio или cloud provider.

Недоступность источника отражается в health и diagnostics.

Локальный retrieval продолжает работать.

### 7. Evidence-first contract

Основной ответ gateway:

```text
Evidence[]
```

Каждый Evidence содержит как минимум:

* source;
* origin;
* document ID;
* URI;
* text;
* retrieval score;
* optional reranker score;
* final score;
* provenance metadata.

Auto-RAG не обязан формировать окончательный текстовый ответ. Интерпретацию Evidence выполняет coding agent.

### 8. Reference and adaptive profiles

Reference profile:

```text
query
→ RetrievalCoordinator
→ local/live connectors
→ fusion
→ Evidence[]
```

Adaptive profile:

```text
memory recall
→ DCD QueryPlan
→ RetrievalCoordinator
→ local/live/optional web
→ fusion
→ feedback persistence
→ episode enrichment
→ Evidence[]
```

Оба профиля имеют одинаковый внешний MCP-контракт.

Adaptive profile не имеет права:

* завершать retrieval после memory hit;
* обходить RetrievalCoordinator;
* автоматически включать web;
* блокировать ответ при отсутствии memory или learning components.

### 9. Model runtime

LM Studio не является обязательной зависимостью.

Model runtime использует независимые контракты:

* `EmbeddingProvider`;
* `RerankerProvider`;
* `LanguageModelProvider`.

Поддерживаются:

* CPU-local providers;
* OpenAI-compatible providers;
* cloud providers при явной policy;
* lexical-only fallback.

Активный vector-enabled index нельзя неявно заменить lexical-only revision.

### 10. MCP integration

Основной agent interface — официальный MCP SDK через stdio.

Старый JSON-lines transport сохраняется только как debug/legacy interface.

До продуктового релиза должны быть подтверждены:

* официальный MCP `ClientSession`;
* минимум один реальный coding agent;
* корректный startup и shutdown;
* отсутствие постороннего вывода в stdout;
* `search` от agent до локального snapshot и обратно.

## Product boundaries

В ближайший roadmap не входят:

* tenant IDs;
* multi-user isolation;
* центральный сервер;
* Kubernetes;
* distributed queues;
* service discovery;
* shared team index;
* remote connector management;
* plugin marketplace;
* server-side credential vault;
* обязательная web UI;
* обязательный cloud inference;
* автоматическая отправка документов во внешние API.

Такие возможности требуют отдельного ADR и не могут появляться как побочный результат локального bootstrap.

## Next phase

Следующая фаза называется:

```text
Phase 7 — Local Workstation Bootstrap
```

### Scope

1. Локальная versioned config schema.
2. Config loader.
3. Узкий connector registry/factory.
4. Обязательная регистрация `LocalSnapshotConnector`.
5. Опциональная регистрация Jira/Wiki/MCP connectors.
6. Credential references без хранения секретов.
7. Startup diagnostics.
8. `list_sources` и `source_status`.
9. CLI и MCP server используют factory вместо пустого `_configured_connectors()`.
10. Offline startup acceptance test.

### Acceptance flow

```text
локальный config
→ gateway startup
→ LocalSnapshotConnector registered
→ MCP ClientSession initialize
→ tools/list
→ search
→ RetrievalCoordinator
→ LocalSnapshotConnector
→ Evidence[]
```

Отдельный acceptance test:

```text
корпоративная сеть недоступна
→ live connectors unhealthy
→ gateway успешно стартует
→ local search возвращает Evidence[]
```

## Consequences

### Positive

* Сохраняется чёткая продуктовая цель.
* Система остаётся пригодной для защищённой локальной рабочей станции.
* Offline retrieval является основной возможностью, а не fallback после отказа сервера.
* Архитектура не усложняется multi-tenant требованиями.
* MCP integration проверяет реальную ценность для coding agents.
* Секреты не попадают в config или локальный индекс.

### Negative

* Один knowledge root не поддерживает concurrent writers.
* Нет общей командной базы знаний.
* Нет централизованного администрирования.
* Установка и первоначальная синхронизация выполняются на каждой рабочей станции.
* Некоторые enterprise-функции сознательно откладываются.

### Accepted trade-off

Эти ограничения соответствуют целевому продукту.

Auto-RAG оптимизируется не для максимального числа пользователей и источников, а для надёжной работы одного инженера со своим локальным контекстом.

## Acceptance criteria

ADR считается выполненным, когда:

1. Новый пользователь может создать локальный config одной командой.
2. Gateway запускается без корпоративной сети.
3. `LocalSnapshotConnector` автоматически регистрируется из config.
4. Пустой `_configured_connectors()` больше не используется в product path.
5. Секреты не записываются в config, manifest, Evidence или memory episode.
6. Официальный MCP `ClientSession` выполняет `initialize`, `list_tools` и `search`.
7. Минимум один реальный coding agent получает Evidence из локального snapshot.
8. Отказ live connector не блокирует local retrieval.
9. Web остаётся выключенным по умолчанию.
10. В коде не появляются tenant, shared index или central control-plane abstractions без нового ADR.

## Final statement

Auto-RAG — это не корпоративная RAG-платформа.

Auto-RAG — это локальный, offline-capable knowledge gateway одного инженера, который предоставляет coding agents проверяемые Evidence из доступных пользователю источников и продолжает работать вне корпоративной сети.

Этот ADR логично зафиксировать как `docs/ADR-004-local-workstation-rag.md`, а следующую фазу назвать `Phase 7 — Local Workstation Bootstrap`.
