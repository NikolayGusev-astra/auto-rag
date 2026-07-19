# ADR-001: Auto-RAG как локальный offline-capable knowledge gateway для AI-агентов

**Статус:** Proposed
**Дата:** 19 июля 2026 года
**Решение заменяет:** неформальную архитектуру универсального local-first RAG
**Затрагивает:** agent integration, retrieval, synchronization, storage, security boundaries

## Контекст

Auto-RAG изначально развивался как локальная RAG-система с:

- ZVec или ChromaDB;
- DCD routing;
- MCP;
- web и federation fallback;
- LLM verification;
- episodic semantic memory;
- multi-tenant cache isolation;
- собственным orchestration pipeline.

Технические ошибки существующего pipeline исправлены к commit `396f598`.

В ходе уточнения требований установлено, что основным сценарием является персональная рабочая станция инженера:

- Auto-RAG запускается локально;
- подключается к Codex, Cursor, Claude и Hermes;
- используется одним инженером;
- hot-seat и совместное использование процесса не предусмотрены;
- рабочая станция защищается Astra Linux, PARSEC и режимом «Смоленск»;
- Jira, Wiki, почта, корпоративные RAG и MCP применяют собственные ACL;
- Auto-RAG обращается к ним с credentials текущего пользователя;
- часть источников индексируется локально для работы вне корпоративной сети.

Следовательно, Auto-RAG не должен выступать как:

- enterprise IAM;
- multi-tenant RAG backend;
- централизованный чат-бот;
- независимый агент рассуждения;
- замена корпоративным источникам или их ACL.

## Решение

Auto-RAG проектируется как:

> Локальный offline-capable knowledge gateway, предоставляющий AI-агентам единый интерфейс поиска и получения документов из live-корпоративных источников и локально синхронизированного снимка.

### Основной интерфейс

Основным внешним протоколом становится MCP over stdio.

Поддерживаемые инструменты:

```
search
fetch
sync
sync_status
list_sources
source_status
```

Опционально может предоставляться localhost HTTP API, но он не считается enterprise API gateway.

### Основной результат

Auto-RAG возвращает структурированные evidence, а не финальный LLM-ответ.

```json
{
  "query": "как обновить кластер",
  "mode": "mixed",
  "results": [
    {
      "id": "confluence:12345#chunk-4",
      "document_id": "confluence:12345",
      "title": "Обновление кластера",
      "text": "...",
      "source": "confluence",
      "uri": "https://wiki.example/pages/12345",
      "origin": "local_snapshot",
      "retrieval_score": 0.81,
      "reranker_score": 0.92,
      "updated_at": "2026-07-10T11:00:00Z",
      "synced_at": "2026-07-18T18:30:00Z"
    }
  ]
}
```

Агент отвечает за:

- дальнейшую декомпозицию задачи;
- повторный поиск;
- рассуждение;
- генерацию ответа;
- выбор действий и инструментов.

### Авторизация

Для live-доступа Auto-RAG использует PAT, OAuth или MCP credentials текущего пользователя.

ACL применяет исходная система:

```
Auto-RAG
→ запрос с credential пользователя
→ Jira/Wiki/MCP проверяет права
→ возвращаются только доступные данные
```

Auto-RAG не реализует собственную RBAC/ABAC модель.

### Локальная защита

Защита локального процесса, индекса и credentials относится к среде Astra Linux/PARSEC.

Auto-RAG обязан:

- соблюдать права файловой системы;
- не ослаблять системные политики;
- не логировать credentials;
- не помещать credentials в индекс;
- не возвращать credentials агентам.

Прикладное шифрование и tenant isolation не являются обязательными функциями ядра, если это не требуется отдельным профилем развёртывания.

### Синхронизация

Каждый источник реализует:

```python
class SourceConnector(Protocol):
    async def search_live(self, request: SearchRequest) -> list[Evidence]: ...
    async def fetch(self, ref: DocumentRef) -> Document: ...
    async def sync_changes(self, cursor: str | None) -> SyncBatch: ...
    async def health(self) -> SourceHealth: ...
```

`SyncBatch` содержит:

- добавленные документы;
- изменённые документы;
- удалённые документы;
- новый cursor;
- предупреждения;
- статистику.

Обновление локального индекса выполняется атомарно:

```
fetch changes → parse → chunk → embed → build staged revision → validate → publish revision
```

При ошибке активный индекс остаётся неизменным.

### Online/offline policy

Состояние определяется отдельно для каждого источника.

```
Confluence live available → live + local
Jira unavailable          → local only
Mail MCP available        → live only или live + local
```

Локальный snapshot не считается fallback второго сорта. Это полноценный источник с явно указанной свежестью.

### Retrieval pipeline

```
Agent search request
→ validate
→ select connectors
→ execute local and available live retrieval
→ normalize to Evidence
→ deduplicate
→ apply version/entity filters
→ rerank
→ source-aware fusion
→ return structured result
```

### Data model

```python
@dataclass(frozen=True)
class Document:
    id: str
    source: str
    source_instance: str
    title: str
    text: str
    uri: str | None
    version: str | None
    updated_at: datetime | None
    content_hash: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class Evidence:
    id: str
    document_id: str
    title: str
    text: str
    source: str
    uri: str | None
    origin: Literal["local_snapshot", "live_corporate", "public_web", "agent_memory"]
    retrieval_score: float
    reranker_score: float | None
    updated_at: datetime | None
    synced_at: datetime | None
    metadata: dict[str, object]
```

### Memory

Episodic memory не входит в документальный knowledge pipeline.

Она оформляется как отдельный optional connector (`MemoryConnector`):

- маркируется как `agent_memory`;
- не short-circuit'ит corporate retrieval;
- не считается документальным источником;
- не смешивается с корпоративными данными без явной fusion-policy.

### Web

Web является explicit opt-in источником. Запускается только когда:

- агент указал `include_web=true`;
- source policy разрешает web;
- либо через отдельный tool.

### Federation

Federation между экземплярами Auto-RAG исключается из reference architecture. Может оставаться experimental extension, но не участвует в стандартном запросе.

### LLM usage

Локальный LLM разрешён для:

- query rewrite;
- entity extraction;
- optional reranking;
- evaluation;
- debug CLI.

Language model является опциональной capability. Базовые sync, indexing, retrieval и MCP-функции не зависят от её наличия. Генерация финального ответа внутри Auto-RAG не входит в основной MCP-контракт.

## Архитектурные компоненты

```
┌───────────────────────────────────────────────┐
│ Codex / Cursor / Claude / Hermes             │
└──────────────────────┬────────────────────────┘
                       │ MCP stdio
                       ▼
┌───────────────────────────────────────────────┐
│ Agent Gateway                                 │
│ search / fetch / sync / status                │
└──────────────────────┬────────────────────────┘
                       ▼
┌───────────────────────────────────────────────┐
│ Retrieval Coordinator                         │
│ source selection / execution / fusion         │
└───────────────┬───────────────────┬───────────┘
                │                   │
                ▼                   ▼
┌────────────────────────┐  ┌───────────────────┐
│ Local Knowledge Store  │  │ Live Connectors   │
│ docs/chunks/vectors     │  │ Jira/Wiki/MCP/RAG│
│ cursors/tombstones      │  │ user credentials │
└──────────────┬─────────┘  └─────────┬─────────┘
               │                      │
               └──────────┬───────────┘
                          ▼
                 Normalize / Dedup
                 Filter / Rerank
                 Evidence response
```

## Последствия

### Положительные

- архитектура соответствует реальному сценарию;
- уменьшается количество ветвей обработки запроса;
- agent и retrieval responsibilities разделены;
- offline mode становится естественной частью системы;
- источники сохраняют контроль ACL;
- снижается объём собственного security-кода;
- упрощается интеграция с разными агентами;
- появляется ясная продуктовая ценность;
- synchronization становится тестируемым отдельным subsystem.

### Отрицательные

- потребуется новый MCP-facing API;
- существующий `rag_async` не станет основным entrypoint;
- часть текущих функций будет переведена в experimental;
- потребуется миграция индекса к документно-ориентированной metadata schema;
- sync adapters сложнее обычного search adapter;
- нужно поддерживать версии схемы и rebuild;
- старый full-RAG CLI и новый retrieval gateway некоторое время будут сосуществовать.

## Отвергнутые варианты

- **Централизованная enterprise RAG-платформа** — дублирует готовые корпоративные решения, требует собственного IAM.
- **Только live proxy без локального индекса** — не обеспечивает работу вне корпоративной сети.
- **Полный агент внутри Auto-RAG** — Codex/Cursor/Claude/Hermes уже выполняют planning и reasoning.
- **Memory-first pipeline** — semantic memory не является надёжной заменой актуальной корпоративной документации.
- **Отдельный offline pipeline** — создаёт дублирование логики; offline = отсутствие части live connectors.

## План миграции

### Фаза 1
- зафиксировать ADR;
- описать MCP schemas;
- добавить новые domain models;
- создать `SourceConnector`;
- обернуть существующие Jira/MCP/ZVec реализации адаптерами.

### Фаза 2
- реализовать `search` и `fetch`;
- вернуть structured evidence;
- отключить генерацию ответа в agent mode;
- добавить source availability.

### Фаза 3
- реализовать Sync Engine;
- добавить cursor, tombstones и staged index;
- мигрировать metadata;
- добавить `sync` и `sync_status`.

### Фаза 4
- перенести memory в optional connector;
- убрать federation из reference pipeline;
- оставить web только explicit;
- пометить `rag_async` legacy/full-RAG mode.

### Фаза 5
- разбить монолитный orchestrator;
- сделать новый gateway reference implementation;
- провести agent integration tests для Codex, Cursor, Claude и Hermes.

## Критерии принятия

1. Агент может вызвать MCP `search`.
2. Результат содержит evidence, URI и freshness metadata.
3. Один запрос может объединять live и local snapshot.
4. При отсутствии сети тот же tool работает только по snapshot.
5. Incremental sync поддерживает add/update/delete.
6. Неуспешная sync не повреждает активный индекс.
7. PAT не появляется в логах и результатах.
8. Memory не short-circuit'ит документальный retrieval.
9. Federation отсутствует в default path.
10. Reference pipeline не требует tenant/ACL модели.
11. Интеграционный тест проходит минимум с двумя агентными клиентами.
12. `rag_async` больше не является обязательным entrypoint для agent mode.

## Решение

Принять архитектуру локального offline-capable knowledge gateway как основное направление Auto-RAG.

Существующий local-first RAG сохранить временно как legacy/full-RAG профиль до завершения миграции.
