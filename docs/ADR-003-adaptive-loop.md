# ADR-003: Сохранение adaptive retrieval loop в Auto-RAG

**Статус:** Proposed
**Дата:** 19 июля 2026 года
**Связанные решения:** ADR-001 (knowledge gateway), ADR-002 (model runtime independence)

## Контекст

Auto-RAG изначально использует адаптивный цикл:

```
Memvid recall → DCD routing → ZVec/MCP/Web retrieval → DCD learning → Memvid enrichment
```

При переходе к gateway-архитектуре (ADR-001) необходимо сохранить ценность adaptive loop
(опыт предыдущих запросов, адаптация маршрутизации, накопление инженерного контекста),
устранив недостатки старого pipeline:

- безусловный memory short-circuit;
- жёстко связанный fallback-каскад;
- ранние возвраты из середины orchestration;
- смешение документальных знаний и агентной памяти;
- обучение на слабых сигналах;
- обязательная зависимость retrieval от LLM;
- автоматический web fallback без политики.

## Решение

Adaptive loop сохраняется как **отдельный профиль** (adaptive), реализуемый как набор
независимых стадий вокруг единого retrieval core. Каждая стадия имеет отдельный контракт
и не подменяет последующие безусловно.

Целевой цикл:

```
Agent request
→ Memvid recall
→ DCD query planning and routing
→ Local / Live / Optional Web retrieval
→ Evidence normalization
→ Deduplication, filtering and reranking
→ Structured evidence response
→ DCD feedback and learning
→ Memvid enrichment
```

## Режимы работы

### Reference profile (baseline)
```
DCD planning → Local and live retrieval → Evidence pipeline → Agent
```
- Memvid необязателен; DCD learning необязателен; web выключен; LLM не требуется;
  CPU-only; reference path для тестов и интеграции.

### Adaptive profile (расширенный)
```
Memvid recall → DCD routing → Local/MCP/optional Web → Evidence fusion → DCD learning → Memvid enrichment
```
- использует опыт; адаптирует routing; включается конфигурацией; не меняет контракт выдачи.

## Архитектурная схема

```
Agent (MCP search/fetch)
  → Agent Gateway (validation/limits)
  → Memvid Recall (hints or memory evidence)
  → DCD Query Planner (QueryPlan)
  → Retrieval Coordinator (Local / Live / Web-opt-in)
  → Evidence Pipeline (Normalize → Dedup → Filter → Rerank)
  → Evidence response
  → DCD Feedback + Memvid Enrichment (post-response)
```

## Memvid Recall

- Назначение: поиск предыдущих эпизодов (задачи, маршруты, сущности, итоги).
- **Memory hit НЕ завершает retrieval по умолчанию.**
- Неправильная модель: `memory match → вернуть ответ → не выполнять retrieval`.
- Принятая модель: `memory match → добавить hints → выполнить retrieval → объединить`.
- Допустимый short-circuit ТОЛЬКО как отдельный cache mode при: точном совпадении
  нормализованного запроса + совпадении embedding profile + совпадении ревизии индекса
  + неизменившемся snapshot + сохранённых provenance/IDs + явной политике. Semantic
  similarity сама по себе недостаточна.

Memory evidence маркируется `origin="agent_memory"`, НЕ маскируется под Jira/Wiki/snapshot.

## DCD Query Planner

DCD строит `QueryPlan`, НЕ выполняет retrieval.

```python
@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    queries: tuple[str, ...]
    domains: tuple[str, ...]
    sources: tuple[str, ...]
    include_local: bool
    include_live: bool
    include_web: bool
    max_results: int
    retrieval_budget_ms: int | None
    hints: dict[str, object]
```

DCD использует: запрос, сущности, версию продукта, доступность источников, memory hints,
историю эффективности, режим online/offline, model capabilities.

## Query decomposition

Compound-запрос → `SubQueryPlan[]` → тот же Retrieval Coordinator → `Evidence[]` → global fusion.
**Запрещается отдельная compound-ветка** с собственной логикой кэша/ACL/scoring/fallback/memory.

## Retrieval Coordinator

Единый contract для всех connector-категорий:
- **Local snapshot:** ZVec, Chroma, lexical, другие локальные backend.
- **Live corporate:** Jira, Confluence, Wiki, почтовые MCP, корпоративные RAG API.
- **Public web:** explicit opt-in (agent `include_web=true` / DCD policy / отдельный tool).
  Недоступность локального НЕ разрешает автоматический web fallback.

## Offline behavior

Offline — НЕ отдельный pipeline. Live → unavailable, local → available, pipeline продолжается.
DCD строит план только из доступных. Evidence из snapshot содержит `synced_at`, `updated_at`,
`origin=local_snapshot`, ID ревизии, freshness metadata.

## Evidence normalization

```python
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
    final_score: float
    updated_at: datetime | None
    synced_at: datetime | None
    metadata: dict[str, object]
```

Разделять: `retrieval_score` (connector), `reranker_score` (reranker), `final_score` (fusion),
`origin` (происхождение). Одно поле `score` НЕ используется для всех смыслов.

## Fusion and ranking

Единый pipeline: `normalize → exact filters → deduplicate → source-aware calibration →
optional reranking → final sort`. Memory evidence НЕ вытесняет документальное из-за
semantic similarity. Базовая политика: live/local = основной слой, agent_memory = hints,
public_web = дополнительный. Коэффициенты — конфигурация, не domain model.

## DCD Feedback

```python
@dataclass(frozen=True)
class RoutingFeedback:
    query: str
    plan_id: str
    selected_sources: tuple[str, ...]
    successful_sources: tuple[str, ...]
    useful_document_ids: tuple[str, ...]
    result_count: int
    latency_ms: int
    agent_feedback: str | None
    explicit_success: bool | None
```

Сильные сигналы: повторное использование evidence агентом, `fetch` документа, отметка
полезности, цитирование, отсутствие повторного запроса, exact entity/version match,
актуальный документ. Слабые (НЕ успех): непустой ответ, score > порог, нейтральный LLM
verdict, отсутствие exception, порядок источника.

**DCD learning safety:** версионирован, обратим, наблюдаем, ограничен минимумом событий,
защищён от одного ошибочного запроса, отделён от production policy. Схема:
`feedback events → aggregate → evaluate vs golden set → candidate policy → canary → activate`.
НЕ синхронно после каждого запроса.

## Memvid Enrichment

Сохраняются: нормализованный запрос, summary задачи, route, document IDs, source URIs,
версии, сущности, synced_at, success, embedding profile, index revision. НЕ сохраняются:
непрозрачный LLM-ответ без provenance, credentials, секреты, весь context без лимитов,
web content как корпоративный факт, неуспешные без negative-метки.

```python
@dataclass(frozen=True)
class MemoryEpisode:
    id: str
    query: str
    summary: str
    route: tuple[str, ...]
    document_ids: tuple[str, ...]
    source_uris: tuple[str, ...]
    entities: tuple[str, ...]
    successful: bool | None
    created_at: datetime
    index_revision: str | None
    embedding_profile_id: str | None
```

## Model runtime independence

Adaptive loop не зависит от LM Studio (ADR-002). Без LLM работают: Memvid recall,
deterministic DCD, local retrieval, live connectors, lexical, fusion, feedback, enrichment.
С CPU: embeddings, lightweight rerank, query classification, entity extraction. С LLM
(опционально): query rewrite, complex decomposition, semantic verification, memory summary.
Отказ LLM → пропуск опциональных стадий, цикл продолжается.

## Embedding compatibility

Memvid и knowledge index МОГУТ использовать разные embedding profiles. Каждый store хранит
свой профиль (`knowledge_index_embedding_profile`, `memory_embedding_profile`). Смена runtime
без перестроения — только при полном совпадении contract (model ID, revision, dimension,
normalization, metric, preprocessing). Размерность сама по себе недостаточна.

## Конфигурация

```yaml
adaptive_loop:
  enabled: true
  memory:
    recall_enabled: true
    enrichment_enabled: true
    allow_short_circuit: false
    max_results: 3
  dcd:
    enabled: true
    learning_enabled: true
    require_explicit_feedback_for_learning: false
  retrieval:
    local_enabled: true
    live_enabled: true
    web_enabled: false
    parallel_sources: true
  fusion:
    reranker_optional: true
    deduplicate: true
    source_balance: true
```

Минимальный (reference):
```yaml
adaptive_loop:
  enabled: false
retrieval:
  local_enabled: true
  live_enabled: true
  web_enabled: false
```

## Наблюдаемость

Trace на каждый запрос: `memory_recall, dcd_plan, connector_execution, evidence_normalization,
deduplication, reranking, response, dcd_feedback, memory_enrichment`. Содержит профиль,
доступные источники, маршрут, число результатов по источникам, причины пропуска,
model capabilities, offline/online, итоговые document IDs, latency стадий. Credentials и
защищаемый контент НЕ записываются.

## Ошибки и деградация

- Memvid недоступен → skip recall + enrichment, run DCD + retrieval.
- DCD недоступен → deterministic default plan (local + configured live).
- Local vector backend недоступен → lexical local или live.
- Live недоступны → local snapshot.
- Web недоступен → skip web.
- Reranker недоступен → calibrated deterministic score.
- DCD learning недоступен → return evidence, store feedback для поздней обработки.
- Ошибки постпроцессинга НЕ отменяют сформированный evidence response.

## Архитектурные ограничения (запрещается)

1. Делать Memvid обязательной зависимостью.
2. Завершать semantic memory hit'ом стандартный retrieval.
3. Реализовывать отдельный compound pipeline.
4. Автоматически включать web при retrieval miss.
5. Обучать DCD только по факту непустого ответа.
6. Смешивать memory и документы без `origin`.
7. Делать DCD learning синхронным условием ответа.
8. Блокировать retrieval из-за отсутствия LLM.
9. Сохранять memory episode без provenance.
10. Использовать разные embedding spaces в одном индексе.

## План реализации (Фазы A–F)

- **A — контракты:** `QueryPlan`, `RoutingFeedback`, `MemoryEpisode`, `MemoryEvidence`;
  `origin=agent_memory`; разделение retrieval/reranker/final scores.
- **B — Memory Connector:** адаптировать Memvid к connector-интерфейсу; удалить
  стандартный semantic short-circuit; provenance; embedding profile validation.
- **C — DCD Planner:** отделить план от выполнения; memory hints; source availability;
  унификация compound decomposition.
- **D — Retrieval Coordinator:** local/live/web connectors; normalize; global dedup + fusion;
  structured evidence.
- **E — Feedback and Learning:** routing feedback; offline aggregation; golden-set eval;
  canary activation.
- **F — Enrichment:** сохранение успешных эпизодов с provenance; исключение credentials;
  negative episodes.

## Критерии принятия (ADR-003)

1. Memvid recall не обязателен.
2. Semantic memory hit не прерывает retrieval по умолчанию.
3. DCD возвращает `QueryPlan`, а не выполняет поиск.
4. Все подзапросы проходят один retrieval coordinator.
5. ZVec, MCP, web возвращают единый `Evidence`.
6. Web выключен по умолчанию.
7. Offline-запрос использует тот же pipeline.
8. DCD learning после возврата evidence или асинхронно.
9. Routing feedback содержит фактическую полезность источников.
10. Memvid enrichment сохраняет provenance.
11. Memory и document evidence различаются через `origin`.
12. Отсутствие LM Studio не блокирует цикл.
13. Отказ reranker не блокирует цикл.
14. Отказ Memvid не блокирует цикл.
15. Отказ DCD learning не блокирует ответ.
16. Adaptive и reference profile имеют одинаковый внешний MCP-контракт.
17. Golden tests: adaptive не хуже reference по базовым retrieval-метрикам.
18. Regression tests: нет возврата memory answer без актуального retrieval при semantic match.

## Решение

Сохранить adaptive loop как adaptive profile. Реализовать как набор опциональных стадий
вокруг единого retrieval coordinator и общего evidence contract. Reference profile
(без памяти и learning) остаётся обязательной базовой конфигурацией и baseline.
