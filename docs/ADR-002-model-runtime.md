# ADR-002: Независимый от LM Studio модельный слой и переносимость индекса

**Статус:** Proposed
**Дата:** 19 июля 2026 года
**Связанные решения:** ADR-001 — Auto-RAG как локальный offline-capable knowledge gateway

## Контекст

Auto-RAG должен работать в нескольких средах:

- рабочая станция с LM Studio;
- CPU-only рабочая станция без GPU;
- полностью автономная среда без LM Studio;
- рабочая станция с локальным inference server;
- среда с разрешённым доступом к облачной модели;
- среда, где генеративная модель отсутствует полностью.

Отсутствие LM Studio не должно блокировать:

- синхронизацию источников;
- построение локального индекса;
- поиск по локальному индексу;
- MCP-интерфейс для агентов;
- возврат evidence и ссылок;
- работу в offline-режиме.

Auto-RAG не должен связывать модель хранения индекса с конкретным inference runtime.

## Решение

Вводится единый Model Runtime Layer, разделённый по функциональным ролям:

```
EmbeddingProvider
RerankerProvider
LanguageModelProvider
```

Каждая роль является независимой и опциональной.

Основной retrieval pipeline обязан работать при наличии только `EmbeddingProvider`.

```
Agent → Auto-RAG → EmbeddingProvider → Local vector index → Evidence
```

`LanguageModelProvider` и `RerankerProvider` улучшают качество, но не являются обязательными для базовой функциональности.

## Поддерживаемые классы runtime

### LM Studio
OpenAI-compatible локальный endpoint. Используется только как один из адаптеров (`LmStudioEmbeddingProvider`, `LmStudioLanguageModelProvider`). LM Studio не упоминается в domain-логике и не является обязательной зависимостью.

### CPU-local
Локальное выполнение моделей непосредственно в процессе или через локальный runtime:
- `SentenceTransformersEmbeddingProvider`
- `OnnxEmbeddingProvider`
- `LlamaCppLanguageModelProvider`
- `OnnxRerankerProvider`

CPU-профиль должен поддерживать: ограничение числа потоков, ограничение batch size, bounded task queue, backpressure, отмену запроса, лимит памяти, последовательную индексацию при недостатке ресурсов.

### OpenAI-compatible endpoint
Универсальный адаптер для vLLM, Ollama, корпоративного inference gateway:
- `OpenAICompatibleEmbeddingProvider`
- `OpenAICompatibleLanguageModelProvider`

### Облачные модели
Облачный provider разрешается отдельной конфигурацией и policy среды. Облачный runtime не должен включаться автоматически. Передача корпоративного текста во внешнюю модель допускается только при явно разрешённой политике.

### No-LLM mode
Auto-RAG должен поддерживать режим без генеративной модели:
```
query → lexical/vector retrieval → deterministic filters → optional CPU reranker → Evidence
```
В этом режиме отключаются: LLM query rewrite, LLM verifier, генерация ответа, LLM-based decomposition.

## Интерфейсы

```python
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class ModelCapabilities:
    provider_id: str
    model_id: str
    revision: str | None
    local: bool
    offline_capable: bool
    max_batch_size: int
    max_input_tokens: int | None


@dataclass(frozen=True)
class EmbeddingCapabilities(ModelCapabilities):
    dimension: int
    normalized: bool
    similarity_metric: str


class EmbeddingProvider(Protocol):
    @property
    def capabilities(self) -> EmbeddingCapabilities: ...
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class RerankerProvider(Protocol):
    async def rerank(self, query: str, evidence: Sequence["Evidence"], limit: int) -> list["Evidence"]: ...


class LanguageModelProvider(Protocol):
    async def complete(self, request: "CompletionRequest") -> "CompletionResponse": ...
```

## Инвариант размерности embeddings

Каждая ревизия локального vector index жёстко связана с embedding profile:

```python
@dataclass(frozen=True)
class EmbeddingProfile:
    provider_family: str
    model_id: str
    model_revision: str | None
    dimension: int
    normalized: bool
    distance_metric: str
    preprocessing_revision: str
```

Индекс обязан хранить этот профиль в metadata. При открытии индекса Auto-RAG сравнивает его профиль с активным provider.

### Совместимое переключение
Переключение runtime допустимо без переиндексации только тогда, когда сохраняется один и тот же embedding contract:
- одна модель + одна ревизия + одна размерность + одинаковая нормализация + одинаковый preprocessing + одинаковая distance metric.

Пример: `LM Studio endpoint → локальный ONNX runtime` при условии численно совместимых embeddings.

### Несовместимое переключение
Следующие изменения требуют нового индекса: другая embedding model, размерность, нормализация, pooling strategy, query/document prefix policy, несовместимая ревизия модели, distance metric.

Auto-RAG не должен пытаться дополнять вектор нулями, обрезать его или применять произвольную проекцию только ради сохранения размерности. **Совпадение размерности не означает семантическую совместимость.**

## Версионирование индекса

Локальный store поддерживает несколько физических ревизий:
```
indexes/
  profile-e5-base-768/
    revision-0001/
  profile-bge-small-384/
    revision-0001/
```
Активная ревизия выбирается manifest-файлом. Переиндексация выполняется staged-образом: старый индекс остаётся активным → строится новый → integrity check → публикуется атомарно.

## CPU-bounded execution

CPU-only режим — управляемый профиль ресурсов, не просто «модель на CPU».

```yaml
runtime:
  profile: cpu_bounded
  cpu:
    max_threads: 4
    embedding_workers: 1
    reranker_workers: 1
    queue_size: 32
    batch_size: 8
    memory_limit_mb: 4096
```

Планировщик (приоритеты):
1. interactive search
2. fetch requested document
3. incremental sync
4. full rebuild

## Сохранение функциональности при деградации

Auto-RAG использует capability negotiation (`RuntimeCapabilities`: embeddings, lexical_search, reranking, query_rewrite, generation, offline). Pipeline выбирается по доступным возможностям, а не по имени runtime.

- Отказ LLM → query rewrite пропускается, выполняется исходный запрос.
- Отказ reranker → deterministic ranking.
- Отказ embedding во время search → lexical fallback.
- Отказ embedding во время sync → документы в staging, job pending, активный индекс не повреждается.

## Облачная безопасность

Cloud provider работает только при явно выбранной policy (`disabled` / `query_only` / `selected_evidence` / `full`). Default — `disabled`.

## Отказоустойчивость
См. раздел «Сохранение функциональности при деградации».

## Наблюдаемость

Каждый ответ содержит технический режим:
```json
{
  "runtime": {
    "retrieval": "hybrid",
    "embedding_provider": "sentence-transformers",
    "reranker": "disabled",
    "language_model": "none",
    "execution": "cpu"
  }
}
```

## Последствия

### Положительные
- LM Studio перестаёт быть обязательной зависимостью;
- система работает на CPU-only станциях;
- retrieval сохраняется без генеративной модели;
- облачные и локальные runtime взаимозаменяемы;
- индекс защищён от скрытой несовместимости embeddings;
- переход между runtime возможен без rebuild при одинаковой модели;
- деградация предсказуема;
- упрощается тестирование;
- agent integration не зависит от конкретного model server.

### Отрицательные
- потребуется registry провайдеров;
- необходимо хранить embedding manifest;
- логика миграции индексов;
- некоторые runtime могут давать численно отличающиеся embeddings при одинаковой заявленной модели;
- CPU-профиль требует scheduler и backpressure;
- конфигурация сложнее.

## Отвергнутые варианты

- **Обязательный LM Studio** — нельзя гарантировать наличие.
- **Один универсальный LLM-клиент** — роли embeddings и generation имеют разные контракты.
- **Проверка только размерности** — недостаточна для совместимости.
- **Автоматическая проекция между размерностями** — ухудшает качество.
- **Автоматическая отправка данных в облако** — нарушает явное управление передачей.

## Критерии принятия

1. Auto-RAG запускается без LM Studio.
2. MCP `search` работает в CPU-only профиле.
3. MCP `search` работает без `LanguageModelProvider`.
4. Смена language model не требует перестройки vector index.
5. Несовместимая embedding model блокирует vector search с понятным сообщением.
6. Совместимый runtime может открыть существующий индекс.
7. Index manifest содержит полный `EmbeddingProfile`.
8. CPU scheduler отдаёт приоритет интерактивному поиску.
9. Отказ reranker не блокирует поиск.
10. Отказ LLM не блокирует поиск.
11. Cloud provider выключен по умолчанию.
12. Режим lexical-only доступен как аварийная деградация.
13. Staged sync не публикует частично построенный индекс.
14. Тесты проверяют минимум два runtime для одной embedding model.
15. Тесты проверяют отказ при совпадающей размерности, но другой модели.

## Решение

Принять provider-independent model runtime layer. Считать LM Studio одним из опциональных адаптеров, а не частью архитектурного ядра Auto-RAG.

Самое важное архитектурное правило: различать **Model runtime portability** и **Embedding-space compatibility**. Можно перейти с LM Studio на CPU/ONNX или облако, если там выполняется та же embedding model с тем же preprocessing contract. Нельзя без переиндексации перейти на другую модель только потому, что у неё та же размерность.
