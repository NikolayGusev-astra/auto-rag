# Задача: MEMVID-001 — адаптация memvid_memory.py под memvid-sdk 2.0.160

## Роль
Ты — senior-инженер, работающий над проектом auto-rag (локальный RAG-пайплайн с опциональным слоем эпизодической памяти memvid).
Действуй по TDD: сначала тест, воспроизводящий баг, затем фикс.

## Контекст (читай перед работой)
- Архитектурные решения: отсутствуют в виде ADR в этом деплое; слой спроектирован dependency-optional (memvid_memory.py:39-47).
- Контракт/спецификация: docstring `MemvidMemory` (memvid_memory.py:1-63) — recall() до RAG, record() после, degrade gracefully.
- Тех-стек: Python 3.11, memvid-sdk==2.0.160 (модуль `memvid_sdk`, НЕ `memvid`), LM Studio на localhost:1234 (OpenAI-совместимый `/v1/embeddings`, модель bge-m3, 1024d).

## Проблема
Слой эпизодической памяти не работает: при установленном `memvid-sdk` 2.0.160 и `RAG_MEMVID_ENABLED=true` вызов `MemvidMemory.for_tenant(...)` возвращает неактивный бэкенд (`active == False`), recall() всегда `[]`, record() — `False`. Память фактически мертва, хотя SDK установлен.

## Root cause (из аудита)
`memvid_memory.py:497` и `_RealMemvidBackend` (строки 209-373) написаны под СТАРЫЙ пакет `memvid` (модуль `memvid`, API `memvid.Memvid.create/open`, `put_bytes_with_options`, `SearchRequest`). На машине установлен `memvid-sdk` 2.0.160, который экспортирует модуль `memvid_sdk` с принципиально другим API:
- `import memvid` → `ModuleNotFoundError` → попадает в `except ImportError` → noop (memvid_memory.py:497-501).
- Даже при исправлении имени модуля на `memvid_sdk`, API несовместим:
  - капсула создаётся `memvid_sdk.create(filename, kind="basic", enable_vec=True)` (memvid_memory.py:233-244 зовёт `M.create(path)` — несовпадение сигнатуры);
  - запись — НЕ `put_bytes`, а `add_memory_cards([{entity, slot, value, kind, tags}])` (SPO-триплеты, обязательны `entity`+`slot`+`value`);
  - поиск — НЕ `search(SearchRequest)`, а `find(query, k=, embedder=EmbeddingProvider)` → возвращает `dict` с ключом `'hits'` (list of dicts), а не объект с `.hits`.
Контракт требует: слой должен реально писать эпизоды и находить их по смыслу при наличии `memvid_sdk` 2.0.160 + LM Studio embedding.

## Definition of Done
1. Написан integration-тест `tests/test_memvid_sdk2.py`, который падает на текущем коде (RED): создаёт `MemvidMemory` с `RAG_MEMVID_ENABLED=true` и реальным `memvid_sdk` в venv, пишет `Episode`, делает `recall()` и ожидает `len(hits) >= 1` и `top_score > 0`.
2. Внесён минимальный фикс в `_RealMemvidBackend` (memvid_memory.py): import `memvid_sdk` как приоритет, `create(kind="basic", enable_vec=True)`, `add_memory_cards` с маппингом Episode→SPO, `find` с `LMStudioEmbedder(EmbeddingProvider)` для vec-поиска, обработка `dict['hits']`. Тест проходит (GREEN).
3. Не сломаны существующие тесты: `pytest tests/ -q` — зелёные (в т.ч. на машине без SDK слой остаётся noop и не падает).
4. Удалён dead code, маскировавший баг: ветки в `_open_capsule`/`put`/`search`, обращающиеся к несуществующему API старого `memvid` (`put_bytes_with_options`, `M.create(path)`, `SearchRequest`).
5. (опц.) Обновлён docstring `MemvidMemory`, если контракт импорта изменился (`memvid_sdk` вместо `memvid`).

## Ограничения
- Не переписывай архитектуру — только целевой фикс бэкенда под SDK 2.0.160.
- Не добавляй новые pip-зависимости без явного ADR: используй уже установленный `memvid-sdk` + `requests` (есть в стеке). `LMStudioEmbedder` — это код-адаптер, не новая зависимость.
- Сохраняй стиль окружающего кода (dataclass Config, Protocol-бэкенд, graceful degrade).
- noop-fallback при отсутствии SDK обязан остаться (rag_async не должен падать без memvid).
