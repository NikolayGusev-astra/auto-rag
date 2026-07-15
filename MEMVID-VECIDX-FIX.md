# Задача: MEMVID-002 — локальный vec-индекс для semantic recall поверх memvid-sdk 2.0.160

## Роль
Ты — senior-инженер, работающий над проектом auto-rag (локальный RAG-пайплайн с опциональным слоем эпизодической памяти memvid).
Действуй по TDD: сначала тест, воспроизводящий баг, затем фикс.

## Контекст (читай перед работой)
- Архитектурные решения: отсутствуют в виде ADR в этом деплое; слой спроектирован dependency-optional (memvid_memory.py:39-47). Предыдущий фикс MEMVID-001 (коммит 3c1fc88) подключил backend к memvid-sdk 2.0.160: `import memvid_sdk`, `create(kind="basic", enable_vec=True)`, `add_memory_cards` (SPO), `find(embedder=LMStudioEmbedder)`.
- Контракт/спецификация: docstring `MemvidMemory` (memvid_memory.py:1-63) — recall() ДОЛЖЕН находить записанный эпизод по смыслу ДО RAG-пайплайна (short-circuit), record() сохраняет его после.
- Тех-стек: Python 3.11, memvid-sdk==2.0.160 (модуль `memvid_sdk`, НЕ `memvid`), LM Studio на localhost:1234 (OpenAI-совместимый `/v1/embeddings`, модель `text-embedding-baai-bge-m3-568m`, 1024d). `_Embedder`/`LMStudioEmbedder` уже умеют эмбеддить через LM Studio (с no-proxy fix из MEMVID-001).

## Проблема
После MEMVID-001 backend стал РЕАЛЬНЫМ (`active=True`), `record()` реально пишет SPO-карточку в capsule (`add_memory_cards` → `added:1`, `commit()` ок). Но `recall()` ВСЕГДА возвращает `[]` — ни по точной фразе, ни по смыслу. Short-circuit памяти не срабатывает, RAG каждый раз идёт в полный пайплайн, записанная память никогда не читается. Эпизодическая память функционально мертва, хотя «запись работает».

## Root cause (из аудита + эксперимента)
`memvid_memory.py:295-337` (`_search_native`) вызывает `self._mem.find(query, embedder=LMStudioEmbedder(...))`. memvid-sdk 2.0.160 в режиме `create(kind="basic")` НЕ строит поисковый индекс из `add_memory_cards` без managed embedding/LLM-бэкенда: `doctor --rebuild_vec_index` показывает `vec=0 dim=0`, а `find()` возвращает `engine: tantivy` (lex-only) и `hits: []` даже на точном совпадении фразы. То есть SDK принимает карточки, но не индексирует их для поиска в локальном basic-режиме.
Контракт требует: `recall()` должен находить записанный эпизод по смыслу (semantic), иначе слой бесполезен — RAG не получает приоритетный ответ из собственной истории.

## Definition of Done
1. Написан integration-тест `tests/test_memvid_vecidx.py`, который падает на текущем коде (RED): с `RAG_MEMVID_ENABLED=true` + реальным `memvid_sdk` пишет Episode, делает `recall()` по СМЫСЛУ (другая формулировка того же вопроса) и ожидает `len(hits) >= 1` и `top_score > 0`.
2. Внесён минимальный фикс в `_RealMemvidBackend` (memvid_memory.py): добавлен лёгкий персистентный vec-индекс поверх SDK — при `put()` эмбеддить `value` карточки через `LMStudioEmbedder` и дописать `{entity, vec, payload}` в файл индекса рядом с capsule (`<capsule>.vecidx.jsonl`); при `search()` прочитать индекс, косинус-ранжировать query-вектор, вернуть top-k как hits. `find()` SDK оставить как опциональный fallback (если вдруг вернёт непусто). Тест проходит (GREEN).
3. Не сломаны существующие тесты: `pytest tests/ -q` зелёный и на системном python (noop, 30 passed/1 skipped), и на venv с SDK (33 passed/1 xfailed из MEMVID-001 остаётся xfailed).
4. Удалён dead code, маскировавший баг: ветка `_search_manual_fallback` (memvid_memory.py:339-378), которая пыталась дёлать `list_frames`/`iter_frames` у SDK 2.0.160 — этих методов нет, блок всегда возвращал `[]` и создавал ложное впечатление, что fallback работает.
5. (опц.) Обновлён docstring `MemvidMemory`, если контракт изменился (теперь semantic recall гарантируется локальным vec-индексом, не зависит от managed SDK-индекса).

## Ограничения
- Не переписывай архитектуру — только целевой фикс: добавить vec-индекс в `_RealMemvidBackend`. `recall()`/`record()` публичный API не меняется.
- Не добавляй новые pip-зависимости без явного ADR: используй stdlib (`json`, `os`, `threading` уже есть) + уже установленный `memvid-sdk` + `requests` (есть в стеке). Локальный vec-индекс — это код, не новая зависимость.
- Сохраняй стиль окружающего кода (dataclass Config, Protocol-бэкенд, graceful degrade: если LM Studio недоступен — эмбеддинг пустой → индекс не пишется → recall возвращает [] но НЕ падает).
- noop-fallback при отсутствии SDK обязан остаться (rag_async не должен падать без memvid).
- Персистентность: vec-индекс должен переживать перезапуск процесса (RAG-воркер и вызывающий — разные процессы), поэтому jsonl рядом с capsule, а не только in-memory dict.
