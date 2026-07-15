# Plan: Интеграция memvid memory layer в auto-rag

**Дата:** 2026-07-08
**Planner:** tencent/hy3:free
**Implementer:** poolside/laguna-m.1:free
**Reviewer:** nvidia/nemotron-3-ultra-550b-a55b:free

## Контекст (что интегрируем)

Скачан архив `hermes-memvid.tar.gz` (лендинг https://z1dru5tybdu1-d.space-z.ai/).
5 модулей: `memvid_memory.py`, `memvid_trace.py`, `memvid_canary.py`,
`hermes_memory_cli.py`, `integration_patch.py`. Они написаны под generic auto-rag,
но у нашего пайплайна есть расхождения, мешающие "drop-in":

1. **RagTrace несовместим.** У нас `rag_core/rag_trace.py` — класс `RagTrace`
   (объект, атрибут `.stages`, поля `duration_ms` / `total_ms`).
   В `memvid_trace.py` ожидается **plain dict** `{"stages": []}` с полями
   `latency_ms` / `total_latency_ms`.
2. **Схема отчёта eval_golden не бьётся с memvid_canary.**
   Наш `eval_golden.py` пишет `{meta, summary, results:[{id, source_ok,
   answer_verdict, total_latency_s, dcd_collection, ...}]}`. Нет `score` (0..1),
   нет `from_memory`. Пути `REPORT_PATH`/`GOLDEN_PATH` захардкожены — не принимает
   `--out` / `--golden`.
   `memvid_canary._load_report` ищет в каждом вопросе `score`, `latency_ms`,
   `trace`, `from_memory` → у нас всё это отсутствует → canary покажет rollback
   на ровном месте.
3. **Env-имена embedding не совпадают.**
   У нас: `EMBEDDING_MODEL="text-embedding-baai-bge-m3-568m"` + `EMBEDDING_URL`.
   У них: `RAG_MEMVID_EMBED_MODEL="bge-m3"` + fallback `RAG_EMBEDDING_URL`.
4. **memvid-sdk не установлен** (`import memvid` → ModuleNotFoundError).
   Код корректно degrade-to-noop, но реально не работает без `pip install memvid-sdk>=2.0`.
5. **Federation не учтён.** `rag_federated.py` + `federated_endpoint.py` (multi-node).
   memvid-capsule — локальный файл per-tenant. В federation не реплицируется.
   Решение: пока LOCAL-ONLY, документировать ограничение в README.

## Модель выполнения

- Каждый task → отдельный git worktree от `main`, branch `implement/<task-id>`.
- TDD обязателен: failing test → implement → pass → full suite → commit.
- Субагенты НЕ получают историю чата, только task spec (ниже).
- Две стадии ревью (spec compliance + quality) одним nemotron-агентом на task
  с комбинированным чеклистом (free-tier pragmatism).
- Интеграционный ревью отдельным nemotron-агентом после всех task.

## Task graph

```
T1 (adapter)      ──┐
                    ├──> T3 (env+wiring) ──> T4 (install+smoke)
T2 (eval-compat)  ──┘
```

---

## TASK-1: RagTrace adapter для memvid_trace

**Objective:** `MemvidTraced` должен принимать объект `RagTrace` (наш) вместо dict.

**Files:**
- NEW `rag_core/memvid_trace_adapter.py` — адаптер поверх `memvid_trace.MemvidTraced`
- MODIFY `rag_core/memvid_trace.py` — разрешить `trace` быть `RagTrace`-объектом
  (duck-typing: если есть `.stages` атрибут и метод `.event()`/`.stage()`,
  писать через него; иначе dict-путь как сейчас)
- NEW `tests/test_memvid_trace_adapter.py`

**Поведение:**
- `recall(query, ..., trace=RagTrace)` → добавляет stage
  `{"stage":"memvid.recall","duration_ms":...,"hits":...,"top_score":...,...}`
  в `trace.stages` через `trace.event("memvid.recall", ...)` или `trace.stage()`.
- `record(...)` → stage `memvid.record`.
- Маппинг имён: `latency_ms`→`duration_ms`, `total_latency_ms`→`trace.total_ms`
  (RagTrace уже считает total_ms из duration_ms, не дублировать).
- `recall_as_context` возвращает prompt-префикс как сейчас (без зависимости от trace-типа).

**TDD:**
1. Test: создаём `RagTrace("q","astra")`, вызываем `recall` с mock backend
   (return [Episode(score=0.9)]), проверяем `len(trace.stages)==1` и
   `trace.stages[0]["stage"]=="memvid.recall"` и `duration_ms>0`.
2. Test: `record` добавляет `memvid.record` stage.
3. Test: при `trace=None` ничего не падает, stages пусты.

**Команды:**
```
cd /tmp/worktree-T1
python3 -m pytest tests/test_memvid_trace_adapter.py -x  # FAIL
# implement
python3 -m pytest tests/test_memvid_trace_adapter.py      # PASS
python3 -m pytest                                          # no regressions
git add -A && git commit -m "feat: RagTrace adapter for memvid_trace"
```

**Исходники memvid_trace.py лежат в** `/tmp/memvid_extract/hermes/memvid_trace.py`
(скопировать в worktree `rag_core/` как основу перед правкой).

---

## TASK-2: Совместимость eval_golden ↔ memvid_canary

**Objective:** `eval_golden.py` выдаёт схему, которую понимает `memvid_canary._load_report`.

**Files:**
- MODIFY `rag_core/eval_golden.py`
- MODIFY `rag_core/memvid_canary.py` (`_load_report` — сделать tolerant к нашей схеме)
- NEW `tests/test_eval_golden_compat.py`

**Правки eval_golden.py:**
- Добавить `argparse`: `--out` (переопределяет `REPORT_PATH`), `--golden`
  (переопределяет `GOLDEN_PATH`). Без аргументов — поведение как сейчас.
- В каждый элемент `results` дописывать:
  - `"score"`: нормализация `answer_verdict` → correct=1.0, partial=0.5, incorrect=0.0
    (если verdict отсутствует — 0.0).
  - `"from_memory"`: bool (пока всегда `false`; станет true после T3 wiring).
  - `"latency_ms"`: `total_latency_s * 1000`.
- Структура верхнего уровня отчёта не меняется (`{meta, summary, results}`).

**Правки memvid_canary._load_report:**
- Сейчас ищет `score`, `latency_ms`, `trace`, `from_memory` в каждом `q`.
- Добавить tolerant-парсинг:
  - `score`: `q.get("score") or q.get("llm_judge_score") or (1.0 if answer_verdict=="correct" else ...)`.
  - `latency_ms`: `q.get("latency_ms") or q.get("total_latency_s",0)*1000`.
  - `from_memory`: `q.get("from_memory", False)`.
- Если `results` пуст, fallback на `raw.get("summary")` (mean_score и т.п.).

**TDD:**
1. Test: сформировать mock-отчёт В СТАРОМ формате (results без score/latency_ms,
   с answer_verdict + total_latency_s), прогнать через `_load_report`,
   проверить `mean_score>0`, `p99_latency>0`.
2. Test: `--out`/`-golden` аргументы парсятся (без запуска полного eval —
   проверить через `argparse` parse только).

**Команды:**
```
cd /tmp/worktree-T2
python3 -m pytest tests/test_eval_golden_compat.py -x  # FAIL
# implement
python3 -m pytest tests/test_eval_golden_compat.py      # PASS
python3 -m pytest
git add -A && git commit -m "feat: eval_golden compat with memvid_canary"
```

---

## TASK-3: Env-bridge + wiring в пайплайн

**Objective:** Маппинг embedding-env + recall→RAG→record блок в `rag_async.py`.

**Files:**
- NEW `rag_core/memvid_config_bridge.py`
- MODIFY `rag_core/memvid_memory.py` (`MemvidConfig.from_env` — добавить
  `EMBEDDING_URL`/`EMBEDDING_MODEL` как fallback источники)
- MODIFY `rag_core/rag_async.py` — опциональный recall→RAG→record,
  активируется `RAG_MEMVID_ENABLED`
- NEW `tests/test_memvid_wiring.py`

**Правки memvid_config_bridge.py:**
```python
def bridge_memvid_env():
    # если RAG_MEMVID_EMBED_MODEL не задан — взять EMBEDDING_MODEL
    # если RAG_MEMVID_EMBED_URL не задан — взять EMBEDDING_URL
    # если RAG_MEMVID_EMBED_DIM не задан — взять EMBEDDING_DIM
    # выставить os.environ[...] если отсутствуют
```
- Вызывать `bridge_memvid_env()` внутри `MemvidConfig.from_env` ДО чтения полей
  (или в `__init__` backend).

**Правки memvid_memory.py:**
- В `from_env`: `embed_url` fallback chain:
  `RAG_MEMVID_EMBED_URL` → `RAG_EMBEDDING_URL` → `EMBEDDING_URL` → default.
- `embed_model` fallback: `RAG_MEMVID_EMBED_MODEL` → `EMBEDDING_MODEL` → default `bge-m3`.

**Wiring в rag_async.py:**
- Импорт `MemvidTraced` + `Episode` + адаптер из T1.
- В `async_rag_search` (или его синхронную обёртку), ДО основного пайплайна:
  ```python
  if memvid_enabled:
      priors = _memory.recall(query, domain=domain, trace=trace)
      if priors and priors[0].score >= _memory.recall_threshold:
          return priors[0].answer, priors[0].sources, trace  # short-circuit
  # ... нормальный RAG ...
  if memvid_enabled:
      _memory.record(Episode(query=..., answer=..., ...), trace=trace)
  ```
- `trace` здесь — НАШ `RagTrace` (объект), адаптер из T1 обрабатывает.
- При `RAG_MEMVID_ENABLED=false` (по умолчанию) — блок полностью выключен,
  _memory НЕ создаётся (lazy init при первом enabled-вызове).

**TDD:**
1. Test: `RAG_MEMVID_ENABLED=false` → `async_rag_search` работает как раньше,
   recall НЕ вызывается (mock backend count==0).
2. Test: `RAG_MEMVID_ENABLED=true` + mock backend → recall вызван перед RAG,
   record вызван после.
3. Test: `bridge_memvid_env()` корректно маппит `EMBEDDING_MODEL`→`RAG_MEMVID_EMBED_MODEL`.

**Команды:**
```
cd /tmp/worktree-T3
python3 -m pytest tests/test_memvid_wiring.py -x  # FAIL
# implement
python3 -m pytest tests/test_memvid_wiring.py      # PASS
python3 -m pytest
git add -A && git commit -m "feat: memvid env-bridge + rag_async wiring"
```

**Зависит от T1** (адаптер trace). В worktree T3 скопировать итоговый
`memvid_trace_adapter.py` из T1 (или реимплементировать совместимо).

---

## TASK-4: Install + smoke

**Objective:** memvid-sdk ставится, noop-path проходит smoke.

**Files:**
- MODIFY `requirements.txt` (дописать `memvid-sdk>=2.0`)
- NEW `scripts/install_memvid.sh`
- NEW `tests/test_memvid_smoke.py`

**Правки:**
- `requirements.txt`: добавить строку `memvid-sdk>=2.0` (рядом сrequests).
- `scripts/install_memvid.sh`: `pip install "memvid-sdk>=2.0"`; проверка
  `python3 -c "import memvid"`; если fail — warn, но не exit 1 (noop режим
  всё равно работает).
- `tests/test_memvid_smoke.py`: запуск `memvid_memory.py` с
  `RAG_MEMVID_ENABLED=false` → import + `MemvidMemory.for_tenant("smoke")`
  → `recall` возвращает `[]`, `record` не падает. Проверка noop-contract.

**TDD:**
1. Test: noop path (enabled=false) — recall=[] , record OK, не падает.
2. Test: при missing memvid-sdk + enabled=true — backend становится Noop,
   recall=[] (graceful).

**Команды:**
```
cd /tmp/worktree-T4
python3 -m pytest tests/test_memvid_smoke.py -x  # FAIL (нет зависимостей/модулей)
# implement
python3 -m pytest tests/test_memvid_smoke.py      # PASS
python3 -m pytest
git add -A && git commit -m "feat: memvid install + smoke tests"
```

---

## Интеграционный ревью (nemotron)

После T1-T4: объединить diff, прогнать `python3 -m pytest` на merged worktree,
проверить что:
- noop-contract не нарушен (RAG_MEMVID_ENABLED=false → поведение идентично baseline)
- trace-совместимость (RagTrace stages пишутся)
- canary сможет запустить eval_golden с --out/--golden
- нет дублей кода между T1/T3

Verdict: pass → merge в main.
