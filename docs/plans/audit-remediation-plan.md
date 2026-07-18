# auto-rag Remediation Plan (по аудиту RESULE-FBL.md, rev e43f8d8)

## Контекст
Репозиторий `NikolayGusev-astra/auto-rag` — локальный RAG с DCD-роутингом,
ZVec/Chroma, memvid episodic memory, MCP/федерацией. Аудит `RESULE-FBL.md`
выявил, что значительная часть заявленных фич сломана молча (idiom
`except Exception: pass`), документация расходится с кодом, а contour
самопроверки качества не может честно измерить систему.

Часть находок УЖЕ исправлена предыдущими раундами (не дублировать):
- C2 trafilatura: импорт добавлен (`rag_search.py:24`)
- C3 requests: импорт есть в `rag_search.py`
- C1 `k` NameError: заменено на `max_results` (`rag_async.py:617`)
- C9 приватные хосты: удалены из `chroma/rag_config.py`
- SSRF/S4/S5 частично: JQL-эскейп и query-кодирование поправлены в раундах 1-2
- memvid sidecar → native single-file MV2 (commit e43f8d8)

## Фаза 0 — Верификация состояния (не повторять старое)
1. `grep` по репо на каждый ID из аудита; отметить в `docs/plans/audit-status.md`
   что уже починено vs ещё живо. НЕ править то, что уже зелёное.
2. Запустить `python -m pytest tests/ -q` — зафиксировать baseline (157 passed).

## Фаза 1 — P0: Security & Data Destruction (БЛОКИРУЮЩИЕ)
- [ ] C4: убрать `shutil.rmtree` из read-path `zvec_adapter._ensure_collection`.
      При сбое `zvec.open()` — логировать и возвращать empty, не удалять базу.
      VERIFY: unit-тест, имитирующий lock/OOM при open → коллекция цела.
- [ ] S1-S3: SSRF-guard — `allow_redirects=False`, per-hop re-validation после
      редиректа, пиннинг резолвленного IP, блок `0.0.0.0/8` и IPv4-mapped IPv6,
      переход на `ipaddress.is_global`. VERIFY: тест на `http://169.254.169.254`
      через 302 и на `::ffff:127.0.0.1`.
- [ ] S4-S5: JQL — экранировать `\` до `"`; `{query_and3}` через `quote_plus`.
      VERIFY: тест с завершающим backslash и спецсимволами URL.
- [ ] S6: федерация — `hmac.compare_digest` для ключа, HTTPS/mTLS вместо
      cleartext HTTP, убрать bind 0.0.0.0+auth-off из import-time snapshot.
- [ ] S8: stdio-MCP подпроцессы — не наследуют весь `os.environ`; передавать
      только нужные переменные.
- [ ] S9: федерация — hop-counter / origin-заголовок против петель.

## Фаза 2 — P1: Функциональность, которая «работает» только на бумаге
- [ ] C6: форма memory-хита. `recall()` должен возвращать результат с ключом
      `chunks` и нормализованным `sources` (dict, не list). Это чинит eval-оценку
      хитов, canary-hit-rate, federated endpoint и падение CLI одним махом.
      VERIFY: `async_rag_search` с memvid-хитом → потребитель получает chunks.
- [ ] C8: `hermes_memory_cli.py` compact/purge — переписать на «новый файл +
      atomic replace» (как уже сделано для migration). Починить `--before`.
      VERIFY: round-trip test — compact уменьшает frames, purge удаляет.
- [ ] C5: `canary_deploy.py` main-flow уничтожает кандидата (baseline vs baseline).
      Восстанавливать кандидата перед прогоном; не затирать WATCHED_FILES.
      VERIFY: tmpdir-test — canary не портит рабочую копию.
- [ ] C2/C3 (остатки): `_blocking_web` обогащает только 1 страницу; вынести
      `return` из цикла в `rag_search.py`. VERIFY: web-fallback реально возвращает.
- [ ] Инкрементальные индексаторы: удалять чанки изменённых/удалённых файлов;
      не помечать `done` при упавшем батче. VERIFY: правка файла → старые чанки ушли.
- [ ] Zero-vector-отравление: сбой эмбеддинга = пропуск/ошибка, никогда `[0.0]*1024`
      в индекс. VERIFY: тест на embedding-failure path.

## Фаза 3 — P2: Концептуальный ремонт
- [ ] DCD confidence: нормализовать по matched-mass или margin; перетюнить
      пороги 0.3/0.2/0.5 вместе; добавить кириллицу в техдомены (или centroid-kNN
      поверх существующих эмбеддингов). VERIFY: русские запросы → правильный домен.
- [ ] DCD retrieval-игнор: либо подключить `primary_source`/collection к ZVec,
      либо удалить мёртвый выход роутера. VERIFY: DCD collection реально используется.
- [ ] memvid: дефолт на augmentation (Option B из `integration_patch.py`),
      гейт short-circuit через `_llm_verify`, TTL + инвалидация по `--clear`,
      счётчик хитов, использование `feedback`. VERIFY: хит не возвращает
      устаревший контент после переиндексации.
- [ ] Приватность: веб-поиск opt-in per domain или гейтирован confidence/score.
- [ ] Eval: один харнесс; golden set ≥100 с негативами/парафразами/русским;
      починить парсер судьи (запятая, целые); убрать «только вверх» fallback.
- [ ] Дедупликация дерева: удалить `chroma/` (адаптер есть в rag_core), починить
      или удалить `rag_v2/` + `streaming.py` + `rag_search.py`, один неймспейс env.

## Фаза 4 — P3: Гигиена
- [ ] ruff BLE001/S110 в CI — запретить `except Exception: pass` в hot paths.
- [ ] pyflakes/ruff: ~35 unused imports, дубли, unreachable code.
- [ ] Удалить тавтологичные тесты (`test_golden_set_consistency`, `test_canary_deploy`
      verdict-or, `test_memvid_smoke` bool-in).
- [ ] Запинить зависимости (lock-файл); привести pyproject ↔ requirements в синхрон.
- [ ] `.gitignore` покрыть golden_eval_report.json, canary_reports/, .canary_baseline/.

## Правила выполнения
- Каждый фикс — отдельный коммит с ID бага в сообщении (C4/S2/...).
- VERIFY — это не только unit-тест, но и (где возможно) прогон живого пайплайна
  с реальным LM Studio / ZVec.
- Не трогать живую capsule памяти без backup (`.legacy.bak` уже есть).
- После каждой фазы — `pytest tests/ -q` и обновление `docs/plans/audit-status.md`.

## Статус
- P0 C4, S1-S3, S4-S5, S6, S8, S9 — TODO
- P1 C6, C8, C5, C2/C3-остатки, инкрементальные индексаторы, zero-vector — TODO
- P2 DCD, memvid-augmentation, приватность, eval, дедупликация — TODO
- P3 ruff, pyflakes, тесты, lock, gitignore — TODO
