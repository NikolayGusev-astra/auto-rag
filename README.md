# Autolycus RAG v2

Готовый RAG-пайплайн для автономного использования. Работает на ZVec 0.5.1 + bge-m3 + MCP/SearXNG. Не требует GPU (CPU достаточно), не требует внешних API (кроме опционального Context7).

## Быстрый старт

```bash
# 1. Зависимости
pip install zvec trafilatura

# 2. SearXNG (опционально, для web fallback)
docker run -d --name searxng -p 8080:8080 searxng/searxng

# 3. Embedding (один из вариантов):
#    a) LM Studio с bge-m3 моделью (локально, GPU/CPU)
#    b) sentence-transformers (CPU, ~0.3-0.5s на запрос)
#       → python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# 4. Конфиг (опционально)
cp .env.example .env
# RAG_EMBEDDING_URL — если другой порт/сервер
# RAG_SEARXNG_URL — если SearXNG не на localhost:8080
# CONTEXT7_API_KEY — для MCP fallback (инженерная документация)

# 5. Запуск
python3 indexer.py --clear        # первая индексация ~14 минут на 2600+ файлов
python3 rag_search.py "ваш запрос"
python3 run_golden.py             # прогон тестов
```

## Архитектура

```
User Query → DCD Router (16 domains, 0-50ms)
  → [conf < 0.1] → reject
  → ZVec bge-m3 search (0.3s, 22566 docs)
    → sessions ZVec (если wiki пуст)
    → Entity Match (≥50% entities?)
      → [pass] → ANSWER (score ≥ 0.4)
      → [fail] → MCP / SearXNG / Trafilatura
```

## Структура

```
rag-v2/
├── rag_core/
│   ├── rag_config.py          — конфиг (env vars, без API ключей)
│   ├── dcd_router.py          — классификатор 16 доменов
│   ├── rag_search.py          — поиск (синхронный)
│   ├── rag_async.py           — поиск (асинхронный, asyncio.gather)
│   ├── zvec_adapter.py        — ZVec адаптер (curl, не requests)
│   ├── rag_mcp_client.py      — MCP клиент
│   ├── indexer.py             — индексатор файлов → ZVec
│   ├── benchmark_rag.py       — замеры скорости
│   ├── run_golden.py          — прогон golden set
│   └── rag_golden.json        — 20 тестовых запросов
├── scripts/
│   └── docker-searxng.sh      — запуск SearXNG
├── requirements.txt
├── .env.example
├── docker-compose.yml         — SearXNG + (опционально)
└── README.md
```

## Зависимости

Обязательные: `zvec`, `trafilatura`
Опциональные: `sentence-transformers` (CPU embedding без LM Studio)
SearXNG: Docker контейнер (`searxng/searxng`)

## Известные проблемы ZVec 0.5.1

### 1. LOCK file — RuntimeError при открытии коллекции

**Симптом:** `RuntimeError: Can't open lock file: /path/to/collection/LOCK`

**Причина:** ZVec не может создать LOCK-файл на некоторых ФС/ядрах. Файл не существует, но zvec.open() падает.

**Решение:** Создать пустой LOCK перед zvec.open():

```python
lock_path = os.path.join(coll_path, "LOCK")
if os.path.exists(coll_path) and not os.path.exists(lock_path):
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o644)
        os.close(fd)
    except OSError:
        pass
coll = zvec.open(coll_path)
```

Функция `ensure_zvec_lock()` встроена в `rag_config.py`.

### 2. ID length limit — "contains invalid characters"

**Симптом:** `Invalid doc: doc[...] contains invalid characters` при insert

**Причина:** ZVec не документирует, но **ID документа не может быть длиннее 64 символов**. Если ID длиннее — получаете "invalid characters".

Также запрещены символы: `:`, `#`, `/`, `[`, `]`, `..`

**Решение:** Генерировать ID длиной ≤64 символов, только alphanumeric + underscore:

```python
import re
safe = re.sub(r'[^a-zA-Z0-9]', '_', raw_source)
if len(safe) > 64:
    safe = safe[:51] + hashlib.md5(safe.encode()).hexdigest()[:12]
```

Функция `_safe_id()` встроена в `indexer.py`.

### 3. FTS index не строится автоматически

**Симптом:** `coll.query(fts=...)` возвращает 0 результатов, хотя doc_count > 0

**Причина:** FTS индекс не создаётся при insert через Python API. Нужен отдельный вызов optimization.

**Решение:** После вставки всех документов запустить:
```python
from zvec import OptimizeOption
coll.optimize(OptimizeOption())
```

Либо не использовать FTS — только vector search (как в этой версии).

### 4. Нельзя открыть коллекцию дважды в одном процессе

**Симптом:** После `zvec.open()` или `zvec.create_and_open()` повторный вызов `zvec.open()` на ту же коллекцию падает с `Can't lock read-write collection`.

**Причина:** RocksDB удерживает блокировку на процесс.

**Решение:** Кэшировать объект коллекции в module-level переменной:
```python
_cached = None
def get_collection():
    global _cached
    if _cached: return _cached
    _cached = zvec.open(path)
    return _cached
```

### 5. index_completeness показывает 0 при рабочем поиске

**Симптом:** `coll.stats.index_completeness.embedding = 0.0`, но поиск работает

**Причина:** ZVec метрика обновляется асинхронно. Это **не означает** что поиск сломан.

**Решение:** Игнорировать. Проверять работоспособность через `coll.query()`.

### 6. Python requests не работает с localhost:1234 (LM Studio)

**Симптом:** `requests.post(...)` к `http://localhost:1234/v1/embeddings` зависает или Connection timed out, хотя curl работает.

**Причина:** Баг Python requests + connection pooling + localhost.

**Решение:** Использовать subprocess curl для эмбеддингов:

```python
import subprocess, json
r = subprocess.run(["curl", "-s", "--max-time", "10", url, "-d", payload, "-H", "Content-Type: application/json"], ...)
```

### 7. create_and_open падает если директория существует

**Симптом:** `ValueError: path validate failed: path[...] exists`

**Причина:** `create_and_open` создаёт НОВУЮ коллекцию и не может перезаписать существующую.

**Решение:** Использовать create_and_open только для пустых/новых путей. Для существующих:

```python
try:
    coll = zvec.create_and_open(path, schema)
except ValueError:
    coll = zvec.open(path)
```

### 8. Filter синтаксис — отличный от SQL

**Симптом:** `coll.query(..., filter='category == "ford-club"')` падает с синтаксической ошибкой

**Причина:** Zvec filter использует ОДИНАРНЫЙ `=`, не `==` и не `LIKE`.

**Правильно:** `filter='category = "ford-club"'`
**Неправильно:** `filter='category == "ford-club"'` или `filter='category LIKE "ford-club"'`

### 9. Параметр topk, не top_k

**Симптом:** `TypeError: unexpected keyword argument 'top_k'`

**Причина:** ZVec использует `topk`, не `top_k`.

### 10. ZVec FastAPI сервер (альтернатива LOCK workaround)

Вместо `ensure_zvec_lock()` можно запустить долгоживущий процесс, который держит коллекцию открытой. Клиенты шлют HTTP-запросы:

```bash
# Запуск сервера
python3 rag_core/zvec_server.py  # порт 8765

# Использование
curl http://localhost:8765/search?q=Ford+Explorer+шрус&topk=5
curl http://localhost:8765/stats
curl http://localhost:8765/health
```

**Когда нужно:** На Windows (LOCK баг не чинится touch), при высокой нагрузке (много клиентов).
**Когда не нужно:** Для одного пользователя на Linux — `ensure_zvec_lock()` проще.

## Работа без LM Studio (CPU только)

1. Установить `sentence-transformers`:
```bash
pip install sentence-transformers
```

2. В `.env` указать:
```bash
RAG_EMBEDDING_URL=sentence-transformers
RAG_EMBEDDING_MODEL=BAAI/bge-m3
```

3. Модель загрузится один раз (~1.1GB RAM). Скорость ~0.3-0.5s на запрос на CPU.

## Работа без SearXNG

Установить `RAG_WEB_SEARCH=false` в `.env`. RAG будет работать только по ZVec коллекции.

## Конфигурация (.env)

```bash
# Embedding
RAG_EMBEDDING_URL=http://localhost:1234/v1/embeddings
RAG_EMBEDDING_MODEL=text-embedding-baai-bge-m3-568m
RAG_EMBEDDING_DIM=1024

# Web search
RAG_WEB_SEARCH=true
RAG_SEARXNG_URL=http://localhost:8080
RAG_SEARXNG=true

# MCP (Context7 — инженерная документация)
RAG_MCP=false
CONTEXT7_API_KEY=
```

## Тесты

```bash
# 20 тестовых запросов, 5 доменов
python3 run_golden.py
# Ожидаемый результат: Recall@5 ≥ 90%
```

## Лицензия

MIT. Сделано для Autolycus Agent (Nous Research Hermes fork).

## Скил: как агент создаёт и расширяет RAG

### Структура llm-wiki

Индексатор ищет `.md` файлы в `WIKI_PATHS` (по умолчанию `~/wiki`). На новой машине:

```bash
mkdir -p ~/wiki/{your-domain,concepts,manuals,sessions}
mkdir -p ~/llm-wiki
```

Агент наполняет wiki созданием `.md` файлов:

```bash
cat > ~/wiki/your-domain/your-topic.md << 'EOF'
---
title: Название
tags: ["тег1", "тег2"]
---
# Content here
EOF
python3 rag_core/indexer.py --incremental
```

Ссылки между страницами: `[[Название страницы]]`. Индексатор парсит frontmatter (title, tags) и разбивает на чанки по ##/###.

### Как создать новый домен

**Шаг 1: Анализ данных.** Агент анализирует массив — Obsidian, Telegram, CRM, почту — и выделяет кластеры ключевых слов (TF-IDF, LDA, или просто top-N):

```python
# Упрощённо: агент собирает 5-10 ключевых слов домена
keywords = {
    "supports": ["техподдержка", "тикет", "инцидент", "sla"],
    "sales": ["коммерческое", "предложение", "договор", "счёт"],
}
```

**Шаг 2: Добавить в DCD роутер.** В `dcd_router.py` дописать домен в `DOMAIN_KEYWORDS`:

```python
"supports": {
    "weight": 3,
    "keywords": {"техподдержка": 4, "тикет": 3, "sla": 3},
    "collections": {"supports": ["техподдержка", "тикет", "инцидент"]},
}
```

**Шаг 3: Добавить путь в индексатор.** В `indexer.py` дописать `WIKI_PATHS`:

```python
WIKI_PATHS = [
    "~/wiki",
    "~/llm-wiki",
    "~/projects/supports/docs",  # новый источник
]
```

**Шаг 4: Добавить коллекцию в поиск (опционально).** Если нужна отдельная коллекция — в `rag_search.py`:

```python
self.zvec_supports = ZVecSearcher("supports")
# в search(): chunks += self.zvec_supports.search(query, topk=k)
```

**Шаг 5: Golden set.** 5 тестовых запросов от пользователя → `rag_golden.json`.

### Что делает агент автоматически

1. Получает доступ к источникам (Obsidian, TG, CRM)
2. Выгружает `.md` в `~/wiki/{domain}/`
3. Добавляет DCD правила если новый паттерн
4. Шепчет `python3 rag_core/indexer.py --incremental` (cron делает это каждый час)
5. Прогоняет golden set, пишет recall

### Интеграция источников

- **Obsidian:** скопировать `.md` в `~/wiki/` или указать `WIKI_PATHS=/path/to/obsidian`
- **Telegram:** выгрузить чат через TG-клиент → `~/wiki/sessions/`
- **CRM/почта:** через MCP или прямой экспорт → `~/wiki/sales/`
- **Web:** прикрутить скрапер → `~/wiki/research/`

Индексатор всё подхватит. Категории определяются по первой папке пути. Если папка совпадает с именем домена — DCD может фильтровать.

### Корпоративный поиск через SearXNG (авторизация, приоритетные домены)

SearXNG умеет искать по конкретным доменам и с авторизацией. Это полезно для корпоративных порталов, wiki, jira — доступных только внутри сети и/или по логину.

**Вариант A: Приоритетные домены** (без авторизации, просто boost)
В `settings.yml` SearXNG:
```yaml
search:
  preferred_domains:
    - corp-wiki.company.ru
    - portal.company.ru
  preferred_domain_boost: 10
```

**Вариант B: Кастомный JSON-движок с куками**
```yaml
engines:
  - name: corp_search
    engine: json_engine
    search_url: https://corp-portal/api/search?q={query}
    url: https://corp-portal{path}
    weight: 100
    headers:
      Cookie: "session=ВАША_SESSION"
      Authorization: "Bearer ВАШ_ТОКЕН"
```

**Вариант C: Авторизация через Trafilatura** (SearXNG ищет ссылки, Trafilatura дёргает с сессией)
```python
import requests, trafilatura
session = requests.Session()
# авторизация
session.post("https://corp-portal/login", data={"login": "user", "password": "pass"})
# дёрнуть каждый результат с сессией
for url in searxng_results:
    html = session.get(url).text
    text = trafilatura.extract(html)
```

**Вариант D: SearXNG через корпоративный прокси**
```yaml
engines:
  - name: corp_wiki
    engine: http
    proxies: http://corp-proxy:3128
    search_url: https://internal-wiki/search?q={query}
```

SearXNG сам пароли не хранит — авторизация на уровне HTTP-запроса к движку (куки, токен, basic auth).

## Ссылки

- ZVec: https://github.com/alibaba/zvec
- SearXNG: https://docs.searxng.org
- Context7: https://context7.com
- LM Studio: https://lmstudio.ai
