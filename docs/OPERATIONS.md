# Auto-RAG Gateway — Operations Guide

> **Версия:** Phase 7+ (20 июля 2026)  
> **ADR:** [ADR-004](adr/004-local-workstation-focus.md)  
> **Тесты:** 368 passed, 5 skipped, 1 xfailed

## Быстрый старт

```bash
# Установка
git clone https://github.com/NikolayGusev-astra/auto-rag.git
cd auto-rag
pip install -e ".[gateway]"

# Конфиг
cp docs/gateway.example.toml ~/.config/auto-rag/gateway.toml
# → заполнить credential_ref и extra.base_url

# Снапшот (wiki в локальный индекс)
python -m rag_core.gateway sync --source local_snapshot

# ZVec сервер
python -m rag_core.zvec_server --port 8678 &

# Запуск
python -m rag_core.gateway.server --config ~/.config/auto-rag/gateway.toml
```

## Hermes MCP регистрация

```bash
hermes mcp add auto-rag \
  --command ~/.venv/Scripts/python.exe \
  --env JIRA_PAT=... \
  --env CONFLUENCE_PAT=... \
  --env JIRA_BASE_URL=https://jira.astralinux.ru \
  --env CONFLUENCE_BASE_URL=https://wiki.astralinux.ru \
  --env HUB_TOKEN=... \
  --env HUB_BASE_URL=https://hub.astra-automation.ru \
  --env EMBED_URL=http://localhost:1234/v1/embeddings \
  --env EMBED_MODEL=text-embedding-baai-bge-m3-568m \
  --env CPU_EMBED_MODEL=intfloat/multilingual-e5-large \
  --env NO_PROXY=* \
  --args -m rag_core.gateway.server --config ~/.config/auto-rag/gateway.toml
```

**Проверка:** `hermes mcp test auto-rag` → `✓ Connected`

## Конфигурация

### gateway.toml

```toml
knowledge_root = "~/.local/share/auto-rag"
local_snapshot = true
web = false
adaptive = true       # DCD Planner + DCD Learner

[sources.jira]
kind = "jira"
enabled = true
credential_ref = "env:JIRA_PAT"

[sources.confluence]
kind = "confluence"
enabled = true
credential_ref = "env:CONFLUENCE_PAT"

[sources.hub]
kind = "hub"
enabled = true
credential_ref = "env:HUB_TOKEN"

[sources.zvec]
kind = "zvec"
enabled = true
# extra.url defaults to http://127.0.0.1:8678

[sources.bitbucket]       # пример: новый MCP-источник
kind = "mcp-proxy"
enabled = true
extra = { tool = "bitbucket_search_code", server = "bitbucket" }
```

### Коннекторы

| kind | Коннектор | Требования |
|------|-----------|-----------|
| `jira` | JiraConnector | `env:JIRA_PAT`, `env:JIRA_BASE_URL` |
| `confluence` | ConfluenceConnector | `env:CONFLUENCE_PAT`, `env:CONFLUENCE_BASE_URL` |
| `hub` | HubConnector | `env:HUB_TOKEN`, `env:HUB_BASE_URL` |
| `zvec` | ZVecHttpConnector | ZVec сервер на `:8678` |
| `mcp-proxy` | GenericMcpConnector | любой Hermes MCP tool |

`LocalSnapshotConnector` регистрируется автоматически при `local_snapshot=true`.

## Model Runtime

### Embedding fallback chain

```
LM Studio (BGE-M3, localhost:1234/v1/embeddings)
  ↓ недоступен
CPU sentence-transformers (multilingual-e5-large)
  ↓ недоступен
None → graceful degradation (реранкер недоступен, retrieval order)
```

**Env:** `EMBED_URL` (default: `http://localhost:1234/v1/embeddings`), `EMBED_MODEL` (default: `bge-m3`), `CPU_EMBED_MODEL` (default: `intfloat/multilingual-e5-large`).

**CPU-fallback:** требует `pip install sentence-transformers`. При первом запуске скачает модель (~1.5 GB).

### RerankAdapter

Cosine-реранкинг через эмбеддинги. `final_score = 0.4 × retrieval_score + 0.6 × reranker_score`. Graceful fallback при недоступности.

## ZVec сервер

Загружает коллекцию один раз при старте, держит в памяти — нет file-lock при конкурентных запросах.

```bash
python -m rag_core.zvec_server --port 8678 --host 127.0.0.1
```

Endpoint: `GET /search?q=...&topk=5`  
Health: `GET /health`

**Без AVX2:** Chroma-адаптер (`rag_core/chroma_adapter.py`).

## DCD Routing

### SourceDiscovery

Автоматически находит документацию продуктов через wiki frontmatter + Confluence API:

```bash
python -m rag_core.gateway discover
```

Wiki → Confluence child pages → `~/.config/auto-rag/routing.json`.  
DcdPlanner читает `routing.json`, выбирает домены/источники по ключевым словам.

### DCD Learner

Обучается из эпизодов (keyword→source аффинити):

```bash
python -m rag_core.gateway dcd-learn
```

Не исключает источники — только бустит приоритет.

## Memvid Enrichment

Каждый `search` сохраняет эпизод в `~/.local/share/auto-rag/episodes.jsonl`:

```json
{"query": "INT-6515", "route": ["confluence","jira"], "document_ids": [...], "reranker_score": 0.67}
```

## Eval Golden

Два слоя оценки:

```bash
# Только детерминированные метрики (precision, recall, MRR, nDCG, citation, latency)
python -m rag_core.eval_golden

# + Qwen-2.5 judge (relevance, coverage, groundedness, conflicts)
python -m rag_core.eval_golden --judge
```

Golden set: `~/wiki/eval/golden_set.jsonl`  
Release gate: детерминированные метрики не ухудшились AND Qwen judge без существенной деградации.

## Новые MCP-источники

Добавить секцию в `gateway.toml`:

```toml
[sources.имя]
kind = "mcp-proxy"
enabled = true
extra = { tool = "имя_mcp_инструмента", server = "имя_mcp_сервера" }
```

Без изменения кода. `GenericMcpConnector` оборачивает любой Hermes MCP tool.

## Портинг на другую машину

См. [porting-checklist.md](../../porting-checklist.md).

Кратко:
1. `git clone` + `pip install -e ".[gateway]"`
2. Скопировать `~/.config/auto-rag/gateway.toml` (поправить пути)
3. Скопировать `~/wiki/rusbitech/`, `~/wiki/eval/`
4. Настроить env-переменные (JIRA_PAT, CONFLUENCE_PAT, HUB_TOKEN, EMBED_URL, EMBED_MODEL)
5. Запустить ZVec-сервер: `python -m rag_core.zvec_server &`
6. Зарегистрировать в Hermes: `hermes mcp add auto-rag ...`
7. Проверить: `hermes mcp test auto-rag` → `✓ Connected`
8. Прогнать тесты: `python -m pytest tests -q`

## Troubleshooting

| Симптом | Проверить |
|---------|----------|
| `ConnectorStub` в результатах | `credential_ref` разрешается? env-переменные установлены? |
| Пустой local_snapshot | `knowledge_root` существует? снапшот опубликован через `sync`? |
| ZVec 503 | Сервер запущен на `:8678`? `/health` возвращает 200? |
| LM Studio недоступен | `EMBED_URL` корректен? `NO_PROXY=*` установлен? CPU-fallback: `pip install sentence-transformers` |
| Реракер не работает | LM Studio ИЛИ CPU fallback должны быть доступны. Graceful degradation — результаты в retrieval order. |
| Heremes не видит auto-rag | `hermes mcp list` → enabled? Проверить `--env` ДО `--args` в команде регистрации. |
| 0 результатов Hub | Hub отдаёт published (11 коллекций) + validated (40 коллекций). `astra.acm` и `astra.aa_controller` в validated. |
