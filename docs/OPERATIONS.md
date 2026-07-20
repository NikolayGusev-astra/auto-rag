# Auto-RAG Gateway — Operations Guide

> **Версия:** Phase 7 (20 июля 2026)  
> **ADR:** [ADR-004](ADR-004-local-workstation-rag.md)  
> **Тесты:** 368 passed, 5 skipped, 1 xfailed (commit `92a78d7`)

## Быстрый старт — минимальный offline

```bash
git clone https://github.com/NikolayGusev-astra/auto-rag.git
cd auto-rag
pip install -e ".[gateway]"
```

```toml
# ~/.config/auto-rag/gateway.toml
knowledge_root = "~/.local/share/auto-rag"
local_snapshot = true
web = false
adaptive = false
```

```bash
# Снапшот (wiki в локальный индекс)
python -m rag_core.gateway sync --source local_snapshot

# Запуск
python -m rag_core.gateway.server --config ~/.config/auto-rag/gateway.toml

# Hermes
hermes mcp add auto-rag \
  --command ~/.venv/Scripts/python.exe \
  --args -m rag_core.gateway.server --config ~/.config/auto-rag/gateway.toml

hermes mcp test auto-rag   # ✓ Connected
```

## Опциональные источники (CURRENT)

### Jira + Confluence + Hub

```toml
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
```

```bash
hermes mcp add auto-rag \
  --env JIRA_PAT=... \
  --env CONFLUENCE_PAT=... \
  --env JIRA_BASE_URL=https://jira.astralinux.ru \
  --env CONFLUENCE_BASE_URL=https://wiki.astralinux.ru \
  --env HUB_TOKEN=... \
  --env HUB_BASE_URL=https://hub.astra-automation.ru \
  --env NO_PROXY=* \
  ...
```

## Опциональные возможности (AVAILABLE)

### Реракер (AVAILABLE — требует LM Studio или CPU)

**LM Studio (BGE-M3):**
```bash
--env EMBED_URL=http://localhost:1234/v1/embeddings
--env EMBED_MODEL=text-embedding-baai-bge-m3-568m
```

**CPU fallback (без LM Studio):**
```bash
pip install sentence-transformers   # ~1.5 GB скачает при первом запуске
--env EMBED_MODEL=bge-m3
--env CPU_EMBED_MODEL=intfloat/multilingual-e5-large
```

**Без реранкера:** gateway работает, возвращает retrieval order без ошибок.

**Цепочка:** LM Studio → CPU sentence-transformers → graceful degradation.

### ZVec сервер (AVAILABLE)

```bash
python -m rag_core.zvec_server --port 8678 &

# gateway.toml
[sources.zvec]
kind = "zvec"
enabled = true
```

**Без AVX2:** `rag_core/chroma_adapter.py`

### DCD Routing (AVAILABLE)

```bash
# Обнаружение документации продуктов
python -m rag_core.gateway discover

# Обучение keyword→source аффинити из эпизодов
python -m rag_core.gateway dcd-learn
```

Включить adaptive planner: `adaptive = true` в gateway.toml.

### Eval Golden (AVAILABLE)

```bash
python -m rag_core.eval_golden           # детерминированные метрики
python -m rag_core.eval_golden --judge   # + Qwen-2.5 judge
```

Golden set: `~/wiki/eval/golden_set.jsonl`

### Новые MCP-источники (AVAILABLE — требует GenericMcpConnector)

```toml
[sources.bitbucket]
kind = "mcp-proxy"
enabled = true
extra = { tool = "bitbucket_search_code", server = "bitbucket" }
```

## Коннекторы

| kind | Статус | Коннектор | Требования |
|------|--------|-----------|-----------|
| `jira` | CURRENT | JiraConnector | `env:JIRA_PAT`, `env:JIRA_BASE_URL` |
| `confluence` | CURRENT | ConfluenceConnector | `env:CONFLUENCE_PAT`, `env:CONFLUENCE_BASE_URL` |
| `hub` | CURRENT | HubConnector | `env:HUB_TOKEN`, `env:HUB_BASE_URL` |
| `zvec` | AVAILABLE | ZVecHttpConnector | ZVec сервер на `:8678` |
| `mcp-proxy` | AVAILABLE | GenericMcpConnector | session factory инжектится в рантайме |

`LocalSnapshotConnector` регистрируется автоматически при `local_snapshot=true`.

## Портинг на другую машину

1. `git clone` + `pip install -e ".[gateway]"`
2. Скопировать `~/.config/auto-rag/gateway.toml` (поправить пути)
3. Скопировать `~/wiki/rusbitech/`, `~/wiki/eval/`
4. Настроить env-переменные в Hermes MCP регистрации
5. Запустить ZVec-сервер: `python -m rag_core.zvec_server &` (опционально)
6. `hermes mcp add auto-rag ...`
7. Проверить: `hermes mcp test auto-rag` → `✓ Connected`
8. Прогнать тесты: `python -m pytest tests -q`

## Troubleshooting

| Симптом | Проверить |
|---------|----------|
| `ConnectorStub` в результатах | `credential_ref` разрешается? env-переменные установлены? |
| Пустой local_snapshot | `knowledge_root` существует? снапшот опубликован через `sync`? |
| ZVec 503 | Сервер запущен на `:8678`? `/health` возвращает 200? |
| LM Studio недоступен | `EMBED_URL` корректен? `NO_PROXY=*` установлен? CPU-fallback: `pip install sentence-transformers` |
| Реракер не работает | LM Studio ИЛИ CPU fallback должны быть доступны. Graceful — retrieval order. |
| Hermes не видит auto-rag | `hermes mcp list` → enabled? `--env` ДО `--args` в команде. |
| 0 результатов Hub | `published` (11 коллекций) + `validated` (40). `astra.acm`, `astra.aa_controller` — в validated. |
