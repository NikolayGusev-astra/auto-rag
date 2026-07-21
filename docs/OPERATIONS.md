# Auto-RAG Gateway — Operations Guide

> **Версия:** ADR-006 Stabilization (21 июля 2026)
> **ADR:** [ADR-006](ADR-006-stabilization-before-expansion.md)
> **Тесты:** 437 passed, 5 skipped, 1 xfailed (commit `d93cfab`)

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
adaptive = true
```

```bash
# Снапшот
python -m rag_core.gateway sync --source local_snapshot

# Запуск
python -m rag_core.gateway.server --config ~/.config/auto-rag/gateway.toml

# Hermes
hermes mcp add auto-rag \
  --command ~/.venv/Scripts/python.exe \
  --args -m rag_core.gateway.server --config ~/.config/auto-rag/gateway.toml

hermes mcp test auto-rag   # ✓ Connected
```

## Источники (все IMPLEMENTED)

| kind | Коннектор | Особенности |
|------|----------|-------------|
| `jira` | JiraConnector | Exact-key → paginated comments (≤500) + linked issues (≤5) + enrichment diagnostics |
| `confluence` | ConfluenceConnector | Empty-body pages → PDF attachment extraction (pymupdf/pdfplumber). `content_status` metadata |
| `lodestone` | LodestoneConnector | Corporate KB: wiki.astralinux.ru, aa-docs, aa-confluence. MCP HTTP |
| `allowlisted-web` | AllowlistedWebConnector | SearXNG с domain filter: aldpro.ru, astralinux.ru. Подавлен для SIRIUS-*/INT-* |
| `hub` | HubConnector | `env:HUB_TOKEN`, `env:HUB_BASE_URL` |
| `zvec` | ZVecHttpConnector | ZVec сервер на `:8678` |
| `searxng` | SearXNGConnector | `http://localhost:8888` |
| `web` | WebSearchConnector | **DISABLED** — corporate-first policy |
| `mcp-proxy` | GenericMcpConnector | Session factory инжектится в рантайме |

`LocalSnapshotConnector` регистрируется автоматически при `local_snapshot=true`.

Источники приоритета: Jira → Confluence → Lodestone → Allowlisted Web → Hub → ZVec → SearXNG. Web off.

### Конфигурация

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

[sources.lodestone]
kind = "lodestone"
enabled = true

[sources.allowlisted_web]
kind = "allowlisted-web"
enabled = true
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

## Возможности

### Реракер (BGE-M3)

**LM Studio:**
```bash
--env EMBED_URL=http://localhost:1234/v1/embeddings
--env EMBED_MODEL=text-embedding-baai-bge-m3-568m
```

**CPU fallback:**
```bash
pip install sentence-transformers   # ~1.5 GB при первом запуске
--env EMBED_MODEL=bge-m3
--env CPU_EMBED_MODEL=intfloat/multilingual-e5-large
```

Цепочка: LM Studio → CPU sentence-transformers → graceful degradation.

### auto-rag doctor

```bash
python -m rag_core.gateway.doctor          # human-readable
python -m rag_core.gateway.doctor --json   # machine-readable
```

Exit codes: 0=ready, 1=config error, 2=snapshot unavailable, 3=degraded.

### Eval Golden

```bash
python -m rag_core.eval_golden           # детерминированные метрики
python -m rag_core.eval_golden --judge   # + Qwen-2.5 judge
```

### DCD Routing

```bash
python -m rag_core.gateway discover
python -m rag_core.gateway dcd-learn   # ручное обучение
```

### Pre-commit Guard

```bash
python scripts/precommit-guard.py        # проверка артефактов
python scripts/precommit-guard.py --fix  # авто-очистка + .gitignore
```

Запрещены в tracked files: `.pytest-tmp-*`, `__pycache__`, `*.pyc`, `*.pyo`, `.mypy_cache`, `.ruff_cache`, `*.egg-info`, `dist/`.

## Портинг на другую машину

1. `git clone` + `pip install -e ".[gateway]"`
2. Скопировать `~/.config/auto-rag/gateway.toml` (поправить пути)
3. Настроить env-переменные в Hermes MCP регистрации
4. `hermes mcp add auto-rag ...`
5. `hermes mcp test auto-rag` → `✓ Connected`
6. `python -m pytest tests -q` → 437 passed

## Troubleshooting

| Симптом | Проверить |
|---------|----------|
| `ConnectorStub` | `credential_ref` разрешается? env-переменные установлены? |
| Пустой local_snapshot | `knowledge_root` существует? снапшот опубликован через `sync`? |
| Jira без комментариев | Exact key в запросе? `enrichment.comments_status` в metadata |
| Confluence PDF пустой | `content_status` = `no_pdf`/`extraction_failed`? Есть PDF-вложения? |
| Lodestone skipped | Транзиентная деградация. Jira+Confluence должны покрыть. |
| ZVec 503 | Сервер на `:8678`? `/health` возвращает 200? |
| Реракер не работает | LM Studio ИЛИ CPU fallback доступны. Graceful — retrieval order. |
| Hermes не видит auto-rag | `hermes mcp list` → enabled? `--env` ДО `--args`. |
| PYTHONPATH не установлен | MCP env: `PYTHONPATH=C:\Users\n.gusev\projects\auto-rag` |
