# Auto-RAG Gateway — Architecture

> **Дата:** 20 июля 2026  
> **ADR:** [ADR-004](adr/004-local-workstation-focus.md) — локальный workstation RAG для одного инженера  
> **Тесты:** 368 passed, 5 skipped, 1 xfailed

## Обзор

Auto-RAG Gateway — локальный MCP stdio-сервер, агрегирующий поиск по Confluence, Jira, Automation Hub и локальному снапшоту. Работает офлайн (local snapshot + ZVec), онлайн с реранкером BGE-M3 через LM Studio или с CPU-fallback.

```
Hermes Agent → MCP ClientSession (stdio)
  → auto-rag-gateway/server.py (FastMCP)
    → DcdPlanner → RetrievalCoordinator
      ├─ JiraConnector       (REST API, Bearer)
      ├─ ConfluenceConnector  (REST API, page ID + title + text)
      ├─ HubConnector         (Galaxy v3, published + validated)
      ├─ LocalSnapshotConnector (hybrid lexical+vector)
      ├─ ZVecHttpConnector    (FastAPI :8678)
      └─ GenericMcpConnector  (любой Hermes MCP tool)
    → fuse() → RerankAdapter (BGE-M3 или CPU) → Evidence[]
    → MemvidEnricher → episode → Response
```

## Слои

### 1. MCP Gateway (`rag_core/gateway/`)

**Вход:** `Hermes Agent` → `server.py` (FastMCP, `--config gateway.toml`)

**TOML-конфиг:** `~/.config/auto-rag/gateway.toml`

**Коннекторы:**

| Коннектор | kind | Источник | Аутентификация |
|-----------|------|----------|---------------|
| `JiraConnector` | `jira` | REST API, `text~` JQL + `issueKey=` | Bearer (`env:JIRA_PAT`) |
| `ConfluenceConnector` | `confluence` | REST API, `id=`, `title~`, `text~` CQL | Bearer (`env:CONFLUENCE_PAT`) |
| `HubConnector` | `hub` | Galaxy v3, `published` + `validated`, client-side matching | Token (`env:HUB_TOKEN`) |
| `LocalSnapshotConnector` | (авто) | hybrid lexical+vector UNION из `docs.jsonl` + `lexical.json` | — |
| `ZVecHttpConnector` | `zvec` | FastAPI `:8678/search`, BGE-M3 векторный поиск | — |
| `GenericMcpConnector` | `mcp-proxy` | любой Hermes MCP-инструмент | — |

**RetrievalCoordinator:** `health_map()` → `kind_availability` → `search()` (parallel) → `fuse()` → `RerankAdapter`

**DcdPlanner:** читает `routing.json` (SourceDiscovery), выбирает домены/источники. DcdLearner обновляет аффинити keyword→source из эпизодов.

**MemvidEnricher:** сохраняет эпизод после каждого поиска: `{query, route, document_ids, reranker_score, index_revision}` в `episodes.jsonl`.

### 2. Model Runtime (`rag_core/gateway/model_runtime/`)

**RobustEmbeddingProvider** (`providers/robust.py`) — fallback-цепочка:

```
LM Studio (BGE-M3, localhost:1234)
  ↓ недоступен
CPU sentence-transformers (multilingual-e5-large)
  ↓ недоступен
None → graceful degradation (без реранкера)
```

**RerankAdapter:** cosine-реранкинг через эмбеддинги. `reranker_score` → `final_score = 0.4 × retrieval + 0.6 × reranker`. Graceful fallback: при недоступности возвращает retrieval order.

**Env:** `EMBED_URL`, `EMBED_MODEL`, `CPU_EMBED_MODEL`

### 3. Index & Eval

| Компонент | Файл | Назначение |
|-----------|------|-----------|
| **ZVec Server** | `rag_core/zvec_server.py` | FastAPI :8678, загружает коллекцию один раз, no file lock |
| **Chroma** | `rag_core/chroma_adapter.py` | Альтернатива без AVX2 |
| **eval_golden** | `rag_core/eval_golden.py` | Два слоя: детерминированные метрики + опциональный Qwen-2.5 judge |
| **DCD Learner** | `rag_core/gateway/adaptive/dcd_learner.py` | keyword→source аффинити из эпизодов |
| **SourceDiscovery** | `rag_core/gateway/adaptive/source_discovery.py` | wiki frontmatter → Confluence → routing.json |

### 4. Поток запроса

```
1. MCP Client → server.search("INT-6515", top_k=5)
2. DcdPlanner: routing.json → domains, sources
3. RetrievalCoordinator.health_map() → доступные коннекторы
4. Параллельный search → Evidence[] (retrieval_score)
5. fuse() → объединение результатов
6. RerankAdapter → BGE-M3 cosine reranking (reranker_score)
7. final_score = 0.4 × retrieval + 0.6 × reranker
8. MemvidEnricher → episode → episodes.jsonl
9. Response → {results, trace, runtime}
```

## Новые MCP-источники

Добавить в `gateway.toml`:

```toml
[sources.bitbucket]
kind = "mcp-proxy"
enabled = true
extra = { tool = "bitbucket_search_code", server = "bitbucket" }
```

`GenericMcpConnector` обернёт любой Hermes MCP-инструмент. Сессия инжектится через `register_mcp_session_factory()` в рантайме.

## Тесты

```
368 passed, 5 skipped, 1 xfailed
```

- `tests/gateway/test_phase7_*` — gateway bootstrap
- `tests/gateway/test_jira_connector.py` — Jira REST
- `tests/gateway/test_confluence_connector.py` — Confluence REST
- `tests/gateway/test_hub_connector.py` — Hub Galaxy v3
- `tests/gateway/test_zvec_http.py` — ZVec HTTP
- `tests/gateway/test_rerank_adapter.py` — BGE-M3 reranker
- `tests/gateway/test_dcd_learner.py` — DCD Learner
- `tests/gateway/test_eval_golden.py` — двухслойный eval
- `tests/gateway/test_source_discovery.py` — SourceDiscovery
