# Auto-RAG Gateway — Architecture

> **Дата:** 20 июля 2026  
> **ADR:** [ADR-004](ADR-004-local-workstation-rag.md) — локальный workstation RAG для одного инженера  
> **Тесты:** 368 passed, 5 skipped, 1 xfailed (commit `92a78d7`)

## Обзор

Auto-RAG Gateway — локальный MCP stdio-сервер, агрегирующий поиск по Confluence, Jira, Automation Hub и локальному снапшоту.

Архитектура включает три категории компонентов:

- **CURRENT** — подключено к gateway path, работает на каждый MCP-запрос
- **AVAILABLE** — реализовано, запускается отдельно, не в hot path
- **TARGET** — следующий этап интеграции

## Основной поток (CURRENT)

```
Hermes Agent → MCP ClientSession (stdio)
  → auto-rag-gateway/server.py (FastMCP, --config gateway.toml)
    → DcdPlanner (routing.json) → RetrievalCoordinator
      ├─ JiraConnector       (REST API)         [CURRENT]
      ├─ ConfluenceConnector  (REST API)         [CURRENT]
      ├─ HubConnector         (Galaxy v3 API)    [CURRENT]
      ├─ LocalSnapshotConnector (hybrid lexical) [CURRENT]
      ├─ ZVecHttpConnector    (FastAPI :8678)    [AVAILABLE — требует ZVec server]
      └─ GenericMcpConnector  (любой MCP tool)   [AVAILABLE — требует session factory]
    → fuse() → Evidence[]
    → MemvidEnricher (опционально, adaptive=true) → episode → Response
```

### Коннекторы

| Коннектор | Статус | kind | Источник |
|-----------|--------|------|----------|
| `JiraConnector` | **CURRENT** | `jira` | REST API, `text~` JQL + `issueKey=` |
| `ConfluenceConnector` | **CURRENT** | `confluence` | REST API, `id=`, `title~`, `text~` CQL |
| `HubConnector` | **CURRENT** | `hub` | Galaxy v3, `published` + `validated` |
| `LocalSnapshotConnector` | **CURRENT** | (авто) | hybrid lexical+vector из локального снапшота |
| `ZVecHttpConnector` | **AVAILABLE** | `zvec` | FastAPI `:8678/search` |
| `GenericMcpConnector` | **AVAILABLE** | `mcp-proxy` | любой Hermes MCP tool |

## Model Runtime

### Embedding fallback chain

| Уровень | Статус | Провайдер |
|---------|--------|-----------|
| LM Studio (BGE-M3, `localhost:1234`) | **AVAILABLE** — требует запущенный LM Studio | `providers/openai_compat.py` |
| CPU sentence-transformers (`multilingual-e5-large`) | **AVAILABLE** — требует `pip install sentence-transformers` | `providers/cpu.py` |
| Graceful degradation | **CURRENT** — без реранкера | `rerank_adapter.py` |

### RerankAdapter

**CURRENT.** Cosine-реранкинг через эмбеддинги. При недоступности embedding-провайдера возвращает retrieval order без ошибок. `final_score = 0.4 × retrieval_score + 0.6 × reranker_score`.

**Env:** `EMBED_URL`, `EMBED_MODEL`, `CPU_EMBED_MODEL`

## DCD Routing

| Компонент | Статус | Описание |
|-----------|--------|----------|
| **DcdPlanner** | **CURRENT** | Читает `routing.json`, выбирает домены/источники |
| **SourceDiscovery** | **AVAILABLE** | `python -m rag_core.gateway discover` — wiki → Confluence → `routing.json` |
| **DCD Learner** | **AVAILABLE** | `python -m rag_core.gateway dcd-learn` — keyword→source аффинити из эпизодов |

## Index & Eval

| Компонент | Статус | Файл |
|-----------|--------|------|
| **ZVec Server** | **AVAILABLE** | `rag_core/zvec_server.py` — FastAPI :8678 |
| **Chroma adapter** | **AVAILABLE** | `rag_core/chroma_adapter.py` — без AVX2 |
| **eval_golden** | **AVAILABLE** | Два слоя: детерм. метрики + Qwen judge |

## Memvid Enrichment

**Только при `adaptive=true` в gateway.toml.** Сохраняет эпизод после поиска: `{query, route, document_ids, reranker_score, index_revision}` в `episodes.jsonl`. Не является частью reference retrieval — это опциональный side effect для DCD-обучения и аналитики.

## Milestones

- **Phase 6** (аудит): SyncEngine non-destructive merge, RevisionManifestStore, IndexPipeline
- **Phase 7** (gateway): FastMCP, connector factory, live Jira/Confluence/Hub, RobustEmbeddingProvider
- **Phase 8 (TARGET)**: интеграция ZVec, DCD Learner, eval_golden в hot path
