# Agent Gateway MCP schemas

Transport for every tool is **MCP stdio**. Requests and responses below are
JSON objects. Errors use the same diagnostic form and never include credentials:

```json
{"tool_error": {"code": "source_unavailable", "message": "Jira is unavailable"}}
```

## `search`

Request:

```json
{"query": "cluster update", "topk": 5, "domain": "engineering", "collection": "wiki", "include_web": false, "continuation_token": null}
```

Response:

```json
{
  "evidence": [{
    "id": "confluence:12345#chunk-4", "document_id": "confluence:12345",
    "title": "Cluster update", "text": "...", "source": "confluence",
    "uri": "https://wiki.example/pages/12345", "origin": "local_snapshot",
    "retrieval_score": 0.81, "reranker_score": 0.92,
    "updated_at": "2026-07-19T08:00:00Z", "synced_at": "2026-07-19T08:05:00Z",
    "metadata": {}
  }],
  "continuation_token": null,
  "runtime": {
    "retrieval": "hybrid", "embedding_provider": "sentence-transformers",
    "reranker": "disabled", "language_model": "none", "execution": "cpu"
  }
}
```

The `runtime` block is required and reports the technical retrieval mode used
for this response.

## `fetch`

Request:

```json
{"document_id": "confluence:12345", "chunk_id": "chunk-4"}
```

Response: one `Evidence` object with the shape shown in `search`.

## `sync`

Request:

```json
{"source": "confluence", "cursor": "previous-cursor"}
```

Response (`SyncBatch`):

```json
{
  "added": [{"id": "confluence:12345", "source": "confluence", "source_instance": "wiki-prod", "title": "Cluster update", "text": "...", "uri": null, "version": "v3", "updated_at": null, "content_hash": "abc123", "metadata": {}}],
  "changed": [], "deleted": ["confluence:12200"], "cursor": "next-cursor",
  "warnings": [], "stats": {"added": 1, "changed": 0, "deleted": 1}
}
```

## `sync_status`

Request:

```json
{"source": "confluence"}
```

Response:

```json
{"source": "confluence", "cursor": "next-cursor", "in_progress": false, "last_synced_at": "2026-07-19T08:05:00Z", "warnings": [], "stats": {"added": 1}}
```

## `list_sources`

Request:

```json
{}
```

Response:

```json
{"sources": [{"source": "confluence", "available": true, "detail": "ok"}]}
```

## `source_status`

Request:

```json
{"source": "confluence"}
```

Response (`SourceHealth`):

```json
{"source": "confluence", "available": true, "detail": "ok"}
```
