# ADR-008: Offline Sync for Live Corporate Connectors

**Status:** Proposed
**Date:** 2026-07-22
**Extends:** ADR-001 (knowledge gateway), ADR-004 (offline-capable), ADR-006 (sync engine)

## 1. Context

The gateway claims offline capability: `local_snapshot` provides retrieval when Jira/Confluence/Lodestone are unreachable.

Currently, `SyncEngine.sync_source(connector)` calls `connector.sync_changes(cursor)`, but all live connectors return empty batches:

| Connector | sync_changes | fetch |
|-----------|-------------|-------|
| JiraConnector | `SyncBatch(added=[])` | `NotImplementedError` |
| ConfluenceConnector | `SyncBatch(added=[])` | `NotImplementedError` |
| LodestoneConnector | `SyncBatch(added=[])` | `NotImplementedError` |
| AllowlistedWebConnector | `SyncBatch(added=[])` | `NotImplementedError` |

The `sync` MCP tool triggers `engine.sync_source(connector)`, but with empty batches, the snapshot only contains what was placed there by an external pipeline.

**Audit finding:** offline mode is not equivalent to live retrieval for corporate sources.

## 2. Problem

```text
User: "Прошу разрешить пилот на 10 инженеров"
Reviewer: "Как Jira comments попадают в snapshot?"
Answer: ↓↓↓ (empty)
```

Without `sync_changes` implementations, the claim "можно заранее проиндексировать всё и работать вне сети" is false for the most important corporate sources.

## 3. Decision

**Implement incremental `sync_changes` for JiraConnector first, ConfluenceConnector second.**

Scope per connector:

### JiraConnector.sync_changes(cursor)
- Query Jira search API with `updated >= cursor`
- Return `SyncBatch(added=[...], changed=[...], deleted=[...], cursor=next_cursor)`
- Each `Document` contains full issue body (summary + description + comments)
- Cursor is the latest `updated` timestamp from processed issues

### ConfluenceConnector.sync_changes(cursor)
- Query Confluence CQL with `lastModified >= cursor`
- Return `SyncBatch` with page content + PDF attachment text
- Same cursor semantics

**Lodestone** and **Allowlisted Web** are excluded from Phase 1: Lodestone is a passthrough search proxy (no notion of incremental sync), and Allowlisted Web is inherently live-only (public web changes continuously).

## 4. Consequences

- **Positive:** offline snapshot becomes a truthful replica of Jira/Confluence content. `sync` MCP tool produces real documents.
- **Positive:** the gateway's offline claim becomes verifiable: `sync → disconnect → search` returns the same results as online.
- **Negative:** sync is per-connector, per-organization schema. Hardcoded Jira/Confluence APIs are a coupling cost.
- **Risk:** incremental cursor semantics depend on Jira/Confluence clock consistency. Mitigation: cursor is best-effort; full rebuild is a separate `sync --full` mode.

## 5. Verification

Integration test:
```text
Fake Jira HTTP server with 3 issues + updated timestamps
→ sync_changes(None) returns all 3
→ sync_changes(cursor) returns 1 (the newer one)
→ engine.sync_source stores them in snapshot
→ offline search retrieves them
```
