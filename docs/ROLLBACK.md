# Rollback Procedure

Auto-RAG does not have a package registry or release server. Rollback is
performed at the Git level.

## Current version rollback

```bash
cd auto-rag
git log --oneline -5            # identify the working commit
git checkout <known-good-sha>   # roll back code
pip install -e ".[gateway,pdf]" # reinstall
pytest -q                       # verify
```

## Per-release rollback (when versioned wheels exist)

```bash
pip install auto-rag==<known-good-version>
hermes mcp restart auto-rag
pytest -q
```

## Snapshot/state rollback

The local snapshot (`knowledge_root`) is **not** versioned with the code.
It lives in `~/.local/share/auto-rag` (or configured `knowledge_root`).

- **Rollback snapshot:** delete `knowledge_root/` directory and re-run `sync`.
- **Rollback index only:** `python -m rag_core.gateway sync --source local_snapshot --full`

## Configuration rollback

`gateway.toml` is a local file. Keep a copy before changes:

```bash
cp ~/.config/auto-rag/gateway.toml ~/.config/auto-rag/gateway.toml.bak
```

## Current limitations

- No automated rollback (manual Git checkout)
- No versioned releases on PyPI/GitHub Releases
- Snapshot is not atomically versioned per sync
- Configuration changes are not logged

These are addressed in ADR-007/008 (packaging and managed delivery) and are
out of scope for the 10-user pilot.
