# Auto-RAG — перенос на рабочий ноутбук

## 1. Клонировать репо
```bash
git clone https://github.com/NikolayGusev-astra/auto-rag.git ~/projects/auto-rag
cd ~/projects/auto-rag
pip install -e ".[all]"
```

## 2. Скопировать конфиги
```bash
# gateway.toml — поправить пути (knowledge_root, doc_root page IDs)
cp ~/.config/auto-rag/gateway.toml ~/.config/auto-rag/

# wiki
cp -r ~/wiki/rusbitech ~/wiki/

# golden set
cp ~/wiki/eval/golden_set.jsonl ~/wiki/eval/
```

## 3. Настроить env-переменные (через Hermes MCP)
```
EMBED_URL=http://localhost:1234/v1/embeddings   # или cloud URL
EMBED_MODEL=text-embedding-baai-bge-m3-568m
CPU_EMBED_MODEL=intfloat/multilingual-e5-large   # fallback при отсутствии LM Studio
JIRA_PAT=...
CONFLUENCE_PAT=...
HUB_TOKEN=...
```

## 4. LM Studio или CPU fallback
- Если LM Studio запущен → BGE-M3 через `localhost:1234`
- Если LM Studio НЕ запущен → CPU fallback (`pip install sentence-transformers`)
- Если нет ни того, ни другого → graceful degradation (без реранкера)

## 5. ZVec сервер
```bash
python -m rag_core.zvec_server --port 8678 &
# или через systemd/автозапуск
```

## 6. Зарегистрировать в Hermes
```bash
hermes mcp add auto-rag \
  --command ~/projects/auto-rag/.venv/Scripts/python.exe \
  --env EMBED_URL=http://localhost:1234/v1/embeddings \
  --env EMBED_MODEL=text-embedding-baai-bge-m3-568m \
  --env CPU_EMBED_MODEL=intfloat/multilingual-e5-large \
  --env JIRA_PAT=... \
  --env CONFLUENCE_PAT=... \
  --env HUB_TOKEN=... \
  --args -m rag_core.gateway.server --config ~/.config/auto-rag/gateway.toml
```

## 7. Проверить
```bash
hermes mcp test auto-rag   # должно показать ✓ Connected
python -m rag_core.eval_golden   # прогон метрик
```

## 8. Новые MCP-источники
Добавить в `gateway.toml`:
```toml
[sources.bitbucket]
kind = "mcp-proxy"
enabled = true
extra = { tool = "bitbucket_search_code", server = "bitbucket" }
```
Коннектор сам обернёт любой Hermes MCP-инструмент в поисковый коннектор.
