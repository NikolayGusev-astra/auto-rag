#!/usr/bin/env bash
set -euo pipefail
pip install "memvid-sdk>=2.0" || {
  echo "WARN: memvid-sdk install failed; memory layer will run in noop mode (RAG unaffected)."
  exit 0
}
python3 -c "import memvid" && echo "memvid-sdk OK" || echo "WARN: import memvid failed"
