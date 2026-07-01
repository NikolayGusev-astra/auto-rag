#!/usr/bin/env bash
# Start SearXNG locally
set -e
docker run -d --name searxng \
  -p 127.0.0.1:8080:8080 \
  -v searxng_data:/etc/searxng \
  -e BASE_URL=http://localhost:8080 \
  searxng/searxng
echo "SearXNG started on http://localhost:8080"