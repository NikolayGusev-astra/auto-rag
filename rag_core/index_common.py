"""Общие хелперы индексаторов (deslop: вынесен дублирующий код
из indexer.py и zvec_incremental_indexer.py).

Только чистые функции без состояния и без привязки к ОС/путям,
чтобы оба индексера (Linux/Autolycus и Windows) могли их переиспользовать.
"""

import hashlib
import logging
import re

logger = logging.getLogger(__name__)


def file_hash(path: str) -> str:
    """MD5 файла потоково (идентичен в обоих индексерах)."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Разбор YAML frontmatter. Битый YAML -> пустые метаданные + лог."""
    text = text.lstrip('\ufeff')
    if not text.startswith('---'):
        return {}, text
    end = text.find('---', 3)
    if end == -1:
        return {}, text
    meta = {}
    try:
        import yaml
        meta = yaml.safe_load(text[3:end].strip()) or {}
    except Exception as e:
        logger.warning("parse_frontmatter: битый YAML, пустые метаданные: %s", e)
    return (meta if isinstance(meta, dict) else {}), text[end + 3:].strip()


def _safe_id(source: str, content: str) -> str:
    """Zvec-safe doc ID: max 64 chars, only alphanumeric and underscore."""
    raw = f"{source}#{hashlib.md5(content.encode()).hexdigest()[:12]}"
    safe = re.sub(r'[^a-zA-Z0-9]', '_', raw)
    if safe[0] == '_':
        safe = 'doc' + safe
    if len(safe) > 64:
        suffix = hashlib.md5(safe.encode()).hexdigest()[:12]
        safe = safe[:51] + suffix
    return safe[:64]
