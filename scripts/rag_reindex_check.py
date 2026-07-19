#!/usr/bin/env python3
"""Cron wrapper: check ZVec freshness, output only if stale."""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_reindex import check_freshness

report = check_freshness()
if report["reindex_needed"]:
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0)
# Silent exit — nothing to report