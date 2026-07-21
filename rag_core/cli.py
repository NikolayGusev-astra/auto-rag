"""Console entry point for auto-rag (unified packaging).

Thin wrapper so `pip install -e .` exposes a `auto-rag` command without
duplicating the existing interactive search logic in rag_search.py.
"""
from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        from rag_core.gateway.doctor import main as doctor_main

        sys.exit(doctor_main(sys.argv[2:]))
    from rag_core.rag_search import main as search_main
    sys.exit(search_main())


if __name__ == "__main__":
    main()
