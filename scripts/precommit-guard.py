#!/usr/bin/env python3
"""Pre-commit guard: reject artifacts that should be gitignored.

Usage: python scripts/precommit-guard.py [--fix]

Without --fix: exits non-zero if forbidden files are tracked.
With --fix: adds entries to .gitignore and removes tracked artifacts.
"""

import subprocess
import sys
from pathlib import Path

FORBIDDEN = [
    ".pytest-tmp-*",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".mypy_cache/",
    ".ruff_cache/",
    "*.egg-info/",
    "dist/",
]

GITIGNORE = Path(__file__).resolve().parent.parent / ".gitignore"


def check() -> int:
    tracked = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True, text=True,
    ).stdout.splitlines()

    violations = []
    for pattern in FORBIDDEN:
        if pattern.endswith("/"):
            prefix = pattern[:-1]
            for f in tracked:
                if f == prefix or f.startswith(prefix + "/") or prefix in f.split("/"):
                    violations.append(f)
        elif "*" in pattern:
            import fnmatch
            for f in tracked:
                if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(Path(f).name, pattern):
                    violations.append(f)

    if violations:
        print("❌ Forbidden artifacts found in tracked files:")
        for v in sorted(set(violations)):
            print(f"   {v}")
        print(f"\n   Run `python scripts/precommit-guard.py --fix` to auto-clean.")
        return 1

    print("✅ No forbidden artifacts in tracked files.")
    return 0


def fix() -> int:
    # 1. Ensure .gitignore has entries
    existing = GITIGNORE.read_text().splitlines() if GITIGNORE.exists() else []
    added = []
    for pattern in FORBIDDEN:
        if pattern not in existing:
            with open(GITIGNORE, "a") as f:
                f.write(f"\n{pattern}")
            added.append(pattern)

    if added:
        print(f"➕ Added to .gitignore: {', '.join(added)}")

    # 2. Remove tracked artifacts
    for pattern in FORBIDDEN:
        subprocess.run(
            ["git", "rm", "-r", "--cached", "--ignore-unmatch", pattern],
            capture_output=True,
        )

    print("✅ Cleaned tracked artifacts and updated .gitignore.")
    return 0


if __name__ == "__main__":
    if "--fix" in sys.argv:
        sys.exit(fix())
    else:
        sys.exit(check())
