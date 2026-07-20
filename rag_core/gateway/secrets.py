"""Credential reference resolution; configuration never carries secret values."""
from __future__ import annotations

import os


def resolve_credential(ref: str | None) -> str | None:
    if ref is None:
        return None
    if ref.startswith("env:"):
        return os.environ[ref.removeprefix("env:")]
    if ref.startswith("keyring:"):
        return None
    raise ValueError("credential references must use env: or keyring:")
