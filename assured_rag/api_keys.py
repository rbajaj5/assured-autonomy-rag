from __future__ import annotations

import os


def normalize_api_key(value: str | None) -> str:
    """Normalize an API key without logging or exposing it."""
    if value is None:
        return ""
    key = value.strip()
    while len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
        key = key[1:-1].strip()
    return key


def get_api_key(env_name: str) -> str:
    return normalize_api_key(os.environ.get(env_name))
