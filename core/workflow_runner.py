"""
Small workflow execution helpers.

These helpers normalize loosely structured LLM/plugin outputs before later
stages consume them.
"""
from __future__ import annotations

import json
from typing import Any


def to_str(x: Any, default: str = "") -> str:
    """Return a string value or the provided default."""
    if x is None:
        return default
    if isinstance(x, str):
        return x
    return default


def to_json_str(x: Any) -> str:
    """Serialize structured values to JSON text."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def dig(obj: Any, *path: str, default: Any = None) -> Any:
    """Read a nested value from dictionaries/lists with a default fallback."""
    cur = obj
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and key.isdigit():
            idx = int(key)
            cur = cur[idx] if 0 <= idx < len(cur) else None
        else:
            return default
        if cur is None:
            return default
    return cur


def safe_object(schema: dict[str, Any], data: Any) -> dict[str, Any]:
    """Normalize a dictionary against a default-value schema."""
    if not isinstance(data, dict):
        return dict(schema)
    out = dict(data)
    for k, default in schema.items():
        v = out.get(k)
        if v is None:
            out[k] = default
        elif isinstance(default, str) and not isinstance(v, str):
            # Keep string fields predictable for prompt rendering.
            out[k] = default
    return out


def coalesce_list(value: Any) -> list:
    """Return a list, wrapping scalar values and treating None as empty."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def as_bool(value: Any, default: bool = False) -> bool:
    """Normalize common boolean representations."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y", "t"):
            return True
        if v in ("false", "0", "no", "n", "f", ""):
            return False
    return default
