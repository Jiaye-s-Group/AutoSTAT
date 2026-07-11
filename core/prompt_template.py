"""
Small prompt-template renderer used by AutoSTAT workflows.

Supported syntax:
- simple variables: {{var_name}}
- dotted lookup: {{obj.key}}

Missing values render as empty strings to keep prompts tidy.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_\.]*)\s*\}\}")

# Prompt templates are stored at the project root under `prompts/`.
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _stringify(value: Any) -> str:
    """Convert a value into stable prompt text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _resolve(name: str, ctx: dict[str, Any]) -> str:
    """Resolve simple and dotted variable names."""
    if "." not in name:
        return _stringify(ctx.get(name))
    parts = name.split(".")
    val: Any = ctx
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            val = getattr(val, p, None)
        if val is None:
            return ""
    return _stringify(val)


def render(template: str, ctx: dict[str, Any]) -> str:
    """Render a template string with values from `ctx`."""
    if not template:
        return ""
    return _PLACEHOLDER_PATTERN.sub(lambda m: _resolve(m.group(1), ctx), template)


def render_file(relative_path: str | Path, ctx: dict[str, Any]) -> str:
    """Render a template file under `prompts/`."""
    path = PROMPTS_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Prompt 模板不存在: {path}")
    return render(path.read_text(encoding="utf-8"), ctx)


def find_missing_vars(template: str, ctx: dict[str, Any]) -> list[str]:
    """Return template variables that are missing from `ctx`."""
    missing = []
    for m in _PLACEHOLDER_PATTERN.finditer(template):
        name = m.group(1).split(".")[0]
        if name not in ctx:
            missing.append(name)
    return missing
