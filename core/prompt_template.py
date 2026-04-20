"""
Coze workflow 里 {{var}} 占位符的渲染器。

Coze 使用的是 Jinja2-like 的双大括号语法，但我们只需要支持：
- 简单变量：{{var_name}}
- 不支持条件/循环（Coze workflow 里没用到）
- 不存在的变量渲染为空字符串（防止 "None" 出现在 prompt 里）

PROMPTS_DIR 统一指向 autostat_local/prompts/
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_\.]*)\s*\}\}")

# prompts 目录固定在 autostat_local 根目录
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _stringify(value: Any) -> str:
    """把任意值转成适合塞进 prompt 的字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    # dict / list 转 JSON（让 LLM 看得清结构）
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _resolve(name: str, ctx: dict[str, Any]) -> str:
    """支持 {{obj.key}} 这样的 . 访问。找不到返回空串。"""
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
    """把 {{var}} 渲染成 ctx 里的值。"""
    if not template:
        return ""
    return _PLACEHOLDER_PATTERN.sub(lambda m: _resolve(m.group(1), ctx), template)


def render_file(relative_path: str | Path, ctx: dict[str, Any]) -> str:
    """
    渲染 prompts/ 下的模板文件。

    relative_path 示例:
      - "loading/do_data_description__llm_sys.txt"
      - Path("loading") / "do_data_description__llm_sys.txt"
    """
    path = PROMPTS_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Prompt 模板不存在: {path}")
    return render(path.read_text(encoding="utf-8"), ctx)


def find_missing_vars(template: str, ctx: dict[str, Any]) -> list[str]:
    """调试用：返回模板里出现了但 ctx 没提供的变量名。"""
    missing = []
    for m in _PLACEHOLDER_PATTERN.finditer(template):
        name = m.group(1).split(".")[0]
        if name not in ctx:
            missing.append(name)
    return missing
