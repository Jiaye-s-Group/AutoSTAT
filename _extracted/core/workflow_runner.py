"""
Workflow 执行辅助工具。

抽出 12 个 Coze Code 节点里重复的"兜底/透传"模式：
- safe_object(schema, data): 按 schema 补齐字段 + 类型兜底
- dig(obj, path): 深层取值
- to_str / to_json_str: 安全字符串化
"""
from __future__ import annotations

import json
from typing import Any


def to_str(x: Any, default: str = "") -> str:
    """任何值 → 字符串；None/非字符串按 default 处理。"""
    if x is None:
        return default
    if isinstance(x, str):
        return x
    return default


def to_json_str(x: Any) -> str:
    """把 dict / list 转成 JSON 字符串，其他类型透传。"""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def dig(obj: Any, *path: str, default: Any = None) -> Any:
    """dig(d, 'a', 'b', 'c') == d['a']['b']['c'], 找不到返回 default。"""
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
    """
    按 schema 规范 data，补齐缺失字段。

    schema 示例：{"title": "", "desc": "", "df": ""}
    （values 表示每个字段的默认值）

    Coze 里的 "composer" 和 "final_list" code 节点都在做这类兜底。
    """
    if not isinstance(data, dict):
        return dict(schema)
    out = dict(data)
    for k, default in schema.items():
        v = out.get(k)
        if v is None:
            out[k] = default
        elif isinstance(default, str) and not isinstance(v, str):
            # 字段期望字符串但值不是 → 兜底
            out[k] = default
    return out


def coalesce_list(value: Any) -> list:
    """不是 list 就包成 list；None 变空 list。"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def as_bool(value: Any, default: bool = False) -> bool:
    """把各种形态的 bool 输入归一化。"""
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
