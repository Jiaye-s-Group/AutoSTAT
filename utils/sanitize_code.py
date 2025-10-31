import re
from typing import Any
import numpy as np


def sanitize_code(code: str) -> str:

    if not isinstance(code, str):
        return ""
    code = code.strip()
    if code.startswith("```") and code.endswith("```"):
        lines = code.splitlines()
        if re.match(r"^```(?:python)?", lines[0].strip()):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


def to_json_serializable(obj: Any) -> Any:
    """将可能含 numpy 类型的对象转换为可 JSON 序列化类型（递归）。"""
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        # 确保是内置 float（JSON 支持）
        return float(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    # fallback: try to cast to float / str
    try:
        return float(obj)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None

