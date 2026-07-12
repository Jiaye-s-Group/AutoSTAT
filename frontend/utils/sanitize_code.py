from core.visualization_code_sanitizer import (
    sanitize_visualization_code as sanitize_generated_visualization_code,
)


def sanitize_code(code):
    """Strip Markdown fences and surrounding whitespace from code."""
    code = code.strip()

    if code.startswith('```python'):
        code = code[9:]
    elif code.startswith('```'):
        code = code[3:]
    
    if code.endswith('```'):
        code = code[:-3]
    
    code = code.strip()
    
    return code


def sanitize_visualization_code(code):
    """Sanitize generated visualization code before frontend execution."""
    return sanitize_generated_visualization_code(sanitize_code(code))


def to_json_serializable(obj):
    """
    将对象转换为可JSON序列化的类型
    
    Args:
        obj: 要转换的对象
    
    Returns:
        可JSON序列化的对象
    """
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    else:
        return obj
