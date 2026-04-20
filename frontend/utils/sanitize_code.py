import ast


def sanitize_code(code):
    """
    清理和标准化代码字符串
    
    Args:
        code: 原始代码字符串
    
    Returns:
        清理后的代码字符串
    """
    # 移除代码前后的空白字符
    code = code.strip()
    
    # 移除代码中的 Markdown 代码块标记
    if code.startswith('```python'):
        code = code[9:]
    elif code.startswith('```'):
        code = code[3:]
    
    if code.endswith('```'):
        code = code[:-3]
    
    # 移除代码前后的空白字符
    code = code.strip()
    
    return code


def sanitize_visualization_code(code):
    """
    清理可视化代码，并强制保留运行环境中提供的全量 df。

    规则：
    - 保留原有的 markdown code fence 清理逻辑
    - 删除任何把 Name('df') 重新赋值的语句
    - 如果需要数据变换，应让上游 LLM 使用 plot_df / agg_df 等新变量名
    """
    code = sanitize_code(code)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    class _DropDfReassign(ast.NodeTransformer):
        @staticmethod
        def _is_df_name(target):
            return isinstance(target, ast.Name) and target.id == "df"

        def visit_Assign(self, node):
            if any(self._is_df_name(target) for target in node.targets):
                return None
            return self.generic_visit(node)

        def visit_AnnAssign(self, node):
            if self._is_df_name(node.target):
                return None
            return self.generic_visit(node)

        def visit_AugAssign(self, node):
            if self._is_df_name(node.target):
                return None
            return self.generic_visit(node)

        def visit_NamedExpr(self, node):
            if self._is_df_name(node.target):
                return None
            return self.generic_visit(node)

    sanitized_tree = _DropDfReassign().visit(tree)
    ast.fix_missing_locations(sanitized_tree)

    try:
        return ast.unparse(sanitized_tree).strip()
    except Exception:
        return code


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
