"""Utilities for sanitizing generated Plotly visualization code."""

from __future__ import annotations

import ast
import inspect
from functools import lru_cache


_FORBIDDEN_TRENDLINE_OPTION_KEYS = {
    "ci",
    "conf_int",
    "confidence",
    "confidence_interval",
    "confidence_level",
}

_TRENDLINE_OPTION_ALLOWLISTS = {
    "ols": {"add_constant", "log_x", "log_y"},
    "lowess": {"frac"},
    "rolling": {"window", "win_type", "function", "function_args", "min_periods", "center"},
    "expanding": {"function", "function_args", "min_periods"},
    "ewm": {
        "com",
        "span",
        "halflife",
        "alpha",
        "min_periods",
        "adjust",
        "ignore_na",
        "function",
        "function_args",
    },
}


def unwrap_code_block(text: str) -> str:
    """Remove a surrounding Markdown code fence from generated code."""
    value = str(text or "").strip()
    if not value.startswith("```"):
        return value

    lines = value.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def sanitize_visualization_code(code: str) -> str:
    """Sanitize generated visualization code before execution."""
    source = unwrap_code_block(code)
    if not source:
        return ""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    sanitized_tree = _VisualizationCodeTransformer().visit(tree)
    ast.fix_missing_locations(sanitized_tree)
    try:
        return ast.unparse(sanitized_tree).strip()
    except Exception:
        return source


class _VisualizationCodeTransformer(ast.NodeTransformer):
    @staticmethod
    def _is_df_name(target: ast.AST) -> bool:
        return isinstance(target, ast.Name) and target.id == "df"

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        if any(self._is_df_name(target) for target in node.targets):
            return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if self._is_df_name(node.target):
            return None
        return self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST | None:
        if self._is_df_name(node.target):
            return None
        return self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.AST | None:
        if self._is_df_name(node.target):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        px_function = _plotly_express_function_name(node)
        if not px_function:
            return node

        allowed_keywords = _plotly_express_keywords(px_function)
        if allowed_keywords is not None:
            node.keywords = [
                keyword
                for keyword in node.keywords
                if keyword.arg is None or keyword.arg in allowed_keywords
            ]

        node.keywords = _sanitize_trendline_options(node.keywords)
        return node


def _plotly_express_function_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id == "px":
            return func.attr
    return ""


@lru_cache(maxsize=128)
def _plotly_express_keywords(function_name: str) -> set[str] | None:
    try:
        import plotly.express as px  # type: ignore

        func = getattr(px, function_name, None)
        if func is None:
            return None
        signature = inspect.signature(func)
    except Exception:
        return None

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return None
    return set(signature.parameters)


def _sanitize_trendline_options(keywords: list[ast.keyword]) -> list[ast.keyword]:
    trendline = ""
    for keyword in keywords:
        if keyword.arg == "trendline":
            trendline = _literal_string(keyword.value).lower()
            break

    sanitized: list[ast.keyword] = []
    for keyword in keywords:
        if keyword.arg != "trendline_options":
            sanitized.append(keyword)
            continue

        cleaned_value = _clean_trendline_options_value(keyword.value, trendline)
        if cleaned_value is not None:
            sanitized.append(ast.keyword(arg=keyword.arg, value=cleaned_value))
    return sanitized


def _clean_trendline_options_value(value: ast.AST, trendline: str) -> ast.AST | None:
    allowed = _TRENDLINE_OPTION_ALLOWLISTS.get(trendline)
    if isinstance(value, ast.Dict):
        return _clean_options_dict(value, allowed)
    if isinstance(value, ast.Call) and _is_dict_call(value):
        return _clean_options_call(value, allowed)

    if trendline == "ols":
        return None
    return value


def _clean_options_dict(node: ast.Dict, allowed: set[str] | None) -> ast.Dict | None:
    keys: list[ast.AST | None] = []
    values: list[ast.AST] = []
    for key_node, value_node in zip(node.keys, node.values):
        key = _literal_string(key_node)
        if key and _should_keep_option_key(key, allowed):
            keys.append(key_node)
            values.append(value_node)
        elif key is None and allowed is None:
            keys.append(key_node)
            values.append(value_node)
    if not keys:
        return None
    return ast.Dict(keys=keys, values=values)


def _clean_options_call(node: ast.Call, allowed: set[str] | None) -> ast.Call | None:
    positional_args = node.args if allowed is None else []
    keywords = [
        keyword
        for keyword in node.keywords
        if keyword.arg is None or _should_keep_option_key(keyword.arg, allowed)
    ]
    if not positional_args and not keywords:
        return None
    return ast.Call(func=node.func, args=positional_args, keywords=keywords)


def _should_keep_option_key(key: str, allowed: set[str] | None) -> bool:
    normalized = key.strip()
    if normalized in _FORBIDDEN_TRENDLINE_OPTION_KEYS:
        return False
    if allowed is not None and normalized not in allowed:
        return False
    return True


def _literal_string(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip()
    return ""


def _is_dict_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "dict"
