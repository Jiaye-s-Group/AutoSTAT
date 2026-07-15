"""Deterministic compatibility checks for generated modeling code.

These checks deliberately reject known removed scikit-learn parameters instead
of silently rewriting a user's analysis. The same check is used by automatic
validation and the front-end execution path.
"""

from __future__ import annotations

import ast
from importlib.metadata import PackageNotFoundError, version


def installed_sklearn_version() -> str:
    try:
        return version("scikit-learn")
    except PackageNotFoundError:
        return "unknown"


def _version_at_least(value: str, major: int, minor: int) -> bool:
    parts: list[int] = []
    for piece in str(value).split("."):
        digits = "".join(character for character in piece if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) >= 2:
            break
    if not parts:
        return False
    parts.extend([0] * (2 - len(parts)))
    return tuple(parts[:2]) >= (major, minor)


def _called_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        suffix = _called_name(node.value)
        return f"{suffix}.{node.attr}" if suffix else node.attr
    return ""


def _literal_string(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def validate_modeling_runtime_compatibility(
    code: str,
    *,
    sklearn_version: str | None = None,
) -> list[str]:
    """Return errors for removed APIs in the active scikit-learn runtime.

    Syntax errors remain the responsibility of the existing safe-code checker;
    returning no compatibility errors for unparsable code preserves its clearer
    syntax diagnostic.
    """
    try:
        tree = ast.parse(str(code or ""), mode="exec")
    except SyntaxError:
        return []

    active_version = sklearn_version or installed_sklearn_version()
    has_modern_logistic = _version_at_least(active_version, 1, 8)
    has_modern_encoder = _version_at_least(active_version, 1, 4)
    has_modern_estimators = _version_at_least(active_version, 1, 4)
    has_modern_kmeans = _version_at_least(active_version, 1, 3)
    errors: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _called_name(node.func).split(".")[-1]
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}

        if has_modern_logistic and call_name in {"LogisticRegression", "LogisticRegressionCV"}:
            if "multi_class" in keywords:
                errors.append(
                    f"scikit-learn {active_version} does not accept "
                    f"{call_name}(multi_class=...). Remove the multi_class keyword."
                )
        if has_modern_encoder and call_name == "OneHotEncoder" and "sparse" in keywords:
            errors.append(
                f"scikit-learn {active_version} does not accept OneHotEncoder(sparse=...). "
                "Use sparse_output=... instead."
            )
        if has_modern_estimators and call_name in {"AdaBoostClassifier", "AdaBoostRegressor", "BaggingClassifier", "BaggingRegressor"}:
            if "base_estimator" in keywords:
                errors.append(
                    f"scikit-learn {active_version} uses estimator=... rather than "
                    f"{call_name}(base_estimator=...)."
                )
        if has_modern_kmeans and call_name in {"KMeans", "MiniBatchKMeans"}:
            algorithm = _literal_string(keywords.get("algorithm"))
            if algorithm in {"auto", "full"}:
                errors.append(
                    f"scikit-learn {active_version} does not accept {call_name}(algorithm={algorithm!r}). "
                    "Use 'lloyd' or 'elkan'."
                )

    return list(dict.fromkeys(errors))
