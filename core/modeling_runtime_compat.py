"""Deterministic compatibility checks for generated modeling code.

These checks deliberately reject known removed scikit-learn parameters instead
of silently rewriting a user's analysis. The same check is used by automatic
validation and the front-end execution path.
"""

from __future__ import annotations

import ast
from importlib.metadata import PackageNotFoundError, version

from core.code_runtime_profile import LARGE_DATASET_ROW_THRESHOLD

_LARGE_DATASET_INFLUENCE_ERROR = (
    "Large-data runtime guard: resid_studentized_external uses a leave-one-out "
    "variance calculation for every observation. On a full dataset larger than "
    f"{LARGE_DATASET_ROW_THRESHOLD:,} rows, calculate that optional diagnostic "
    "only on a deterministic capped diagnostic sample, or use an internal "
    "residual diagnostic; keep the primary model's n_obs unchanged."
)
_LARGE_DATASET_RECORD_EXPORT_ERROR = (
    "Large-data runtime guard: do not embed a full table with "
    "to_dict(orient='records') in result_dict on a large dataset. Return a "
    "bounded preview/top-N rows plus row_count/schema, or use a managed table "
    "artifact reference."
)


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


def _last_called_name(node: ast.expr) -> str:
    return _called_name(node).split(".")[-1]


def _combine_scales(scales: list[str]) -> str:
    if "full" in scales:
        return "full"
    if scales and all(scale == "bounded" for scale in scales):
        return "bounded"
    return "unknown"


def _expression_scale(node: ast.AST | None, scales: dict[str, str]) -> str:
    """Infer whether a dataframe expression preserves the full input sample.

    This intentionally handles only transparent, common dataframe chains.  If
    the source cannot be established, validation does not reject the code.
    """
    if node is None:
        return "unknown"
    if isinstance(node, ast.Name):
        return scales.get(node.id, "unknown")
    if isinstance(node, ast.Attribute):
        return _expression_scale(node.value, scales)
    if isinstance(node, ast.Subscript):
        return _expression_scale(node.value, scales)
    if isinstance(node, ast.Call):
        name = _last_called_name(node.func)
        if name in {"sample", "head"}:
            return "bounded"
        if name in {"tail", "nlargest", "nsmallest"}:
            return "bounded"
        receiver_scale = (
            _expression_scale(node.func.value, scales)
            if isinstance(node.func, ast.Attribute)
            else "unknown"
        )
        argument_scales = [_expression_scale(arg, scales) for arg in node.args]
        return _combine_scales([receiver_scale, *argument_scales])
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return _combine_scales([_expression_scale(item, scales) for item in node.elts])
    return "unknown"


def _model_input_scale(node: ast.AST, scales: dict[str, str]) -> str:
    """Return the scale of a statsmodels model expression, when recognizable."""
    model_constructors = {
        "OLS",
        "ols",
        "WLS",
        "wls",
        "GLS",
        "gls",
        "GLSAR",
        "RLM",
        "rlm",
        "GLM",
        "glm",
        "Logit",
        "logit",
        "Probit",
        "probit",
        "MNLogit",
        "mnlogit",
    }
    matches: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or _last_called_name(child.func) not in model_constructors:
            continue
        input_nodes = [*child.args[:2]]
        input_nodes.extend(
            keyword.value
            for keyword in child.keywords
            if keyword.arg in {"endog", "exog", "y", "X", "data"}
        )
        matches.append(_combine_scales([_expression_scale(item, scales) for item in input_nodes]))
    return _combine_scales(matches)


def _influence_scale(node: ast.AST, scales: dict[str, str]) -> str:
    direct_scale = _expression_scale(node, scales)
    if direct_scale != "unknown":
        return direct_scale
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _last_called_name(child.func) == "get_influence":
            if isinstance(child.func, ast.Attribute):
                return _expression_scale(child.func.value, scales)
    return "unknown"


def _assignment_targets(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [name for item in node.elts for name in _assignment_targets(item)]
    return []


def _large_dataset_external_residual_issues(tree: ast.Module) -> list[str]:
    """Detect the known leave-one-out diagnostic on a transparently full sample."""
    scales: dict[str, str] = {"df": "full"}
    issues: list[str] = []

    assignments = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ),
        key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)),
    )
    for assignment in assignments:
        value = assignment.value
        if value is None:
            continue
        for attribute in ast.walk(value):
            if not isinstance(attribute, ast.Attribute) or attribute.attr != "resid_studentized_external":
                continue
            if _influence_scale(attribute.value, scales) == "full":
                issues.append(_LARGE_DATASET_INFLUENCE_ERROR)

        inferred_scale = _model_input_scale(value, scales)
        if inferred_scale == "unknown":
            inferred_scale = _influence_scale(value, scales)
        if inferred_scale == "unknown":
            inferred_scale = _expression_scale(value, scales)
        targets = (
            [name for target in assignment.targets for name in _assignment_targets(target)]
            if isinstance(assignment, ast.Assign)
            else _assignment_targets(assignment.target)
        )
        for target in targets:
            scales[target] = inferred_scale

    return list(dict.fromkeys(issues))


def _large_dataset_records_export_issues(tree: ast.Module) -> list[str]:
    """Detect full-sample DataFrame.to_dict(orient='records') exports."""
    scales: dict[str, str] = {"df": "full"}
    issues: list[str] = []

    statements = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr))
        ),
        key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)),
    )
    for statement in statements:
        value = getattr(statement, "value", None)
        if value is None:
            continue
        for call in ast.walk(value):
            if not isinstance(call, ast.Call):
                continue
            if _last_called_name(call.func) != "to_dict":
                continue
            orient = None
            for keyword in call.keywords:
                if keyword.arg == "orient":
                    orient = _literal_string(keyword.value)
                    break
            if orient != "records":
                continue
            receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
            if _expression_scale(receiver, scales) == "full":
                issues.append(_LARGE_DATASET_RECORD_EXPORT_ERROR)

        inferred_scale = _model_input_scale(value, scales)
        if inferred_scale == "unknown":
            inferred_scale = _influence_scale(value, scales)
        if inferred_scale == "unknown":
            inferred_scale = _expression_scale(value, scales)
        targets = (
            [name for target in statement.targets for name in _assignment_targets(target)]
            if isinstance(statement, ast.Assign)
            else _assignment_targets(statement.target)
            if isinstance(statement, ast.AnnAssign)
            else []
        )
        for target in targets:
            scales[target] = inferred_scale

    return list(dict.fromkeys(issues))


def validate_modeling_runtime_compatibility(
    code: str,
    *,
    sklearn_version: str | None = None,
    n_rows: int = 0,
) -> list[str]:
    """Return deterministic runtime and method-integrity errors.

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
        if call_name == "NegativeBinomial":
            is_discrete_count_model = (
                len(node.args) >= 2
                or "endog" in keywords
                or "exog" in keywords
            )
            if not is_discrete_count_model and "alpha" not in keywords:
                errors.append(
                    "statsmodels GLM NegativeBinomial() requires an explicit alpha. "
                    "The default alpha=1.0 is not an estimated dispersion parameter; "
                    "supply a justified alpha or fit a discrete NegativeBinomial model "
                    "with endog and exog so dispersion is estimated."
                )

    if int(n_rows or 0) > LARGE_DATASET_ROW_THRESHOLD:
        errors.extend(_large_dataset_external_residual_issues(tree))
        errors.extend(_large_dataset_records_export_issues(tree))

    return list(dict.fromkeys(errors))
