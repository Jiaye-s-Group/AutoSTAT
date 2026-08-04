"""Compact, data-derived constraints shared by generated-code stages."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import pandas as pd

from core.safe_code import generated_code_execution_policy


# Generated code is validated in a bounded subprocess.  These values describe
# an execution envelope, not a modeling choice: they prevent diagnostics with
# leave-one-out or quadratic work from turning a large-data analysis into an
# unresponsive application.
LARGE_DATASET_ROW_THRESHOLD = 100_000
LARGE_DATASET_DIAGNOSTIC_MAX_ROWS = 50_000


def _to_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (TypeError, ValueError):
            return pd.DataFrame()
    if isinstance(data, list):
        try:
            return pd.DataFrame(data)
        except (TypeError, ValueError):
            return pd.DataFrame()
    return pd.DataFrame()


def infer_data_row_count(data: Any, *, fallback_rows: int = 0) -> int:
    """Return the available row count without materializing a DataFrame.

    Workflow validation receives JSON records while the interactive executor
    receives a DataFrame.  Keeping this conversion small makes the same
    resource checks available to both paths.
    """
    if isinstance(data, pd.DataFrame):
        return int(len(data.index))
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (TypeError, ValueError):
            return int(fallback_rows or 0)
    if isinstance(data, (list, tuple)):
        return int(len(data))
    return int(fallback_rows or 0)


def _execution_budget(n_rows: int) -> dict[str, Any]:
    is_large = n_rows > LARGE_DATASET_ROW_THRESHOLD
    rules = [
        "Prefer vectorized pandas/numpy operations to Python row loops.",
        "Do not materialize all pairwise distances, similarities, or cross-products unless their size is provably bounded.",
        "Keep returned tables and diagnostic artifacts bounded; return summaries or top rows rather than one record per input row.",
    ]
    if is_large:
        rules.extend(
            [
                "Fit the primary model on the full eligible sample when the requested method scales to it; do not silently replace the primary analysis with a sample.",
                "For optional diagnostics, use a deterministic capped sample (random_state fixed, at most diagnostic_sample_max_rows) and report diagnostic_n separately from model n_obs.",
                "Do not run leave-one-out, per-observation refitting, full influence diagnostics, nested cross-validation, or unbounded Python row loops on the full dataset.",
                "In particular, do not access statsmodels resid_studentized_external on a full large sample: it computes leave-one-out variance. Use an internal residual diagnostic or a bounded diagnostic sample instead.",
            ]
        )
    return {
        "is_large_dataset": is_large,
        "large_dataset_row_threshold": LARGE_DATASET_ROW_THRESHOLD,
        "diagnostic_sample_max_rows": LARGE_DATASET_DIAGNOSTIC_MAX_ROWS,
        "rules": rules,
    }


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _version_at_least(value: str, major: int, minor: int) -> bool:
    """Compare the numeric prefix without adding a packaging dependency."""
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


def _modeling_library_profile() -> dict[str, Any]:
    """Describe only API rules that affect generated modeling code."""
    sklearn_version = _installed_version("scikit-learn")
    rules: list[str] = []
    if _version_at_least(sklearn_version, 1, 8):
        rules.append(
            "LogisticRegression and LogisticRegressionCV do not accept multi_class; omit that keyword."
        )
    if _version_at_least(sklearn_version, 1, 4):
        rules.append(
            "OneHotEncoder does not accept sparse; use sparse_output when a dense output is required."
        )
        rules.append(
            "AdaBoost and Bagging estimators use estimator, not base_estimator."
        )
    if _version_at_least(sklearn_version, 1, 3):
        rules.append(
            "KMeans algorithm accepts lloyd or elkan; do not use the removed auto or full values."
        )
    return {
        "scikit_learn": {
            "version": sklearn_version,
            "compatibility_rules": rules,
        }
    }


def build_code_runtime_constraints(
    data: Any,
    *,
    target: str = "",
    fallback_rows: int = 0,
    fallback_columns: int = 0,
    include_modeling_library_compatibility: bool = False,
) -> str:
    """Return JSON constraints that prevent invalid data-dependent parameters.

    The profile reports constraints only. It never chooses an analysis method or
    changes the user's requested statistical task.
    """
    df = _to_dataframe(data)
    n_rows = infer_data_row_count(data, fallback_rows=fallback_rows)
    n_columns = int(len(df.columns)) if len(df.columns) else int(fallback_columns or 0)
    numeric_columns = (
        int(len(df.select_dtypes(include="number").columns)) if not df.empty else 0
    )
    categorical_columns = max(0, n_columns - numeric_columns)
    constant_columns: list[str] = []
    if not df.empty:
        constant_columns = [
            str(column)
            for column in df.columns
            if int(df[column].nunique(dropna=False)) <= 1
        ][:20]

    target_info: dict[str, Any] = {
        "available": False,
        "n_classes": 0,
        "minimum_class_count": 0,
    }
    if target and target in df.columns:
        values = df[target].dropna()
        counts = values.value_counts(dropna=False)
        target_info = {
            "available": True,
            "n_classes": int(len(counts)),
            "minimum_class_count": int(counts.min()) if not counts.empty else 0,
        }

    max_silhouette_clusters = max(0, n_rows - 1)
    max_pca_components = max(0, min(n_rows, numeric_columns))
    profile = {
        "execution_safety": generated_code_execution_policy(),
        "dataset": {
            "n_rows": n_rows,
            "n_columns": n_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "constant_columns": constant_columns,
            "target": target_info,
        },
        "execution_budget": _execution_budget(n_rows),
        "parameter_bounds": {
            "clustering": {
                "silhouette_minimum_samples": 3,
                "silhouette_cluster_range": (
                    {"min": 2, "max": max_silhouette_clusters}
                    if max_silhouette_clusters >= 2
                    else None
                ),
            },
            "cross_validation": {
                "maximum_splits_by_sample_count": n_rows,
                "maximum_stratified_splits": target_info["minimum_class_count"],
            },
            "nearest_neighbors": {
                "maximum_neighbors_before_split": max(0, n_rows - 1),
            },
            "dimension_reduction": {
                "maximum_components": max_pca_components,
            },
        },
        "rules": [
            "Recompute bounds after filtering, dropping missing values, or train/test splitting.",
            "Do not evaluate an algorithm or metric when its mathematical preconditions are not met.",
            "Do not use constant columns as model features unless explicitly required.",
        ],
    }
    if include_modeling_library_compatibility:
        profile["libraries"] = _modeling_library_profile()
    return json.dumps(profile, ensure_ascii=False, sort_keys=True)
