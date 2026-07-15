"""Compact, data-derived constraints shared by generated-code stages."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import pandas as pd


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
    n_rows = int(len(df.index)) if len(df.index) else int(fallback_rows or 0)
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
        "dataset": {
            "n_rows": n_rows,
            "n_columns": n_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "constant_columns": constant_columns,
            "target": target_info,
        },
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
