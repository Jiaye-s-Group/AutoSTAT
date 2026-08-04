"""Compact full-table metadata for data-understanding prompts."""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd


def _json_safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return str(value)
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, (bool, int, str)):
        return value
    return str(value)


def _numeric_profile(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {}
    return {
        "min": _json_safe_scalar(numeric.min()),
        "p25": _json_safe_scalar(numeric.quantile(0.25)),
        "median": _json_safe_scalar(numeric.median()),
        "mean": _json_safe_scalar(numeric.mean()),
        "p75": _json_safe_scalar(numeric.quantile(0.75)),
        "max": _json_safe_scalar(numeric.max()),
        "std": _json_safe_scalar(numeric.std(ddof=0)),
    }


def _datetime_profile(series: pd.Series) -> dict[str, Any]:
    values = series.dropna()
    if values.empty:
        return {}
    return {
        "min": _json_safe_scalar(values.min()),
        "max": _json_safe_scalar(values.max()),
    }


def _top_values(series: pd.Series, *, limit: int) -> list[dict[str, Any]]:
    try:
        counts = series.dropna().value_counts().head(limit)
    except Exception:
        return []
    return [
        {"value": _json_safe_scalar(value), "count": int(count)}
        for value, count in counts.items()
    ]


def build_dataframe_profile(
    df: pd.DataFrame,
    *,
    max_profile_columns: int = 120,
    top_value_count: int = 5,
) -> dict[str, Any]:
    """Return prompt-safe metadata computed from the full DataFrame."""
    if df is None or df.empty:
        return {
            "full_dataset_loaded": False,
            "preview_rows_are_sample_only": True,
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "column_profiles": {},
        }

    row_count, column_count = df.shape
    dtype_by_column = {str(column): str(dtype) for column, dtype in df.dtypes.items()}
    missing_counts = df.isna().sum()
    missing_by_column = {
        str(column): int(count)
        for column, count in missing_counts.items()
        if int(count) > 0
    }
    dtype_counts = {
        str(dtype): int(count)
        for dtype, count in df.dtypes.astype(str).value_counts().items()
    }

    column_profiles: dict[str, dict[str, Any]] = {}
    for column in list(df.columns)[:max_profile_columns]:
        series = df[column]
        name = str(column)
        missing_count = int(missing_counts[column])
        profile: dict[str, Any] = {
            "dtype": dtype_by_column[name],
            "non_missing_count": int(row_count - missing_count),
            "missing_count": missing_count,
            "missing_pct": round(missing_count / max(1, row_count), 6),
        }
        try:
            profile["unique_count"] = int(series.nunique(dropna=True))
        except Exception:
            profile["unique_count"] = None

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            profile["numeric_summary"] = _numeric_profile(series)
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile["datetime_summary"] = _datetime_profile(series)
        else:
            profile["top_values"] = _top_values(series, limit=top_value_count)
        column_profiles[name] = profile

    return {
        "full_dataset_loaded": True,
        "preview_rows_are_sample_only": True,
        "row_count": int(row_count),
        "column_count": int(column_count),
        "total_cells": int(row_count * column_count),
        "total_missing_values": int(missing_counts.sum()),
        "columns_with_missing": missing_by_column,
        "dtype_counts": dtype_counts,
        "columns": [str(column) for column in df.columns],
        "profiled_column_count": len(column_profiles),
        "omitted_profile_column_count": max(0, int(column_count) - len(column_profiles)),
        "column_profiles": column_profiles,
    }


def build_dataframe_profile_json(df: pd.DataFrame, **kwargs: Any) -> str:
    return json.dumps(build_dataframe_profile(df, **kwargs), ensure_ascii=False, default=str)
