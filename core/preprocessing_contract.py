"""Semantic contract and post-execution checks for generated preprocessing code.

The code sandbox can establish that a script runs, but it cannot establish that
the script honoured the user-approved preprocessing plan.  This module keeps a
small, conservative contract alongside that plan and validates observable
input/output behaviour.  Constraints are only enabled when they are stated
unambiguously, so ordinary preprocessing tasks retain their existing freedom.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import pandas as pd


_RETAIN_ALL_ROWS = re.compile(
    r"(?:保留(?:全部|所有)?\s*\d*[\s,，]*行|保留全部记录|不(?:删|删除)(?:除)?(?:数据)?行|"
    r"不删除记录|仅(?:做)?质量控制[^。；;]{0,80}不删除|"
    r"keep\s+all\s+(?:rows|records)|retain\s+all\s+rows|do\s+not\s+delete\s+(?:any\s+)?rows)",
    re.IGNORECASE,
)
_NO_NEW_FIELDS = re.compile(
    r"(?:不创建(?:任何)?(?:富集|峰值|下游模型|派生|新)?变量|不进行(?:任何)?特征工程|"
    r"仅保留原始字段|不(?:新增|创建)列|do\s+not\s+(?:create|add)\s+(?:new\s+)?(?:features|columns))",
    re.IGNORECASE,
)
_QC_REQUESTED = re.compile(r"(?:QC\s*(?:摘要|summary|记录|核查)|质量控制(?:摘要|记录|核查)?|可复核|可审计)", re.I)
_ACTUAL_MISSING = re.compile(r"(?:实际|真实)(?:的)?缺失|actual(?:ly)?\s+missing|true\s+missing", re.I)
_NO_STANDARDIZATION = re.compile(r"(?:不(?:进行|做)?(?:任何)?(?:标准化|归一化|中心化|缩放|类别编码|log1p?|对数变换)|no\s+(?:standardi[sz]ation|normalization|log(?:1p)?\s*transform|encoding))", re.I)


def contract_as_prompt(contract: dict[str, Any]) -> str:
    return json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True)


def build_preprocessing_contract(
    *,
    columns: list[Any],
    user_input: str = "",
    add_preference: str = "",
    suggestion: str = "",
    refined_suggestions: str = "",
) -> dict[str, Any]:
    """Create only constraints that can be inferred without guessing intent."""
    column_names = [str(column) for column in columns]
    text = "\n".join(
        part for part in (user_input, add_preference, suggestion, refined_suggestions) if str(part or "").strip()
    )
    retain_all_rows = bool(_RETAIN_ALL_ROWS.search(text))
    preserve_schema = bool(_NO_NEW_FIELDS.search(text))
    require_qc_summary = bool(_QC_REQUESTED.search(text))

    actual_missing_fill_columns: list[str] = []
    for column in column_names:
        escaped = re.escape(column)
        nearby = re.compile(
            rf"(?:仅|只|only).{{0,24}}{escaped}.{{0,90}}(?:填补|填充|imput)|"
            rf"{escaped}.{{0,90}}(?:仅|只|only).{{0,35}}(?:填补|填充|imput)|"
            rf"{escaped}.{{0,90}}(?:实际|真实|actual(?:ly)?|true).{{0,35}}(?:缺失|missing)",
            re.I,
        )
        if nearby.search(text) and _ACTUAL_MISSING.search(text):
            actual_missing_fill_columns.append(column)

    # When the approved plan says that *only* actual missing values in named
    # columns can be filled, every remaining source column is immutable.
    strict_minimal_changes = bool(
        actual_missing_fill_columns
        and re.search(r"(?:除(?:了|外)|其余字段|其他字段|不对|only\s+(?:actual\s+)?missing)", text, re.I)
    )
    immutable_columns = (
        [column for column in column_names if column not in actual_missing_fill_columns]
        if strict_minimal_changes
        else []
    )

    coordinate_qc = all(name in column_names for name in ("bin_id", "chromosome", "start_bp", "end_bp")) and bool(
        re.search(r"(?:end_bp|start_bp|bin_id|相邻(?:合格)?窗口|50\s*bp|坐标|coordinate)", text, re.I)
    )
    required_qc_keys = ["rows_before", "columns_before", "rows_after", "columns_after", "missing_by_column", "modified_fields", "modified_row_count"] if require_qc_summary else []
    if coordinate_qc and require_qc_summary:
        required_qc_keys.extend(
            [
                "duplicate_bin_id_count",
                "coordinate_or_id_mismatch_count",
                "end_bp_rule_violation_count",
                "adjacent_window_gap_violation_count",
            ]
        )

    return {
        "version": 1,
        "retain_all_rows": retain_all_rows,
        "preserve_schema": preserve_schema,
        "actual_missing_fill_columns": actual_missing_fill_columns,
        "immutable_columns": immutable_columns,
        "require_qc_summary": require_qc_summary,
        "required_qc_keys": required_qc_keys,
        "coordinate_qc": coordinate_qc,
        "prohibit_standardization_or_encoding": bool(_NO_STANDARDIZATION.search(text)),
        "valid": True,
        "issues": [],
    }


def _changed_mask(before: pd.Series, after: pd.Series) -> pd.Series:
    """Return values changed while treating two missing values as equal."""
    before_value = before.reset_index(drop=True)
    after_value = after.reset_index(drop=True)
    equal = before_value.eq(after_value)
    equal = equal | (before_value.isna() & after_value.isna())
    return ~equal.fillna(False)


def _coordinate_text(values: pd.Series) -> pd.Series:
    raw = values.astype("string")
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.notna() & numeric.map(math.isfinite)
    integral = finite & ((numeric - numeric.round()).abs() < 1e-9)
    result = raw.copy()
    if integral.any():
        result.loc[integral] = numeric.loc[integral].round().astype("int64").astype("string")
    return result


def _coordinate_qc_counts(df: pd.DataFrame) -> dict[str, int]:
    required = {"bin_id", "chromosome", "start_bp", "end_bp"}
    if not required.issubset(df.columns):
        return {
            "duplicate_bin_id_count": 0,
            "coordinate_or_id_mismatch_count": 0,
            "end_bp_rule_violation_count": 0,
            "adjacent_window_gap_violation_count": 0,
        }

    start = pd.to_numeric(df["start_bp"], errors="coerce")
    end = pd.to_numeric(df["end_bp"], errors="coerce")
    coordinate_present = start.notna() & end.notna()
    end_valid = coordinate_present & ((end - start - 49).abs() < 1e-9)
    bin_text = df["bin_id"].astype("string")
    expected = df["chromosome"].astype("string") + ":" + _coordinate_text(df["start_bp"]) + "-" + _coordinate_text(df["end_bp"])
    format_valid = bin_text.str.match(r"^[^:\s]+:-?\d+(?:\.\d+)?--?\d+(?:\.\d+)?$", na=False)
    bin_valid = coordinate_present & bin_text.eq(expected) & format_valid
    eligible = end_valid & bin_valid

    adjacent_bad = pd.Series(False, index=df.index)
    eligible_frame = pd.DataFrame(
        {"chromosome": df.loc[eligible, "chromosome"].astype("string"), "start": start.loc[eligible]}
    )
    if not eligible_frame.empty:
        ordered = eligible_frame.sort_values(["chromosome", "start"], kind="stable")
        gaps = ordered.groupby("chromosome", sort=False)["start"].diff()
        adjacent_bad.loc[ordered.index] = gaps.notna() & ((gaps - 50).abs() >= 1e-9)

    mismatch = ~(end_valid & bin_valid)
    return {
        "duplicate_bin_id_count": int(bin_text.duplicated(keep=False).sum()),
        "coordinate_or_id_mismatch_count": int(mismatch.sum()),
        "end_bp_rule_violation_count": int((~end_valid).sum()),
        "adjacent_window_gap_violation_count": int(adjacent_bad.sum()),
    }


def _normalise_fields(value: Any) -> list[str] | None:
    if not isinstance(value, (list, tuple, set)):
        return None
    return sorted({str(item) for item in value})


def validate_preprocessing_result(
    *,
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    qc_summary: Any,
    contract: dict[str, Any] | None,
) -> list[str]:
    """Validate observable preprocessing behaviour against a declared contract."""
    contract = dict(contract or {})
    if not contract:
        return []
    issues = list(contract.get("issues") or [])
    if not isinstance(output_df, pd.DataFrame):
        return issues + ["process_df must be a pandas.DataFrame."]

    if contract.get("retain_all_rows") and len(input_df) != len(output_df):
        issues.append(f"The approved plan retains all rows, but rows changed from {len(input_df)} to {len(output_df)}.")

    if contract.get("preserve_schema") and list(input_df.columns) != list(output_df.columns):
        issues.append("The approved plan preserves the source schema, but output columns changed.")

    comparable_columns = [column for column in input_df.columns if column in output_df.columns]
    actual_changed: dict[str, pd.Series] = {}
    if len(input_df) == len(output_df):
        for column in comparable_columns:
            changed = _changed_mask(input_df[column], output_df[column])
            if bool(changed.any()):
                actual_changed[str(column)] = changed

    for column in contract.get("immutable_columns") or []:
        if column in actual_changed:
            issues.append(f"Column {column!r} is immutable under the approved plan but was modified.")

    for column in contract.get("actual_missing_fill_columns") or []:
        if column not in input_df.columns or column not in output_df.columns or len(input_df) != len(output_df):
            continue
        changed = actual_changed.get(str(column))
        if changed is not None and bool((changed & ~input_df[column].isna().reset_index(drop=True)).any()):
            issues.append(f"Column {column!r} changed values that were not genuinely missing in the input.")

    if contract.get("require_qc_summary"):
        if not isinstance(qc_summary, dict):
            return issues + ["The approved plan requires a machine-readable qc_summary dictionary."]
        for key in contract.get("required_qc_keys") or []:
            if key not in qc_summary:
                issues.append(f"qc_summary.{key} is required by the approved plan.")
        expected_basics = {
            "rows_before": len(input_df),
            "columns_before": len(input_df.columns),
            "rows_after": len(output_df),
            "columns_after": len(output_df.columns),
        }
        for key, expected in expected_basics.items():
            if key in qc_summary and qc_summary.get(key) != expected:
                issues.append(f"qc_summary.{key} must be {expected!r}, got {qc_summary.get(key)!r}.")
        if "missing_by_column" in qc_summary:
            expected_missing = {str(column): int(output_df[column].isna().sum()) for column in output_df.columns}
            reported_missing = qc_summary.get("missing_by_column")
            if not isinstance(reported_missing, dict) or {
                str(key): value for key, value in reported_missing.items()
            } != expected_missing:
                issues.append("qc_summary.missing_by_column must report the actual missing count for every output column.")
        expected_fields = sorted(actual_changed)
        reported_fields = _normalise_fields(qc_summary.get("modified_fields"))
        if reported_fields is not None and reported_fields != expected_fields:
            issues.append(f"qc_summary.modified_fields must be {expected_fields!r}, got {reported_fields!r}.")
        if "modified_row_count" in qc_summary:
            changed_rows = pd.Series(False, index=range(len(input_df)))
            for changed in actual_changed.values():
                changed_rows = changed_rows | changed.reset_index(drop=True)
            expected_rows = int(changed_rows.sum())
            if qc_summary.get("modified_row_count") != expected_rows:
                issues.append(
                    f"qc_summary.modified_row_count must be {expected_rows!r}, got {qc_summary.get('modified_row_count')!r}."
                )
        if contract.get("coordinate_qc"):
            expected_counts = _coordinate_qc_counts(output_df)
            for key, expected in expected_counts.items():
                if key in qc_summary and qc_summary.get(key) != expected:
                    issues.append(f"qc_summary.{key} must be {expected!r}, got {qc_summary.get(key)!r}.")
    return issues


def format_preprocessing_contract_violations(issues: list[str]) -> str:
    return "Preprocessing contract validation failed:\n- " + "\n- ".join(issues)
