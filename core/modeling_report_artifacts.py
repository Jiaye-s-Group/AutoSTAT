from __future__ import annotations

import json
import math
import re
from typing import Any


MAX_MODELS = 40
MAX_COEFFICIENT_ROWS = 200
MAX_DIAGNOSTIC_ROWS = 120
MAX_CANDIDATE_ROWS = 200
MAX_CELL_CHARS = 500

_ARTIFACT_KEY_RE = re.compile(
    r"(?:b64|base64|bytes|pickle|joblib|image|png|jpg|jpeg|svg|html|figure|fig_json|plotly)",
    re.I,
)

_CANDIDATE_WINDOW_KEYS = (
    "candidate_windows",
    "candidate_window_results",
    "candidate_bins",
    "candidate_bin_results",
    "candidate_regions",
    "candidate_results",
    "window_results",
    "top_windows",
    "top_candidate_windows",
)

_CANDIDATE_SEGMENT_KEYS = (
    "candidate_segments",
    "candidate_segment_results",
    "segments",
    "segment_results",
    "top_segments",
    "top_candidate_segments",
)

_DIAGNOSTIC_KEYS = (
    "diagnostics",
    "model_diagnostics",
    "residual_summary",
    "residuals_summary",
    "residual_diagnostics",
    "zero_inflation",
    "overdispersion",
)


def build_modeling_report_artifacts(
    result_json: Any,
    *,
    target: str = "",
    language: str = "zh",
) -> dict[str, Any]:
    """Normalize executed modeling outputs into a stable report-facing schema."""
    payload = maybe_json_loads(result_json)
    if not isinstance(payload, dict):
        return {
            "schema_version": 1,
            "available": False,
            "target": _safe_scalar(target),
            "language": str(language or "zh"),
            "issues": ["result_json is not a JSON object."],
        }

    models = _extract_models(payload)
    coefficients = _extract_coefficients(payload, models)
    diagnostics = _extract_diagnostics(payload, models)
    candidate_windows = _extract_named_records(payload, _CANDIDATE_WINDOW_KEYS)
    candidate_segments = _extract_named_records(payload, _CANDIDATE_SEGMENT_KEYS)

    return {
        "schema_version": 1,
        "available": True,
        "target": _safe_scalar(target),
        "language": str(language or "zh"),
        "analysis_manifest": _safe_value(payload.get("analysis_manifest")),
        "model_count": len(models),
        "models": _limit_rows(models, MAX_MODELS),
        "coefficients": _table_block(coefficients, max_rows=MAX_COEFFICIENT_ROWS),
        "diagnostics": _table_block(diagnostics, max_rows=MAX_DIAGNOSTIC_ROWS),
        "candidate_windows": _table_block(candidate_windows, max_rows=MAX_CANDIDATE_ROWS),
        "candidate_segments": _table_block(candidate_segments, max_rows=MAX_CANDIDATE_ROWS),
        "primary_outputs": {
            "has_models": bool(models),
            "has_coefficients": bool(coefficients),
            "has_diagnostics": bool(diagnostics),
            "has_candidate_windows": bool(candidate_windows),
            "has_candidate_segments": bool(candidate_segments),
            "has_candidate_results": bool(candidate_windows or candidate_segments),
        },
    }


def has_candidate_report_outputs(result_json: Any) -> bool:
    artifacts = build_modeling_report_artifacts(result_json)
    primary = artifacts.get("primary_outputs") if isinstance(artifacts, dict) else {}
    return bool(
        isinstance(primary, dict)
        and (
            primary.get("has_candidate_windows")
            or primary.get("has_candidate_segments")
            or primary.get("has_candidate_results")
        )
    )


def maybe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value


def _extract_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return []
    out: list[dict[str, Any]] = []
    for index, raw_model in enumerate(raw_models, start=1):
        if not isinstance(raw_model, dict):
            continue
        row: dict[str, Any] = {
            "index": index,
            "name": _safe_scalar(
                raw_model.get("name")
                or raw_model.get("model")
                or raw_model.get("model_name")
                or raw_model.get("method")
                or f"Model {index}"
            ),
        }
        for key in ("family", "type", "task_type", "n_obs", "selected", "rank"):
            if key in raw_model:
                row[key] = _safe_value(raw_model.get(key))
        if isinstance(raw_model.get("model_spec"), dict):
            row["model_spec"] = _safe_value(raw_model.get("model_spec"))
        if isinstance(raw_model.get("metrics"), dict):
            row["metrics"] = _safe_value(raw_model.get("metrics"))
        out.append(row)
    return out


def _extract_coefficients(payload: dict[str, Any], models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    del models
    rows: list[dict[str, Any]] = []
    raw_models = payload.get("models")
    if isinstance(raw_models, list):
        for index, raw_model in enumerate(raw_models, start=1):
            if not isinstance(raw_model, dict):
                continue
            model_name = _safe_scalar(
                raw_model.get("name")
                or raw_model.get("model")
                or raw_model.get("model_name")
                or f"Model {index}"
            )
            rows.extend(
                _coefficient_rows_from_value(
                    raw_model.get("coefficients") or raw_model.get("coef"),
                    model_name=model_name,
                )
            )

    for key in ("coefficients", "coef", "coefficient_table", "model_coefficients"):
        rows.extend(_coefficient_rows_from_value(payload.get(key), model_name=""))
    return _dedupe_rows(rows)


def _coefficient_rows_from_value(value: Any, *, model_name: str) -> list[dict[str, Any]]:
    value = maybe_json_loads(value)
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for term, estimate in value.items():
            if isinstance(estimate, dict):
                row = {"model": model_name, "term": _safe_scalar(term)}
                row.update({str(key): _safe_value(child) for key, child in estimate.items()})
            else:
                row = {"model": model_name, "term": _safe_scalar(term), "estimate": _safe_value(estimate)}
            rows.append({key: value for key, value in row.items() if value not in ("", None)})
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                row = {"model": model_name}
                row.update({str(key): _safe_value(child) for key, child in item.items()})
                rows.append({key: value for key, value in row.items() if value not in ("", None)})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                rows.append({"model": model_name, "term": _safe_scalar(item[0]), "estimate": _safe_value(item[1])})
    return rows


def _extract_diagnostics(payload: dict[str, Any], models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    del models
    rows: list[dict[str, Any]] = []
    raw_models = payload.get("models")
    if isinstance(raw_models, list):
        for index, raw_model in enumerate(raw_models, start=1):
            if not isinstance(raw_model, dict):
                continue
            model_name = _safe_scalar(
                raw_model.get("name")
                or raw_model.get("model")
                or raw_model.get("model_name")
                or f"Model {index}"
            )
            row: dict[str, Any] = {"model": model_name}
            for key in _DIAGNOSTIC_KEYS:
                if key in raw_model:
                    value = raw_model.get(key)
                    if isinstance(value, dict):
                        row.update({str(child_key): _safe_value(child) for child_key, child in value.items()})
                    else:
                        row[key] = _safe_value(value)
            metric_dict = raw_model.get("metrics")
            if isinstance(metric_dict, dict):
                for key, value in metric_dict.items():
                    if _looks_diagnostic_key(key):
                        row[str(key)] = _safe_value(value)
            if len(row) > 1:
                rows.append(row)
    for key in _DIAGNOSTIC_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend(_records_from_value(value))
        elif isinstance(value, dict):
            rows.append({str(child_key): _safe_value(child) for child_key, child in value.items()})
    return _dedupe_rows(rows)


def _extract_named_records(payload: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for container in _iter_dicts(payload):
        for key in keys:
            if key in container:
                rows.extend(_records_from_value(container.get(key)))
    return _dedupe_rows(rows)


def _records_from_value(value: Any) -> list[dict[str, Any]]:
    value = maybe_json_loads(value)
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                rows.append({str(key): _safe_value(child) for key, child in item.items() if not _is_heavy_key(key)})
        return rows
    if isinstance(value, dict):
        if any(isinstance(child, (dict, list)) for child in value.values()):
            rows: list[dict[str, Any]] = []
            for key, child in value.items():
                child_rows = _records_from_value(child)
                if child_rows:
                    for row in child_rows:
                        row.setdefault("source", _safe_scalar(key))
                    rows.extend(child_rows)
            return rows
        return [{str(key): _safe_value(child) for key, child in value.items() if not _is_heavy_key(key)}]
    return []


def _iter_dicts(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stack = [value]
    while stack and len(out) < 200:
        current = stack.pop(0)
        if isinstance(current, dict):
            out.append(current)
            for key, child in current.items():
                if _is_heavy_key(key):
                    continue
                if isinstance(child, (dict, list, tuple)):
                    stack.append(child)
        elif isinstance(current, (list, tuple)):
            stack.extend(list(current)[:50])
    return out


def _table_block(rows: list[dict[str, Any]], *, max_rows: int) -> dict[str, Any]:
    limited = _limit_rows(rows, max_rows)
    return {
        "count": len(rows),
        "rows": limited,
        "omitted_count": max(0, len(rows) - len(limited)),
    }


def _limit_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if len(rows) <= max_rows:
        return rows
    head = max_rows // 2
    tail = max_rows - head
    return rows[:head] + rows[-tail:]


def _safe_value(value: Any) -> Any:
    value = maybe_json_loads(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else ""
    if isinstance(value, str):
        return _safe_scalar(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            if _is_heavy_key(key):
                continue
            out[str(key)] = _safe_value(child)
        return out
    if isinstance(value, (list, tuple)):
        if len(value) > 20:
            return {
                "count": len(value),
                "sample": [_safe_value(item) for item in list(value)[:10]],
                "omitted_count": len(value) - 10,
            }
        return [_safe_value(item) for item in value]
    return _safe_scalar(value)


def _safe_scalar(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) <= MAX_CELL_CHARS:
        return text
    return text[:MAX_CELL_CHARS].rstrip() + f"...[truncated {len(text) - MAX_CELL_CHARS} chars]"


def _is_heavy_key(key: Any) -> bool:
    return bool(_ARTIFACT_KEY_RE.search(str(key or "")))


def _looks_diagnostic_key(key: Any) -> bool:
    key_text = str(key or "").lower()
    return any(
        token in key_text
        for token in (
            "aic",
            "bic",
            "log_likelihood",
            "dispersion",
            "pearson",
            "residual",
            "zero_",
            "overdispersion",
        )
    )


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        cleaned = {key: value for key, value in row.items() if value not in ("", None, [], {})}
        if not cleaned:
            continue
        fingerprint = json.dumps(cleaned, sort_keys=True, ensure_ascii=False, default=str)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(cleaned)
    return out
