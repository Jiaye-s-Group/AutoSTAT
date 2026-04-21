from __future__ import annotations

import html
import json
import math
import re
from typing import Any


_METRIC_LABELS = {
    "accuracy": "准确率",
    "precision": "精确率",
    "recall": "召回率",
    "f1": "F1值",
    "f1_score": "F1值",
    "auc": "AUC",
    "roc_auc": "AUC",
    "pr_auc": "PR-AUC",
    "logloss": "LogLoss",
    "loss": "Loss",
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "mape": "MAPE",
    "smape": "SMAPE",
    "r2": "R2",
    "explained_variance": "解释方差",
    "silhouette": "轮廓系数",
    "ari": "ARI",
    "nmi": "NMI",
    "davies_bouldin": "Davies-Bouldin",
    "calinski_harabasz": "Calinski-Harabasz",
}

_AUX_LABELS = {
    "experiment_setting": "实验设置",
    "setting": "实验设置",
    "feature_set": "特征组合",
    "feature_combo": "特征组合",
    "feature_combination": "特征组合",
    "dataset": "数据集",
    "data_split": "数据划分",
    "train_time": "训练时间",
    "training_time": "训练时间",
    "inference_time": "推理时间",
    "parameter_count": "参数量",
    "param_count": "参数量",
    "params": "参数量",
    "parameters": "参数量",
}

_CLASSIFICATION_PRIORITY = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_score",
    "auc",
    "roc_auc",
    "pr_auc",
    "logloss",
    "loss",
]

_REGRESSION_PRIORITY = [
    "mse",
    "rmse",
    "mae",
    "mape",
    "smape",
    "r2",
    "explained_variance",
]

_CLUSTER_PRIORITY = [
    "silhouette",
    "ari",
    "nmi",
    "davies_bouldin",
    "calinski_harabasz",
]

_LOWER_BETTER_KEYWORDS = (
    "loss",
    "error",
    "mae",
    "mse",
    "rmse",
    "mape",
    "smape",
    "logloss",
    "davies_bouldin",
    "aic",
    "bic",
)

_HIGHER_BETTER_KEYWORDS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "r2",
    "explained_variance",
    "silhouette",
    "ari",
    "nmi",
    "calinski_harabasz",
    "ks",
)

_MODEL_NAME_KEYS = ("name", "model", "model_name", "method", "algorithm", "estimator")
_METRIC_CONTAINER_KEYS = ("metrics", "metric", "scores", "score", "evaluation", "eval", "performance", "results")
_AUX_PRIORITY_KEYS = [
    "experiment_setting",
    "setting",
    "feature_set",
    "feature_combo",
    "feature_combination",
    "dataset",
    "data_split",
    "train_time",
    "training_time",
    "inference_time",
    "parameter_count",
    "param_count",
    "params",
    "parameters",
]
_IGNORE_MODEL_KEYS = {
    "artifacts",
    "artifact_warning",
    "best_model",
    "best_model_b64",
    "best_score",
    "best_metric",
    "best_value",
    "coef",
    "coefficients",
    "feature_importance",
    "feature_importances",
    "importance",
    "importance_scores",
    "intermediate",
    "is_best",
    "pred",
    "prediction",
    "predictions",
    "proba",
    "probabilities",
    "rank",
    "residual",
    "residuals",
}


def maybe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return value

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def build_model_comparison_table_bundle(
    result_json: Any,
    *,
    target: str = "",
    user_input: str = "",
    additional_preference: str = "",
) -> dict[str, Any]:
    parsed = maybe_json_loads(result_json)
    if not isinstance(parsed, dict):
        return _empty_bundle()

    models = _extract_models(parsed)
    if not models:
        return _empty_bundle()

    rows: list[dict[str, Any]] = []
    metric_keys: list[str] = []
    aux_keys: list[str] = []
    seen_metric_keys: set[str] = set()
    seen_aux_keys: set[str] = set()

    for index, model in enumerate(models, start=1):
        if not isinstance(model, dict):
            continue

        model_name = _extract_model_name(model, index)
        metrics = _extract_metrics(model)
        aux_fields = _extract_aux_fields(model)

        for key in metrics:
            if key not in seen_metric_keys:
                metric_keys.append(key)
                seen_metric_keys.add(key)

        for key in aux_fields:
            if key not in seen_aux_keys:
                aux_keys.append(key)
                seen_aux_keys.add(key)

        rows.append(
            {
                "model_name": model_name,
                "metrics": metrics,
                "aux_fields": aux_fields,
            }
        )

    if not rows:
        return _empty_bundle()

    task_type = _infer_task_type(rows, parsed)
    metric_keys = _sort_metric_keys(metric_keys, task_type)
    aux_keys = _sort_aux_keys(aux_keys)
    best_marks = _build_best_marks(rows, metric_keys)

    column_keys = ["model_name", *aux_keys, *metric_keys]
    column_labels = [
        "方法/模型",
        *[_label_for_aux(key) for key in aux_keys],
        *[_label_for_metric(key) for key in metric_keys],
    ]

    rendered_rows: list[list[str]] = []
    for row_index, row in enumerate(rows):
        rendered = [_format_cell_value(row["model_name"])]
        rendered.extend(_format_column_value(row["aux_fields"].get(key)) for key in aux_keys)
        rendered.extend(
            _format_metric_value(row["metrics"].get(key), mark_best=best_marks.get((row_index, key), False))
            for key in metric_keys
        )
        rendered_rows.append(rendered)

    title = _build_table_title(
        task_type=task_type,
        target=target,
        user_input=user_input,
        additional_preference=additional_preference,
    )

    markdown_table = _build_markdown_table(column_labels, rendered_rows)
    html_table = _build_html_table(column_labels, rendered_rows)
    best_model_text = _extract_best_model_text(parsed)

    return {
        "has_table": bool(markdown_table),
        "title": title,
        "caption": f"表1 {title}" if title else "",
        "task_type": task_type,
        "column_keys": column_keys,
        "column_labels": column_labels,
        "rows": rendered_rows,
        "markdown_table": markdown_table,
        "html_table": html_table,
        "best_model_text": best_model_text,
    }


def append_model_comparison_table(markdown_text: str, bundle: dict[str, Any]) -> str:
    text = _strip_existing_model_comparison_table((markdown_text or "").strip())
    if not bundle.get("has_table"):
        return text

    markdown_table = str(bundle.get("markdown_table") or "").strip()
    if not markdown_table:
        return text

    if "内容汇总表格" in text and markdown_table in text:
        return text

    appendix = f"内容汇总表格\n\n{markdown_table}"
    if not text:
        return appendix
    return f"{text}\n\n{appendix}".strip()


def build_modeling_execution_summary_markdown(result_json: Any, bundle: dict[str, Any]) -> str:
    parts: list[str] = []
    best_model_text = str(bundle.get("best_model_text") or "").strip()
    if best_model_text:
        parts.append(best_model_text)

    detail_block = _build_model_detail_markdown(bundle)
    if detail_block:
        parts.append(detail_block)

    if parts:
        return "\n\n".join(part for part in parts if part).strip()

    parsed = maybe_json_loads(result_json)
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    return str(result_json or "").strip()


def _empty_bundle() -> dict[str, Any]:
    return {
        "has_table": False,
        "title": "",
        "caption": "",
        "task_type": "generic",
        "column_keys": [],
        "column_labels": [],
        "rows": [],
        "markdown_table": "",
        "html_table": "",
        "best_model_text": "",
    }


def _extract_models(result_dict: dict[str, Any]) -> list[dict[str, Any]]:
    models_value = result_dict.get("models")

    if isinstance(models_value, list):
        return [item for item in models_value if isinstance(item, dict)]

    if isinstance(models_value, dict):
        out: list[dict[str, Any]] = []
        for model_name, metrics in models_value.items():
            if isinstance(metrics, dict):
                item = dict(metrics)
                item.setdefault("name", str(model_name))
                out.append(item)
        return out

    best_model = result_dict.get("best_model")
    if isinstance(best_model, dict):
        return [best_model]

    return []


def _extract_model_name(model_dict: dict[str, Any], index: int) -> str:
    for key in _MODEL_NAME_KEYS:
        value = model_dict.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"模型{index}"


def _extract_metrics(model_dict: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    for key in _METRIC_CONTAINER_KEYS:
        value = model_dict.get(key)
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                if _is_scalar(inner_value):
                    metrics[_normalize_key(inner_key)] = inner_value

    for key, value in model_dict.items():
        normalized_key = _normalize_key(key)
        if normalized_key in _IGNORE_MODEL_KEYS or normalized_key in {_normalize_key(name_key) for name_key in _MODEL_NAME_KEYS}:
            continue
        if normalized_key in {_normalize_key(metric_key) for metric_key in _METRIC_CONTAINER_KEYS}:
            continue
        if normalized_key in {_normalize_key(aux_key) for aux_key in _AUX_PRIORITY_KEYS}:
            continue
        if not _is_numeric(value):
            continue
        metrics.setdefault(normalized_key, value)

    return metrics


def _extract_aux_fields(model_dict: dict[str, Any]) -> dict[str, Any]:
    aux_fields: dict[str, Any] = {}

    for key, value in model_dict.items():
        normalized_key = _normalize_key(key)
        if normalized_key in _IGNORE_MODEL_KEYS:
            continue
        if normalized_key in {_normalize_key(name_key) for name_key in _MODEL_NAME_KEYS}:
            continue
        if normalized_key in {_normalize_key(metric_key) for metric_key in _METRIC_CONTAINER_KEYS}:
            continue
        if isinstance(value, dict) or isinstance(value, list):
            continue
        if _looks_like_metric_key(normalized_key) and _is_numeric(value):
            continue
        if _is_scalar(value):
            aux_fields[normalized_key] = value

    return aux_fields


def _infer_task_type(rows: list[dict[str, Any]], result_dict: dict[str, Any]) -> str:
    metric_keys = {metric_key for row in rows for metric_key in row.get("metrics", {})}
    model_names = " ".join(str(row.get("model_name", "")) for row in rows).lower()
    best_model = result_dict.get("best_model")
    best_metric_text = ""
    if isinstance(best_model, dict):
        best_metric_text = " ".join(str(best_model.get(key, "")) for key in ("metric", "score_name", "value_name")).lower()

    joined_metric_keys = " ".join(metric_keys) + " " + model_names + " " + best_metric_text

    if any(keyword in joined_metric_keys for keyword in ("accuracy", "precision", "recall", "f1", "auc", "classifier", "roc")):
        return "classification"
    if any(keyword in joined_metric_keys for keyword in ("mse", "rmse", "mae", "mape", "regressor", "regression", "r2")):
        return "regression"
    if any(keyword in joined_metric_keys for keyword in ("silhouette", "ari", "nmi", "cluster", "clustering", "davies_bouldin")):
        return "clustering"
    return "generic"


def _sort_metric_keys(metric_keys: list[str], task_type: str) -> list[str]:
    if task_type == "classification":
        priority = _CLASSIFICATION_PRIORITY
    elif task_type == "regression":
        priority = _REGRESSION_PRIORITY
    elif task_type == "clustering":
        priority = _CLUSTER_PRIORITY
    else:
        priority = []

    remaining = [key for key in metric_keys if key not in priority]
    ordered = [key for key in priority if key in metric_keys]
    ordered.extend(sorted(remaining))
    return ordered


def _sort_aux_keys(aux_keys: list[str]) -> list[str]:
    ordered = [key for key in _AUX_PRIORITY_KEYS if key in aux_keys]
    ordered.extend(sorted(key for key in aux_keys if key not in ordered))
    return ordered


def _build_best_marks(rows: list[dict[str, Any]], metric_keys: list[str]) -> dict[tuple[int, str], bool]:
    marks: dict[tuple[int, str], bool] = {}

    for metric_key in metric_keys:
        direction = _metric_direction(metric_key)
        if direction is None:
            continue

        numeric_values: list[tuple[int, float]] = []
        for row_index, row in enumerate(rows):
            value = row.get("metrics", {}).get(metric_key)
            if _is_numeric(value):
                numeric_values.append((row_index, float(value)))

        if len(numeric_values) < 2:
            continue

        best_value = max(value for _, value in numeric_values) if direction == "higher" else min(value for _, value in numeric_values)
        for row_index, value in numeric_values:
            if math.isclose(value, best_value, rel_tol=1e-9, abs_tol=1e-12):
                marks[(row_index, metric_key)] = True

    return marks


def _metric_direction(metric_key: str) -> str | None:
    lowered = _normalize_key(metric_key)
    if any(keyword in lowered for keyword in _LOWER_BETTER_KEYWORDS):
        return "lower"
    if any(keyword in lowered for keyword in _HIGHER_BETTER_KEYWORDS):
        return "higher"
    return None


def _build_table_title(
    *,
    task_type: str,
    target: str,
    user_input: str,
    additional_preference: str,
) -> str:
    target_text = _clean_context_text(target)
    _ = user_input, additional_preference

    if task_type == "classification":
        return "不同模型在分类任务上的性能比较"
    if task_type == "regression":
        if target_text:
            return f"不同模型在{target_text}预测任务上的性能比较"
        return "不同模型在回归任务上的性能比较"
    if task_type == "clustering":
        return "不同模型在聚类任务上的性能比较"
    return "不同模型在建模任务上的结果比较"


def _extract_best_model_text(result_dict: dict[str, Any]) -> str:
    best_model = result_dict.get("best_model")
    if isinstance(best_model, str) and best_model.strip():
        return f"最佳模型：{best_model.strip()}。"

    if not isinstance(best_model, dict):
        return ""

    best_name = ""
    for key in ("name", "model", "model_name", "method"):
        value = best_model.get(key)
        if isinstance(value, str) and value.strip():
            best_name = value.strip()
            break

    metric_key = ""
    for key in ("metric", "score_name", "value_name"):
        value = best_model.get(key)
        if isinstance(value, str) and value.strip():
            metric_key = _normalize_key(value)
            break

    metric_value = None
    for key in ("value", "score", "metric_value", "best_score"):
        value = best_model.get(key)
        if value is not None:
            metric_value = value
            break

    if best_name and metric_key and metric_value is not None:
        return f"最佳模型：{best_name}，{_label_for_metric(metric_key)}为{_format_plain_value(metric_value)}。"
    if best_name:
        return f"最佳模型：{best_name}。"
    return ""


def _label_for_metric(metric_key: str) -> str:
    normalized_key = _normalize_key(metric_key)
    return _METRIC_LABELS.get(normalized_key, str(metric_key).strip() or normalized_key)


def _label_for_aux(aux_key: str) -> str:
    normalized_key = _normalize_key(aux_key)
    return _AUX_LABELS.get(normalized_key, str(aux_key).strip() or normalized_key)


def _build_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers or not rows:
        return ""

    header_line = "| " + " | ".join(_escape_markdown_cell(cell) for cell in headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines]).strip()


def _build_html_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers or not rows:
        return ""

    head_html = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
    row_html = []
    for row in rows:
        row_html.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>")

    return (
        '<table class="report-model-comparison-table">'
        "<thead><tr>"
        f"{head_html}"
        "</tr></thead>"
        "<tbody>"
        f"{''.join(row_html)}"
        "</tbody></table>"
    )


def _build_model_detail_markdown(bundle: dict[str, Any]) -> str:
    column_keys = bundle.get("column_keys")
    column_labels = bundle.get("column_labels")
    rows = bundle.get("rows")
    task_type = str(bundle.get("task_type") or "").strip()

    if not isinstance(column_keys, list) or not isinstance(column_labels, list) or not isinstance(rows, list):
        return ""
    if len(column_keys) != len(column_labels) or not rows:
        return ""

    metric_indexes = [
        index for index, key in enumerate(column_keys)
        if index > 0 and (_looks_like_metric_key(str(key)) or _normalize_key(key) in _METRIC_LABELS)
    ]
    aux_indexes = [index for index in range(1, len(column_keys)) if index not in metric_indexes]

    task_type_text = {
        "classification": "分类",
        "regression": "回归",
        "clustering": "聚类",
    }.get(task_type, "建模")

    lines = ["模型展示细节"]
    for row in rows:
        if not isinstance(row, list) or not row:
            continue

        model_name = str(row[0]).strip()
        if not model_name:
            continue

        lines.append(f"- 模型名称：{model_name}")
        lines.append(f"  - 模型类型：{task_type_text}")

        for index in aux_indexes:
            if index >= len(row):
                continue
            value = str(row[index]).strip()
            if not value or value == "—":
                continue
            lines.append(f"  - {column_labels[index]}：{value}")

        metric_lines: list[str] = []
        for index in metric_indexes:
            if index >= len(row):
                continue
            value = str(row[index]).strip()
            if not value or value == "—":
                continue
            metric_lines.append(f"    - {column_labels[index]}：{value}")

        if metric_lines:
            lines.append("  - 主要性能指标：")
            lines.extend(metric_lines)

    return "\n".join(lines).strip()


def _format_metric_value(value: Any, *, mark_best: bool = False) -> str:
    text = _format_plain_value(value)
    if text == "—":
        return text
    if mark_best:
        return f"{text}（最优）"
    return text


def _format_column_value(value: Any) -> str:
    return _format_plain_value(value)


def _format_cell_value(value: Any) -> str:
    return _format_plain_value(value)


def _format_plain_value(value: Any) -> str:
    if value is None:
        return "—"

    if isinstance(value, bool):
        return "是" if value else "否"

    if _is_numeric(value):
        numeric = float(value)
        if not math.isfinite(numeric):
            return "—"
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return f"{numeric:.4f}"

    text = str(value).strip()
    return text or "—"


def _escape_markdown_cell(value: str) -> str:
    return str(value or "").replace("|", "\\|")


def _strip_existing_model_comparison_table(text: str) -> str:
    if not text:
        return ""

    lines = text.replace("\r\n", "\n").split("\n")
    kept_lines: list[str] = []
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith("|"):
            if kept_lines:
                while kept_lines and not kept_lines[-1].strip():
                    kept_lines.pop()
                prev = kept_lines[-1].strip() if kept_lines else ""
                if prev and len(prev) <= 30:
                    kept_lines.pop()

            while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith("|")):
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue

        kept_lines.append(lines[i])
        i += 1

    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

def _normalize_key(key: Any) -> str:
    text = str(key or "").strip().lower()
    text = text.replace("%", "pct")
    text = re.sub(r"[\s\-/]+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text.strip("_")


def _looks_like_metric_key(metric_key: str) -> bool:
    lowered = _normalize_key(metric_key)
    metric_tokens = set(_CLASSIFICATION_PRIORITY + _REGRESSION_PRIORITY + _CLUSTER_PRIORITY)
    if lowered in metric_tokens:
        return True
    return any(token in lowered for token in ("acc", "auc", "precision", "recall", "f1", "mse", "rmse", "mae", "mape", "r2", "loss", "error"))


def _clean_context_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    if len(cleaned) > 20:
        return ""
    if any(token in cleaned for token in (",", "，", ";", "；", "\n")):
        return ""
    return cleaned


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
