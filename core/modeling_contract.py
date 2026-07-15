from __future__ import annotations

import json
import math
import re
from typing import Any


INFERENCE_MARKERS = (
    "关联",
    "推断",
    "回归系数",
    "置信区间",
    "稳健标准误",
    "hc3",
    "complete-case",
    "complete case",
    "嵌套模型",
    "association",
    "inference",
    "confidence interval",
    "robust standard error",
    "nested model",
)
PREDICTION_MARKERS = (
    "预测",
    "分类",
    "准确率",
    "训练集",
    "测试集",
    "prediction",
    "classification",
    "accuracy",
    "train/test",
    "test set",
    "linear regression",
    "random forest",
    "xgboost",
    "neural network",
    "线性回归",
    "随机森林",
    "神经网络",
)
UNSUPERVISED_MARKERS = ("聚类", "降维", "pca", "clustering", "unsupervised")


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def _resolve_outcome(target: str, columns: list[Any], text: str) -> str:
    column_names = [str(column) for column in columns]
    target_text = str(target or "").strip()
    if target_text in column_names:
        return target_text
    lowered = text.lower()
    marker_matches = list(re.finditer(r"结局|目标|outcome|target", lowered, re.I))
    nearby_candidates: list[tuple[int, int, str]] = []
    mentioned: list[str] = []
    for index, column in enumerate(column_names):
        escaped = re.escape(column.lower())
        column_pattern = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
        column_matches = list(re.finditer(column_pattern, lowered, re.I))
        if not column_matches:
            continue
        mentioned.append(column)
        for column_match in column_matches:
            for marker_match in marker_matches:
                gap = max(
                    marker_match.start() - column_match.end(),
                    column_match.start() - marker_match.end(),
                    0,
                )
                if gap <= 40:
                    nearby_candidates.append((gap, index, column))

    if nearby_candidates:
        return min(nearby_candidates)[2]
    return mentioned[0] if len(mentioned) == 1 else target_text


def build_analysis_contract(
    *,
    target: str,
    columns: list[Any],
    user_input: str = "",
    add_preference: str = "",
    refined_suggestions: str = "",
    task_type: str = "auto",
) -> dict[str, Any]:
    text = "\n".join(
        part for part in (str(user_input or ""), str(add_preference or ""), str(refined_suggestions or "")) if part
    )
    column_names = [str(column) for column in columns]
    outcome = _resolve_outcome(target, column_names, text)

    requested_task_type = str(task_type or "auto").strip().lower()
    explicit_task_types = {"association_inference", "prediction", "unsupervised"}

    if requested_task_type in explicit_task_types:
        task_type = requested_task_type
    elif _contains_any(text, UNSUPERVISED_MARKERS):
        task_type = "unsupervised"
    elif _contains_any(text, INFERENCE_MARKERS):
        task_type = "association_inference"
    elif _contains_any(text, PREDICTION_MARKERS) or outcome:
        task_type = "prediction"
    else:
        task_type = "unspecified"

    inferential = task_type == "association_inference"
    requires_outcome = task_type in {"association_inference", "prediction"}
    no_split_requested = bool(
        inferential
        or re.search(r"(?:不|无需|不要).{0,8}(?:训练.?测试|划分|split)", text, re.I)
        or re.search(r"no\s+(?:train.?test\s+)?split", text, re.I)
    )
    split_strategy = "none" if no_split_requested else "random_80_20"

    covariance = "HC3" if re.search(r"\bHC3\b", text, re.I) else ("model_default" if inferential else "not_applicable")
    if re.search(r"共同.{0,12}(?:完整案例|complete.case)|common.{0,8}complete.case", text, re.I):
        sample_rule = "common_complete_case"
    elif re.search(r"complete.case", text, re.I):
        sample_rule = "complete_case"
    else:
        sample_rule = "model_specific"

    prohibited_operations: list[str] = []
    if split_strategy == "none":
        prohibited_operations.append("train_test_split")
    if re.search(r"(?:不|禁止|不要).{0,8}(?:逐步|stepwise)", text, re.I):
        prohibited_operations.append("stepwise_selection")
    if inferential:
        prohibited_operations.extend(["best_model_selection", "mandatory_model_export"])

    issues: list[str] = []
    if requires_outcome and outcome not in column_names:
        if str(outcome or "").strip():
            issues.append(f"Outcome column {outcome!r} is not present in the dataset columns.")
        else:
            issues.append("The selected task type requires an outcome chosen from the dataset columns.")

    return {
        "version": 1,
        "task_type": task_type,
        "outcome": outcome,
        "requires_outcome": requires_outcome,
        "sample_rule": sample_rule,
        "split_strategy": split_strategy,
        "covariance": covariance,
        "model_structure": "nested" if re.search(r"嵌套|nested", text, re.I) else "as_requested",
        "required_result_key": "analysis_manifest",
        "prohibited_operations": sorted(set(prohibited_operations)),
        "valid": not issues,
        "issues": issues,
    }


def contract_as_prompt(contract: dict[str, Any]) -> str:
    return json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True)


def validate_result_against_contract(
    *,
    code: str,
    result_json: Any,
    contract: dict[str, Any],
) -> list[str]:
    issues = list(contract.get("issues") or [])
    if not isinstance(result_json, dict) or not result_json:
        issues.append("result_dict is empty or is not a JSON object.")
        return issues

    manifest = result_json.get(str(contract.get("required_result_key") or "analysis_manifest"))
    if not isinstance(manifest, dict):
        issues.append("result_dict.analysis_manifest is required for method verification.")
        return issues

    expected_pairs = {
        "task_type": contract.get("task_type"),
        "outcome": contract.get("outcome"),
        "sample_rule": contract.get("sample_rule"),
        "split_strategy": contract.get("split_strategy"),
        "covariance": contract.get("covariance"),
        "model_structure": contract.get("model_structure"),
    }
    for key, expected in expected_pairs.items():
        if expected in (None, "", "not_applicable"):
            continue
        actual = manifest.get(key)
        if str(actual or "").strip().lower() != str(expected).strip().lower():
            issues.append(f"analysis_manifest.{key} must be {expected!r}, got {actual!r}.")

    code_lower = str(code or "").lower()
    for prohibited in contract.get("prohibited_operations") or []:
        if prohibited == "train_test_split" and "train_test_split" in code_lower:
            issues.append("train_test_split is prohibited by the analysis contract.")
        elif prohibited == "stepwise_selection" and re.search(r"stepwise|sequentialfeatureselector|rfe\s*\(", code_lower):
            issues.append("Stepwise/automatic feature selection is prohibited by the analysis contract.")

    if str(contract.get("covariance") or "").upper() == "HC3" and "hc3" not in code_lower:
        issues.append("The analysis contract requires HC3 robust covariance, but HC3 is absent from the code.")
    return issues


def validate_result_schema(
    *,
    result_json: Any,
    contract: dict[str, Any],
) -> list[str]:
    """Validate the evidence fields required by each analysis task type."""
    if not isinstance(result_json, dict):
        return ["result_dict must be a JSON object."]

    models = result_json.get("models")
    if not isinstance(models, list) or not models:
        return ["result_dict.models must be a non-empty list of executed analyses."]

    issues: list[str] = []
    task_type = str(contract.get("task_type") or "").strip().lower()
    for index, model in enumerate(models, start=1):
        path = f"result_dict.models[{index - 1}]"
        if not isinstance(model, dict):
            issues.append(f"{path} must be a JSON object.")
            continue
        if not str(model.get("name") or "").strip():
            issues.append(f"{path}.name is required.")
        n_obs = model.get("n_obs")
        if not _is_finite_number(n_obs) or float(n_obs) <= 0:
            issues.append(f"{path}.n_obs must be a positive number.")

        metrics = model.get("metrics")
        if not isinstance(metrics, dict) or not any(
            _is_finite_number(value) for value in metrics.values()
        ):
            issues.append(f"{path}.metrics must contain at least one finite numeric value.")

        if task_type == "association_inference":
            coefficients = model.get("coefficients")
            if not isinstance(coefficients, list) or not coefficients:
                issues.append(f"{path}.coefficients must be a non-empty list for inference tasks.")
                continue
            for coefficient_index, coefficient in enumerate(coefficients):
                coefficient_path = f"{path}.coefficients[{coefficient_index}]"
                if not isinstance(coefficient, dict):
                    issues.append(f"{coefficient_path} must be a JSON object.")
                    continue
                if not str(coefficient.get("term") or "").strip():
                    issues.append(f"{coefficient_path}.term is required.")
                for key in ("estimate", "std_error", "ci_lower", "ci_upper", "p_value"):
                    if not _is_finite_number(coefficient.get(key)):
                        issues.append(f"{coefficient_path}.{key} must be a finite number.")

    return issues


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def format_contract_violations(issues: list[str]) -> str:
    return "Analysis contract validation failed:\n- " + "\n- ".join(issues)
