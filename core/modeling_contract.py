from __future__ import annotations

import ast
import json
import math
import re
from typing import Any

from core.modeling_report_artifacts import has_candidate_report_outputs


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
MAX_EMBEDDED_RESULT_ROWS = 1_000

_NON_INFERENTIAL_MODEL_FAMILIES = {
    "random_forest",
    "xgboost",
    "lightgbm",
    "kmeans",
}
_PLACEHOLDER_TERMS = {
    "placeholder",
    "dummy",
    "fake",
    "stub",
    "status_code",
}

_MODEL_FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "zero_inflated_negative_binomial",
        (
            "zeroinflatednegativebinomial",
            "zero_inflated_negative_binomial",
            "zero-inflated negative binomial",
            "zero inflated negative binomial",
            "zero-inflated nb",
            "zero inflated nb",
            "zinb",
            "零膨胀负二项",
            "零膨胀模型",
        ),
    ),
    ("negative_binomial", ("negativebinomial", "negative binomial", "负二项")),
    ("poisson", ("poisson", "泊松")),
    ("logistic_regression", ("logistic regression", "逻辑回归")),
    ("linear_regression", ("linear regression", "线性回归", "ols")),
    ("random_forest", ("random forest", "随机森林")),
    ("xgboost", ("xgboost",)),
    ("lightgbm", ("lightgbm",)),
    ("kmeans", ("k-means", "kmeans", "k均值")),
)

_NON_REQUIRED_MODEL_CONTEXT_PATTERNS = (
    r"allowed[_\s-]*model[_\s-]*famil(?:y|ies)",
    r"allowed\s+famil(?:y|ies)",
    r"prohibited[_\s-]*operations?",
    r"model[_\s-]*constraints?",
    r"required[_\s-]*model[_\s-]*specs?",
    r"prohibited[_\s-]*methods?",
    r"forbidden[_\s-]*methods?",
    r"候选模型家族",
    r"允许(?:的)?模型",
    r"禁止(?:的)?操作",
    r"模型约束",
    r"可选",
    r"未来",
    r"扩展",
    r"敏感性",
    r"备选",
    r"候选",
    r"仅作为(?:说明|参考|补充)",
    r"只作为(?:说明|参考|补充)",
    r"\boptional\b",
    r"\bfuture\b",
    r"\bextension\b",
    r"\bsensitivity\b",
    r"\bcandidate\b",
    r"\ballowed\b",
    r"\bpermitted\b",
    r"\bmay\s+(?:be\s+)?(?:use|fit|run|consider)",
    r"\bcan\s+(?:be\s+)?(?:use|fit|run|consider)",
    r"\bif\s+(?:using|adopting|chosen|selected)\b",
    r"(?:如果|若|如若|假如).{0,20}(?:使用|采用|选择|执行|拟合)",
)

_HARD_NEGATION_PATTERN = (
    r"(?:不|不要|无需|不需|不得|不应|不能|禁止|排除|避免|不生成|不执行|不拟合|不训练|不作为|"
    r"\bdo\s+not\b|\bdon't\b|\bwithout\b|\bexclude\b|\bavoid\b|\bforbid\b|\bprohibit\b|"
    r"\bmust\s+not\b|\bshould\s+not\b|\bno\b)"
)

_REQUIRED_MODEL_INTENT_PATTERN = (
    r"(?:必须|明确|执行|拟合|训练|构建|建立|采用|使用|保留|"
    r"\bmust\b|\brequired\b|\brequire\b|\bfit\b|\brun\b|\bexecute\b|\btrain\b|\bbuild\b|\buse\b)"
)

_REFINEMENT_BLOCKS_FALLBACK_PATTERNS = (
    r"(?:不|不要|无需|不需|不得|不应|不能|禁止|排除|避免|不生成|不执行|不拟合|不训练)",
    r"(?:只|仅).{0,12}(?:生成|执行|拟合|保留|使用|一个|1\s*个|one)",
    r"(?:覆盖|替换|清空|最终|以此为准)",
    r"\b(?:do\s+not|don't|without|exclude|avoid|forbid|prohibit|must\s+not|should\s+not|no)\b",
    r"\b(?:only|replace|override|final)\b",
)

_MODEL_COUNT_LIMIT_PATTERNS = (
    r"(?:最多|至多|不超过|不多于|<=|≤|只|仅).{0,16}(?P<count>\d+|一|二|两|三)\s*个?.{0,16}(?:模型|辅助模型)",
    r"(?:models|result_dict\.models).{0,24}(?:最多|至多|不超过|不多于|<=|≤).{0,8}(?P<count>\d+|一|二|两|三)",
    r"\b(?:at\s+most|no\s+more\s+than|only)\s+(?P<count>\d+|one|two|three)\s+(?:auxiliary\s+)?models?\b",
    r"\b(?P<count>one|single)\s+auxiliary\s+model\b",
    r"\bdo\s+not\s+generate\s+multiple\s+models\b",
)

_PRIMARY_RESULT_CANDIDATE_PATTERNS = (
    r"candidate[_\s-]*(?:results?|segments?|tables?)",
    r"候选(?:区段|窗口|结果|表)",
    r"主结果.{0,24}(?:candidate|候选)",
)


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


def _family_from_text(text: str) -> str:
    families = _families_from_text(text)
    return families[0] if families else ""


def _families_from_text(text: str) -> list[str]:
    lowered = str(text or "").lower()
    compact = re.sub(r"[\s_\-]", "", lowered)
    families: list[str] = []
    for family, patterns in _MODEL_FAMILY_PATTERNS:
        pattern_matches = any(
            pattern.lower() in lowered
            or re.sub(r"[\s_\-]", "", pattern.lower()) in compact
            for pattern in patterns
        )
        if pattern_matches:
            families.append(family)
    if (
        "zero_inflated_negative_binomial" in families
        and "negative_binomial" in families
        and not _has_standalone_negative_binomial_mention(lowered)
    ):
        families = [family for family in families if family != "negative_binomial"]
    return families


def _has_standalone_negative_binomial_mention(text: str) -> bool:
    value = str(text or "")
    for match in re.finditer(r"(?:negative[\s_-]*binomial|负二项)", value, re.I):
        prefix = value[max(0, match.start() - 32): match.start()]
        if re.search(r"(?:zero[\s_-]*inflated|零膨胀)\s*$", prefix, re.I):
            continue
        return True
    return False


def _line_is_non_required_model_context(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return True
    return bool(
        any(re.search(pattern, text, re.I) for pattern in _NON_REQUIRED_MODEL_CONTEXT_PATTERNS)
        or re.search(_HARD_NEGATION_PATTERN, text, re.I)
    )


def _refinement_blocks_model_suggestion_fallback(refined_suggestions: str) -> bool:
    text = str(refined_suggestions or "").strip()
    if not text:
        return False
    return any(re.search(pattern, text, re.I) for pattern in _REFINEMENT_BLOCKS_FALLBACK_PATTERNS) or any(
        re.search(pattern, text, re.I) for pattern in _NON_REQUIRED_MODEL_CONTEXT_PATTERNS
    )


_NUMBER_WORDS = {
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "one": 1,
    "single": 1,
    "two": 2,
    "three": 3,
}


def _parse_count_token(value: Any) -> int | None:
    token = str(value or "").strip().lower()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _requested_max_models(text: str) -> int | None:
    joined = str(text or "")
    counts: list[int] = []
    for pattern in _MODEL_COUNT_LIMIT_PATTERNS:
        for match in re.finditer(pattern, joined, re.I):
            if "multiple models" in match.group(0).lower():
                counts.append(1)
                continue
            count = _parse_count_token(match.groupdict().get("count"))
            if count is not None:
                counts.append(count)
    return min(counts) if counts else None


def _primary_result_type_from_text(text: str, task_type: str) -> str:
    if any(re.search(pattern, str(text or ""), re.I) for pattern in _PRIMARY_RESULT_CANDIDATE_PATTERNS):
        return "candidate_results"
    _ = task_type
    return "models"


def _prohibited_model_families(text: str) -> list[str]:
    prohibited: set[str] = set()
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or not re.search(_HARD_NEGATION_PATTERN, line, re.I):
            continue
        prohibited.update(_families_from_text(line))
    return sorted(prohibited)


def _extract_hard_constraints(text: str, task_type: str) -> dict[str, Any]:
    max_models = _requested_max_models(text)
    prohibited_model_families = _prohibited_model_families(text)
    primary_result_type = _primary_result_type_from_text(text, task_type)
    return {
        "primary_result_type": primary_result_type,
        "max_models": max_models,
        "prohibited_model_families": prohibited_model_families,
    }


def _renumber_model_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, spec in enumerate(specs, start=1):
        spec["id"] = f"model_{index:02d}"
    return specs


def _hard_constraints_block_model_suggestion_fallback(hard_constraints: dict[str, Any]) -> bool:
    if hard_constraints.get("max_models") is not None:
        return True
    if hard_constraints.get("prohibited_model_families"):
        return True
    if hard_constraints.get("primary_result_type") == "candidate_results":
        return True
    return False


def _source_mentions_multiple_required_models(text: str) -> bool:
    """Return true only for explicit multi-model execution/comparison wording."""
    value = str(text or "")
    return bool(
        re.search(r"(?:方案|模型)\s*[1一]\s*[/、,，和与&+]\s*(?:方案|模型)?\s*[2二]", value, re.I)
        or re.search(r"\b(?:run|fit|execute)\s+both\b", value, re.I)
        or re.search(r"\bcompare\b.{0,60}\b(?:and|vs\.?|versus)\b", value, re.I)
        or re.search(r"(?:比较|对比).{0,60}(?:和|与|及|/)", value, re.I)
        or re.search(r"(?:必须执行|都要执行|同时执行).{0,60}(?:和|与|及|/)", value, re.I)
        or re.search(r"\brequired\s+models?\s*:", value, re.I)
    )


def _required_model_specs(
    instruction_text: str,
    *,
    prohibited_model_families: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract explicit model-family requirements from final instructions.

    Repeated mentions of the same model family usually describe one model's
    outcome, features, diagnostics, and reporting fields.  They must not become
    separate model_01/model_02/... requirements unless the text explicitly asks
    for distinct models.
    """
    specs_by_family: dict[str, dict[str, Any]] = {}
    family_order: list[str] = []
    prohibited_model_families = prohibited_model_families or set()
    allow_same_family_duplicates = _source_mentions_multiple_required_models(instruction_text)
    for raw_line in str(instruction_text or "").splitlines():
        line = raw_line.strip(" -•\t")
        if _line_is_non_required_model_context(line):
            continue
        if not re.search(_REQUIRED_MODEL_INTENT_PATTERN, line, re.I):
            continue
        families = [
            family
            for family in _families_from_text(line)
            if family not in prohibited_model_families
        ]
        if not families:
            continue
        lowered = line.lower()
        requires_offset = bool(re.search(r"(?:offset|exposure|暴露(?:量)?|偏置)", lowered, re.I))
        for family in families:
            requires_zero_inflation = family == "zero_inflated_negative_binomial" or bool(
                re.search(r"(?:zero[\s_-]*inflated|零膨胀)", lowered, re.I)
            )

            key = family
            if allow_same_family_duplicates and re.search(r"(?:方案|模型)\s*\d+|model\s*\d+", line, re.I):
                key = f"{family}:{len(specs_by_family) + 1}"

            existing = specs_by_family.get(key)
            if existing is None:
                specs_by_family[key] = {
                    "id": "",
                    "family": family,
                    "requires_offset": requires_offset,
                    "requires_zero_inflation": requires_zero_inflation,
                    "source": line,
                    "_source_lines": [line],
                }
                family_order.append(key)
                continue

            existing["requires_offset"] = bool(existing.get("requires_offset")) or requires_offset
            existing["requires_zero_inflation"] = bool(existing.get("requires_zero_inflation")) or requires_zero_inflation
            source_lines = existing.setdefault("_source_lines", [])
            if line not in source_lines:
                source_lines.append(line)
                existing["source"] = " | ".join(source_lines)

    specs = [specs_by_family[key] for key in family_order]
    for index, item in enumerate(specs, start=1):
        item["id"] = f"model_{index:02d}"
        item.pop("_source_lines", None)
    return specs


def _model_spec_key(spec: dict[str, Any]) -> tuple[str, bool, bool]:
    return (
        str(spec.get("family") or ""),
        bool(spec.get("requires_offset")),
        bool(spec.get("requires_zero_inflation")),
    )


def _merge_required_model_specs(
    *,
    final_instruction_text: str,
    model_suggestion: str,
    hard_constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep final user-facing requirements first, then cautiously add old suggestions.

    A revision that contains negation, "only", optional/future/allowed-family
    wording is treated as an override.  In that case, old suggestions must not be
    merged back into required model specs.
    """
    prohibited = set(hard_constraints.get("prohibited_model_families") or [])
    specs = _required_model_specs(
        final_instruction_text,
        prohibited_model_families=prohibited,
    )
    if (
        _refinement_blocks_model_suggestion_fallback(final_instruction_text)
        or _hard_constraints_block_model_suggestion_fallback(hard_constraints)
    ):
        return _renumber_model_specs(specs)

    seen = {_model_spec_key(spec) for spec in specs}
    for fallback_spec in _required_model_specs(
        model_suggestion,
        prohibited_model_families=prohibited,
    ):
        key = _model_spec_key(fallback_spec)
        if key in seen:
            continue
        specs.append(dict(fallback_spec))
        seen.add(key)

    return _renumber_model_specs(specs)


def build_analysis_contract(
    *,
    target: str,
    columns: list[Any],
    user_input: str = "",
    add_preference: str = "",
    refined_suggestions: str = "",
    model_suggestion: str = "",
    task_type: str = "auto",
) -> dict[str, Any]:
    text = "\n".join(
        part
        for part in (
            str(user_input or ""),
            str(add_preference or ""),
            str(model_suggestion or ""),
            str(refined_suggestions or ""),
        )
        if part
    )
    column_names = [str(column) for column in columns]
    outcome = _resolve_outcome(target, column_names, text)

    requested_task_type = str(task_type or "auto").strip().lower()
    explicit_task_types = {"association_inference", "prediction", "unsupervised"}

    if requested_task_type in explicit_task_types:
        task_type = requested_task_type
    elif _contains_any(text, INFERENCE_MARKERS):
        task_type = "association_inference"
    elif _contains_any(text, UNSUPERVISED_MARKERS):
        task_type = "unsupervised"
    elif _contains_any(text, PREDICTION_MARKERS) or outcome:
        task_type = "prediction"
    else:
        task_type = "unspecified"

    final_instruction_text = "\n".join(
        part
        for part in (
            str(user_input or ""),
            str(add_preference or ""),
            str(refined_suggestions or ""),
        )
        if part
    )
    hard_constraints = _extract_hard_constraints(final_instruction_text or text, task_type)
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

    required_model_specs = _merge_required_model_specs(
        final_instruction_text=final_instruction_text,
        model_suggestion=model_suggestion,
        hard_constraints=hard_constraints,
    )
    max_models = hard_constraints.get("max_models")
    contract_warnings: list[str] = []
    if isinstance(max_models, int) and max_models >= 0 and len(required_model_specs) > max_models:
        contract_warnings.append(
            f"required_model_specs were limited to {max_models} item(s) by explicit hard constraints."
        )
        required_model_specs = _renumber_model_specs(required_model_specs[:max_models])

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
        "primary_result_type": hard_constraints["primary_result_type"],
        "max_embedded_rows": MAX_EMBEDDED_RESULT_ROWS,
        "top_n_candidate_segments": 50 if inferential else None,
        "max_models": max_models,
        "hard_constraints": hard_constraints,
        "allowed_model_families": _allowed_model_families(task_type),
        "prohibited_operations": sorted(set(prohibited_operations)),
        "required_model_specs": required_model_specs,
        "required_model_specs_source": "natural_language_explicit" if required_model_specs else "none",
        "contract_warnings": contract_warnings,
        "valid": not issues,
        "issues": issues,
    }


def contract_as_prompt(contract: dict[str, Any]) -> str:
    return json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True)


def _allowed_model_families(task_type: str) -> list[str]:
    if task_type == "association_inference":
        return [
            "linear_regression",
            "poisson",
            "negative_binomial",
            "zero_inflated_negative_binomial",
            "logistic_regression",
        ]
    if task_type == "prediction":
        return [
            "linear_regression",
            "logistic_regression",
            "random_forest",
            "xgboost",
            "lightgbm",
        ]
    if task_type == "unsupervised":
        return ["kmeans"]
    return []


def _code_semantic_tokens(code: str) -> dict[str, set[str]]:
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return {"imports": set(), "calls": set(), "attributes": set(), "assigns": set()}

    imports: set[str] = set()
    calls: set[str] = set()
    attributes: set[str] = set()
    assigns: set[str] = set()
    imported_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attributes.add(node.attr.lower())
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            else:
                targets = [node.target]
            for target in targets:
                for name in _assignment_names(target):
                    assigns.add(name.lower())
        elif isinstance(node, ast.Call):
            called = _called_name(node.func).lower()
            if called:
                calls.add(called)
                calls.add(called.rsplit(".", 1)[-1])
                if called in imported_aliases:
                    calls.add(imported_aliases[called])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported = alias.name.lower()
                imports.add(imported)
                imports.add(imported.rsplit(".", 1)[-1])
                if alias.asname:
                    imported_aliases[alias.asname.lower()] = imported.rsplit(".", 1)[-1]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported = alias.name.lower()
                imports.add(imported)
                imports.add(imported.rsplit(".", 1)[-1])
                if alias.asname:
                    imported_aliases[alias.asname.lower()] = imported.rsplit(".", 1)[-1]
    return {"imports": imports, "calls": calls, "attributes": attributes, "assigns": assigns}


def _called_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _called_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _assignment_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for item in node.elts:
            names.extend(_assignment_names(item))
        return names
    return []


def _has_semantic_name(tokens: dict[str, set[str]], *names: str) -> bool:
    wanted = {name.lower() for name in names}
    return bool((tokens.get("imports") or set()) & wanted or (tokens.get("calls") or set()) & wanted)


def _has_attribute(tokens: dict[str, set[str]], *names: str) -> bool:
    wanted = {name.lower() for name in names}
    return bool((tokens.get("attributes") or set()) & wanted)


def _has_assignment(tokens: dict[str, set[str]], *names: str) -> bool:
    wanted = {name.lower() for name in names}
    return bool((tokens.get("assigns") or set()) & wanted)


def validate_code_against_contract(
    *,
    code: str,
    contract: dict[str, Any],
) -> list[str]:
    """Static contract checks that should run before executing generated code."""
    code_text = str(code or "")
    semantic_tokens = _code_semantic_tokens(code_text)
    issues: list[str] = []

    for prohibited in contract.get("prohibited_operations") or []:
        if prohibited == "train_test_split":
            if _has_semantic_name(semantic_tokens, "train_test_split"):
                issues.append("train_test_split is prohibited by the analysis contract.")
        elif prohibited == "stepwise_selection":
            if _has_semantic_name(semantic_tokens, "SequentialFeatureSelector", "RFE", "RFECV") or _has_attribute(
                semantic_tokens, "stepwise"
            ):
                issues.append("Stepwise/automatic feature selection is prohibited by the analysis contract.")
        elif prohibited == "best_model_selection":
            if _has_semantic_name(
                semantic_tokens,
                "GridSearchCV",
                "RandomizedSearchCV",
                "HalvingGridSearchCV",
                "HalvingRandomSearchCV",
                "cross_val_score",
                "cross_validate",
            ) or _has_attribute(semantic_tokens, "best_estimator_", "best_params_", "best_score_") or _has_assignment(
                semantic_tokens,
                "best_model",
                "best_estimator",
                "best_params",
                "best_score",
            ):
                issues.append("Best-model selection is prohibited by the analysis contract.")
        elif prohibited == "mandatory_model_export":
            if _has_semantic_name(semantic_tokens, "dump", "dumps", "compress", "b64encode", "save_model") or _has_attribute(
                semantic_tokens, "dump", "dumps", "compress", "b64encode", "save_model"
            ) or _has_assignment(
                semantic_tokens,
                "best_model_b64",
                "model_export",
                "model_bytes",
                "serialized_model",
            ):
                issues.append("Mandatory model export is prohibited by the analysis contract.")

    return issues


def validate_result_against_contract(
    *,
    code: str,
    result_json: Any,
    contract: dict[str, Any],
) -> list[str]:
    issues = list(contract.get("issues") or [])
    issues.extend(validate_code_against_contract(code=code, contract=contract))
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
    if str(contract.get("covariance") or "").upper() == "HC3" and "hc3" not in code_lower:
        issues.append("The analysis contract requires HC3 robust covariance, but HC3 is absent from the code.")

    models = result_json.get("models") if isinstance(result_json, dict) else None
    if isinstance(models, list):
        used_indexes: set[int] = set()
        for expected in contract.get("required_model_specs") or []:
            if not isinstance(expected, dict):
                continue
            family = str(expected.get("family") or "")
            compact_code = re.sub(r"[\s_\-]", "", code_lower)
            family_markers = {
                "zero_inflated_negative_binomial": ("zeroinflatednegativebinomial", "zeroinflated"),
                "negative_binomial": ("negativebinomial",),
                "poisson": ("poisson",),
                "logistic_regression": ("logisticregression", "logit"),
                "linear_regression": ("linearregression", "ols"),
                "random_forest": ("randomforest",),
                "xgboost": ("xgboost",),
                "lightgbm": ("lightgbm",),
                "kmeans": ("kmeans",),
            }.get(family, ())
            if family_markers and not any(marker in compact_code for marker in family_markers):
                issues.append(f"The code does not contain implementation evidence for required model {expected.get('id')!r} ({family}).")
            match_index = None
            for index, model in enumerate(models):
                if index in used_indexes or not isinstance(model, dict):
                    continue
                model_spec = model.get("model_spec") if isinstance(model.get("model_spec"), dict) else {}
                observed = " ".join(
                    [
                        str(model.get("name") or ""),
                        str(model.get("family") or ""),
                        str(model_spec.get("family") or ""),
                    ]
                )
                if _family_from_text(observed) == family:
                    match_index = index
                    break
            if match_index is None:
                issues.append(
                    f"The approved plan requires model {expected.get('id')!r} ({family}), but no executed model matches it."
                )
                continue
            used_indexes.add(match_index)
            model = models[match_index]
            model_spec = model.get("model_spec") if isinstance(model.get("model_spec"), dict) else {}
            if expected.get("requires_offset"):
                if "offset" not in code_lower and "exposure" not in code_lower:
                    issues.append(f"The code does not contain offset/exposure evidence for model {expected.get('id')!r}.")
                offset = model_spec.get("offset_column") or model.get("offset_column")
                if not str(offset or "").strip():
                    issues.append(f"Model {expected.get('id')!r} requires an offset/exposure, but none was reported.")
            if expected.get("requires_zero_inflation"):
                zero_part = model_spec.get("zero_inflation")
                if zero_part is not True and "zero" not in str(model.get("name") or "").lower():
                    issues.append(f"Model {expected.get('id')!r} requires a zero-inflated specification.")
    return issues


def validate_result_schema(
    *,
    result_json: Any,
    contract: dict[str, Any],
) -> list[str]:
    """Validate the evidence fields required by each analysis task type."""
    if not isinstance(result_json, dict):
        return ["result_dict must be a JSON object."]

    issues: list[str] = []
    task_type = str(contract.get("task_type") or "").strip().lower()
    issues.extend(_embedded_result_size_issues(result_json, contract=contract))
    if contract.get("primary_result_type") == "candidate_results" and not has_candidate_report_outputs(result_json):
        issues.append(
            "result_dict must contain candidate_windows/candidate_results and/or candidate_segments for a candidate-results task."
        )

    models = result_json.get("models")
    has_primary_outputs = has_primary_analysis_outputs(result_json)
    if not isinstance(models, list) or not models:
        if contract.get("required_model_specs") or task_type != "association_inference" or not has_primary_outputs:
            return issues + ["result_dict.models must be a non-empty list of executed analyses."]
        return issues

    max_models = contract.get("max_models")
    if _is_finite_number(max_models) and int(float(max_models)) >= 0 and len(models) > int(float(max_models)):
        issues.append(
            f"result_dict.models must contain at most {int(float(max_models))} model(s) according to the analysis contract."
        )

    for index, model in enumerate(models, start=1):
        path = f"result_dict.models[{index - 1}]"
        if not isinstance(model, dict):
            issues.append(f"{path} must be a JSON object.")
            continue
        if not str(model.get("name") or "").strip():
            issues.append(f"{path}.name is required.")
        issues.extend(_placeholder_model_issues(model, path=path, task_type=task_type))
        observed_model_spec = model.get("model_spec") if isinstance(model.get("model_spec"), dict) else {}
        observed_family = str(observed_model_spec.get("family") or model.get("family") or "").strip()
        allowed_families = {
            str(item)
            for item in (contract.get("allowed_model_families") or [])
            if str(item).strip()
        }
        if task_type == "association_inference" and observed_family in _NON_INFERENTIAL_MODEL_FAMILIES:
            issues.append(
                f"{path}.model_spec.family {observed_family!r} is not a coefficient-based auxiliary inference model."
            )
        elif allowed_families and observed_family and observed_family not in allowed_families:
            issues.append(f"{path}.model_spec.family {observed_family!r} is not allowed by the analysis contract.")
        if contract.get("required_model_specs"):
            model_spec = model.get("model_spec")
            if not isinstance(model_spec, dict):
                issues.append(f"{path}.model_spec is required for verified model specifications.")
            else:
                for key in ("family", "outcome", "features"):
                    if key not in model_spec:
                        issues.append(f"{path}.model_spec.{key} is required for verified model specifications.")
                family = str(model_spec.get("family") or "").strip()
                if "family" in model_spec and not family:
                    issues.append(f"{path}.model_spec.family must be non-empty.")
                expected_outcome = str(contract.get("outcome") or "").strip()
                observed_outcome = str(model_spec.get("outcome") or "").strip()
                if "outcome" in model_spec and expected_outcome and observed_outcome != expected_outcome:
                    issues.append(
                        f"{path}.model_spec.outcome must match the contract outcome {expected_outcome!r}."
                    )
                features = model_spec.get("features")
                if "features" in model_spec:
                    if not isinstance(features, list) or not [
                        feature for feature in features if str(feature or "").strip()
                    ]:
                        issues.append(f"{path}.model_spec.features must be a non-empty list.")
                    else:
                        feature_names = [str(feature).strip() for feature in features]
                        if expected_outcome and expected_outcome in feature_names:
                            issues.append(f"{path}.model_spec.features must not include the outcome column.")
                        if task_type == "association_inference":
                            location_features = {
                                feature
                                for feature in feature_names
                                if feature.lower() in {"bin_id", "start_bp", "end_bp"}
                            }
                            if location_features:
                                issues.append(
                                    f"{path}.model_spec.features must not include genomic window identifier/location columns: "
                                    + ", ".join(sorted(location_features))
                                    + "."
                                )
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


def has_primary_analysis_outputs(result_json: Any) -> bool:
    if not isinstance(result_json, dict):
        return False
    for key in (
        "analysis_tables",
        "candidate_results",
        "candidate_windows",
        "candidate_segments",
        "summaries",
    ):
        value = result_json.get(key)
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, list) and value:
            return True
    artifacts = result_json.get("artifacts")
    if isinstance(artifacts, dict):
        return bool(
            artifacts.get("candidate_results")
            or artifacts.get("candidate_segments")
            or artifacts.get("analysis_tables")
        )
    return False


def _placeholder_model_issues(model: dict[str, Any], *, path: str, task_type: str) -> list[str]:
    issues: list[str] = []
    text_fields = [
        str(model.get("name") or ""),
        str(model.get("family") or ""),
    ]
    model_spec = model.get("model_spec") if isinstance(model.get("model_spec"), dict) else {}
    text_fields.append(str(model_spec.get("family") or ""))
    if any(term in field.lower() for field in text_fields for term in _PLACEHOLDER_TERMS):
        issues.append(f"{path} appears to be a placeholder model, not an executed analysis.")

    metrics = model.get("metrics")
    if isinstance(metrics, dict):
        metric_keys = {str(key).strip().lower() for key in metrics}
        if metric_keys and metric_keys.issubset({"status_code", "status", "placeholder"}):
            issues.append(f"{path}.metrics contains only placeholder/status values.")

    n_obs = model.get("n_obs")
    if task_type in {"association_inference", "prediction"} and _is_finite_number(n_obs) and float(n_obs) <= 1:
        issues.append(f"{path}.n_obs={n_obs!r} is too small to evidence an executed model.")

    coefficients = model.get("coefficients")
    if isinstance(coefficients, list):
        for coefficient_index, coefficient in enumerate(coefficients):
            if not isinstance(coefficient, dict):
                continue
            term = str(coefficient.get("term") or "").strip().lower()
            if term in _PLACEHOLDER_TERMS or any(marker in term for marker in _PLACEHOLDER_TERMS):
                issues.append(
                    f"{path}.coefficients[{coefficient_index}].term appears to be a placeholder."
                )
    return issues


def _embedded_result_size_issues(result_json: Any, *, contract: dict[str, Any]) -> list[str]:
    max_rows = int(contract.get("max_embedded_rows") or MAX_EMBEDDED_RESULT_ROWS)
    issues: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, list):
            if len(value) > max_rows and _looks_like_record_list(value):
                issues.append(
                    f"{path or 'result_dict'} embeds {len(value)} row records; return only preview/top rows "
                    f"(≤{max_rows}) plus row_count/schema or a managed artifact reference."
                )
                return
            for index, item in enumerate(value[: max_rows + 1]):
                visit(item, f"{path}[{index}]" if path else f"[{index}]")
            return
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))

    visit(result_json, "")
    return issues


def _looks_like_record_list(value: list[Any]) -> bool:
    if not value:
        return False
    sample = value[: min(len(value), 5)]
    return all(isinstance(item, dict) for item in sample)


def validate_modeling_result(
    *,
    code: str,
    result_json: Any,
    contract: dict[str, Any],
) -> list[str]:
    """Run all core modeling result checks used by both workflow and UI paths."""
    issues = validate_result_against_contract(
        code=code,
        result_json=result_json,
        contract=contract,
    )
    issues.extend(validate_result_schema(result_json=result_json, contract=contract))
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
