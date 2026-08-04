"""
Modeling workflow 本地实现。

Workflow stages:
    Start → Condition(modeling_auto==True)
      → Sec4_get_model_suggestion(LLM)   [推荐模型]
      → Sec4_refine_suggestion(LLM)      [精炼]
      → get_query(LLM) → Knowledge(RAG) → format_recall(plugin)
      → sec4_code_generation(LLM)        [生成训练代码]
      → Variable assign_1: code_modeling = code
      → Loop(max 5): [修复循环]
          ├─ code_runner(HTTP→本地)
          ├─ if success: break
          └─ sec4_code_fixed(LLM) → 更新 code_modeling
      → Code_2(取 Loop 的 final_code + result_list)
      → sec4_result_format_prompt(LLM)   [解析结果]
      → Sec4_summary_html(LLM)           [章节正文]
      → Sec4_check_abstract(LLM)         [摘要]
      → sec4_composer(plugin)
      → Code(兜底) → End

输出:
    {
      "summary_4": {title, desc, result, code},
      "abstract_4": "...",
      "model_suggestion": "..."
    }
"""
from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from core.bounded_code_execution import run_bounded_safe_exec
from core.llm_client import (
    LLMOutputIncompleteError,
    LLMOutputTruncatedError,
    chat,
    chat_code,
    chat_suggestion,
    submit_with_context,
)
from core.code_runtime_profile import build_code_runtime_constraints, infer_data_row_count
from core.modeling_runtime_compat import validate_modeling_runtime_compatibility
from core.modeling_contract import (
    build_analysis_contract,
    contract_as_prompt,
    format_contract_violations,
    has_primary_analysis_outputs,
    validate_code_against_contract,
    validate_modeling_result,
)
from core.modeling_report_artifacts import build_modeling_report_artifacts
from core.modeling_table_utils import (
    build_model_comparison_table_bundle,
)
from core.prompt_template import render_file
from core.suggestion_revision import normalize_suggestion_output, revise_suggestion
from core.rag_retriever import retrieve
from core.report_language import (
    app_language_instruction,
    app_language_name,
    app_language_ref_context_empty,
    is_english_language,
    normalize_app_language,
)
from core.workflow_runner import to_str
from workflows._plugins import (
    format_recall,
    sec4_composer,
)

MAX_FIX_ATTEMPTS = 5
MODELING_CODE_PROMPT_MAX_CHARS = 6000
MODELING_LONG_STRING_MAX_CHARS = 800
MODELING_PROMPT_HEAD_MAX_CHARS = int(os.getenv("AUTOSTAT_MODELING_PROMPT_HEAD_CHARS", "12000"))
MODELING_PROMPT_DATA_MAX_CHARS = int(os.getenv("AUTOSTAT_MODELING_PROMPT_DATA_CHARS", "12000"))
MODELING_PROMPT_USER_TEXT_MAX_CHARS = int(os.getenv("AUTOSTAT_MODELING_PROMPT_USER_TEXT_CHARS", "6000"))
MODELING_PROMPT_CONTEXT_MAX_CHARS = int(os.getenv("AUTOSTAT_MODELING_PROMPT_CONTEXT_CHARS", "12000"))
MODELING_PROMPT_MAX_COLUMNS = int(os.getenv("AUTOSTAT_MODELING_PROMPT_MAX_COLUMNS", "300"))
MODELING_GENERIC_LIST_MAX_ITEMS = 12
MODELING_MODEL_LIST_MAX_ITEMS = 30
MODELING_IMPORTANCE_TOP_K = 20
MODELING_SAMPLE_VALUES = 8
MODELING_BASE64_MIN_CHARS = 256

_BASE64_KEY_HINTS = ("b64", "base64")
_ARTIFACT_KEY_HINTS = (
    "artifact",
    "artifacts",
    "model_bytes",
    "pickle",
    "joblib",
    "gz_bytes",
    "gzip_bytes",
)
_LARGE_RECORD_KEY_HINTS = (
    "records",
    "prediction_records",
    "predictions_df_records",
    "rows",
)
_IMPORTANCE_KEY_HINTS = (
    "importance",
    "importances",
    "feature_importance",
    "feature_importances",
)
_CORE_RESULT_KEYS = {
    "dataset",
    "task",
    "task_type",
    "target",
    "models",
    "best_model",
    "metrics",
    "score",
    "intermediate",
    "feature_importance",
    "feature_importances",
    "coefficients",
    "coef",
    "artifacts",
    "artifact_warning",
    "analysis_manifest",
}


def _build_modeling_ctx(
    *,
    data: str,
    df_head: str,
    columns: list,
    target: str = "",
    train_code: str = "",
    user_input: str = "",
    user_prompt: str = "",
    add_preference: str = "",
    preference_selected: str = "",
    ref_context: str = "",
    language: str = "zh",
    task_type: str = "auto",
) -> dict[str, Any]:
    """构造 modeling workflow 公共上下文。"""
    language = normalize_app_language(language)
    full_columns = [str(column) for column in (columns or [])]
    contract = build_analysis_contract(
        target=target,
        columns=full_columns,
        user_input=user_input or user_prompt,
        add_preference=add_preference,
        task_type=task_type,
    )
    prompt_columns = _compact_columns_for_prompt(full_columns, target=target)
    return {
        "data": _truncate_prompt_text(data, MODELING_PROMPT_DATA_MAX_CHARS),
        "df_head": _compact_df_head_for_prompt(
            df_head,
            prompt_columns=prompt_columns,
            max_chars=MODELING_PROMPT_HEAD_MAX_CHARS,
        ),
        "columns": prompt_columns,
        "_all_columns": full_columns,
        "target": target or "",
        "train_code": _truncate_prompt_text(train_code or "", MODELING_CODE_PROMPT_MAX_CHARS),
        "user_input": _truncate_prompt_text(user_input or "", MODELING_PROMPT_USER_TEXT_MAX_CHARS),
        "user_prompt": _truncate_prompt_text(
            user_prompt or user_input or "", MODELING_PROMPT_USER_TEXT_MAX_CHARS
        ),
        "add_preference": _truncate_prompt_text(
            add_preference or "", MODELING_PROMPT_USER_TEXT_MAX_CHARS
        ),
        "additional_preference": _truncate_prompt_text(
            add_preference or "", MODELING_PROMPT_USER_TEXT_MAX_CHARS
        ),
        "preference_selected": _truncate_prompt_text(
            preference_selected or "", MODELING_PROMPT_USER_TEXT_MAX_CHARS
        ),
        "preference_select": _truncate_prompt_text(
            preference_selected or "", MODELING_PROMPT_USER_TEXT_MAX_CHARS
        ),
        "ref_context": _truncate_prompt_text(
            ref_context or app_language_ref_context_empty(language),
            MODELING_PROMPT_CONTEXT_MAX_CHARS,
        ),
        "language": language,
        "language_name": app_language_name(language),
        "language_instruction": app_language_instruction(language),
        "analysis_contract": contract,
        "analysis_contract_json": contract_as_prompt(contract),
        "task_type": task_type or "auto",
        "runtime_constraints_json": build_code_runtime_constraints(
            data,
            target=target,
            include_modeling_library_compatibility=True,
        ),
    }


def ensure_analysis_contract(ctx: dict[str, Any]) -> dict[str, Any]:
    """Backfill the current contract schema for contexts kept in an old UI session."""
    working_ctx = dict(ctx)
    existing = working_ctx.get("analysis_contract")
    if isinstance(existing, dict) and "required_model_specs" in existing:
        working_ctx.setdefault("analysis_contract_json", contract_as_prompt(existing))
        return working_ctx
    contract = build_analysis_contract(
        target=str(working_ctx.get("target") or ""),
        columns=list(working_ctx.get("_all_columns") or working_ctx.get("columns") or []),
        user_input=str(working_ctx.get("user_input") or working_ctx.get("user_prompt") or ""),
        add_preference=str(working_ctx.get("add_preference") or working_ctx.get("additional_preference") or ""),
        refined_suggestions=str(working_ctx.get("refined_suggestions") or working_ctx.get("refine_suggestion") or ""),
        model_suggestion=str(working_ctx.get("model_suggestion") or ""),
        task_type=str(working_ctx.get("task_type") or "auto"),
    )
    working_ctx["analysis_contract"] = contract
    working_ctx["analysis_contract_json"] = contract_as_prompt(contract)
    return working_ctx


def _truncate_prompt_text(value: Any, max_chars: int) -> str:
    text = to_str(value).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n...[truncated for modeling prompt; original chars={len(text)}]"


def _compact_columns_for_prompt(columns: Any, *, target: str = "") -> list[str]:
    raw_columns = [str(column) for column in (columns or [])]
    if len(raw_columns) <= MODELING_PROMPT_MAX_COLUMNS:
        return raw_columns

    target_text = str(target or "").strip()
    kept: list[str] = []
    if target_text and target_text in raw_columns:
        kept.append(target_text)
    for column in raw_columns:
        if column in kept:
            continue
        kept.append(column)
        if len(kept) >= MODELING_PROMPT_MAX_COLUMNS:
            break
    omitted = len(raw_columns) - len(kept)
    kept.append(f"...[{omitted} omitted columns; full df is still available at runtime]")
    return kept


def _compact_df_head_for_prompt(
    df_head: Any,
    *,
    prompt_columns: list[str],
    max_chars: int,
) -> str:
    text = to_str(df_head).strip()
    if len(text) <= max_chars:
        return text

    try:
        parsed = json.loads(text)
    except Exception:
        return _truncate_prompt_text(text, max_chars)

    if isinstance(parsed, dict):
        keep = {column for column in prompt_columns if not column.startswith("...[")}
        compacted = {str(key): value for key, value in parsed.items() if str(key) in keep}
        compact_text_value = json.dumps(compacted, ensure_ascii=False, default=str)
        if len(compact_text_value) <= max_chars:
            return compact_text_value
    return _truncate_prompt_text(text, max_chars)


def _empty_modeling_result() -> dict[str, Any]:
    return {
        "summary_4": {
            "title": "", "desc": "", "result": "", "code": "",
            "table_title": "", "table_markdown": "", "table_html": "",
        },
        "abstract_4": "",
        "model_suggestion": "",
    }


def run_modeling_phase1(
    *,
    data: str,
    df_head: str,
    columns: list,
    modeling_auto: bool = True,
    target: str = "",
    train_code: str = "",
    user_input: str = "",
    user_prompt: str = "",
    add_preference: str = "",
    preference_selected: str = "",
    ref_context: str = "",
    language: str = "zh",
    task_type: str = "auto",
) -> dict[str, Any]:
    """Phase 1: 生成 model_suggestion + refined_suggestions，快速返回给前端展示。"""
    if not modeling_auto:
        return {"model_suggestion": "", "refined_suggestions": "", "_ctx": {}}

    ctx = _build_modeling_ctx(
        data=data, df_head=df_head, columns=columns, target=target,
        train_code=train_code, user_input=user_input, user_prompt=user_prompt,
        add_preference=add_preference, preference_selected=preference_selected,
        ref_context=ref_context, language=language,
        task_type=task_type,
    )

    try:
        sug_sys = render_file("modeling/sec4_get_model_suggestion_llm_sys.txt", ctx)
        sug_user = render_file("modeling/sec4_get_model_suggestion_llm_user.txt", ctx)
        model_suggestion = normalize_suggestion_output(
            chat_suggestion(sug_sys, sug_user, name="model.get_suggestion")
        )
        ctx["model_suggestion"] = model_suggestion

        ref_sys = render_file("modeling/sec4_refine_suggestion_llm_sys.txt", ctx)
        ref_user = render_file("modeling/sec4_refine_suggestion_llm_user.txt", ctx)
        refined_suggestions = normalize_suggestion_output(
            chat_suggestion(ref_sys, ref_user, name="model.refine")
        )
    except (LLMOutputIncompleteError, LLMOutputTruncatedError) as exc:
        return {
            "model_suggestion": "",
            "refined_suggestions": "",
            "analysis_contract": {"valid": False, "issues": [str(exc)]},
            "_ctx": ctx,
            "_status": "suggestion_incomplete",
            "_error": str(exc),
        }
    ctx["refined_suggestions"] = refined_suggestions
    ctx["refine_suggestion"] = refined_suggestions
    contract = build_analysis_contract(
        target=ctx.get("target", ""),
        columns=ctx.get("_all_columns") or ctx.get("columns", []),
        user_input=ctx.get("user_input") or ctx.get("user_prompt", ""),
        add_preference=ctx.get("add_preference", ""),
        refined_suggestions=refined_suggestions,
        model_suggestion=model_suggestion,
        task_type=ctx.get("task_type", "auto"),
    )
    ctx["analysis_contract"] = contract
    ctx["analysis_contract_json"] = contract_as_prompt(contract)
    ctx["runtime_constraints_json"] = build_code_runtime_constraints(
        data,
        target=str(contract.get("outcome") or target or ""),
        include_modeling_library_compatibility=True,
    )

    return {
        "model_suggestion": model_suggestion,
        "refined_suggestions": refined_suggestions,
        "analysis_contract": contract,
        "_ctx": ctx,
    }


def revise_modeling_phase1(
    *,
    ctx: dict[str, Any],
    original_requirements: str,
    revision_instruction: str,
) -> dict[str, Any]:
    revised_ctx = dict(ctx)
    try:
        revised = revise_suggestion(
            stage_label="modeling",
            original_requirements=original_requirements,
            current_suggestion=str(ctx.get("model_suggestion") or ""),
            revision_instruction=revision_instruction,
            hard_constraints=str(ctx.get("analysis_contract_json") or ""),
            language_instruction=str(ctx.get("language_instruction") or ""),
        )
        revised_ctx["model_suggestion"] = revised
        ref_sys = render_file("modeling/sec4_refine_suggestion_llm_sys.txt", revised_ctx)
        ref_user = render_file("modeling/sec4_refine_suggestion_llm_user.txt", revised_ctx)
        refined = normalize_suggestion_output(
            chat_suggestion(ref_sys, ref_user, name="model.refine_revision")
        )
    except (LLMOutputIncompleteError, LLMOutputTruncatedError) as exc:
        return {
            "model_suggestion": "",
            "refined_suggestions": "",
            "analysis_contract": {"valid": False, "issues": [str(exc)]},
            "_ctx": revised_ctx,
            "_status": "suggestion_incomplete",
            "_error": str(exc),
        }
    revised_ctx["refined_suggestions"] = refined
    revised_ctx["refine_suggestion"] = refined
    contract = build_analysis_contract(
        target=revised_ctx.get("target", ""),
        columns=revised_ctx.get("_all_columns") or revised_ctx.get("columns", []),
        user_input=revised_ctx.get("user_input") or revised_ctx.get("user_prompt", ""),
        add_preference=revised_ctx.get("add_preference", ""),
        refined_suggestions=refined,
        model_suggestion=revised,
        task_type=revised_ctx.get("task_type", "auto"),
    )
    revised_ctx["analysis_contract"] = contract
    revised_ctx["analysis_contract_json"] = contract_as_prompt(contract)
    return {
        "model_suggestion": revised,
        "refined_suggestions": refined,
        "analysis_contract": contract,
        "_ctx": revised_ctx,
    }


def generate_modeling_code(*, ctx: dict[str, Any]) -> dict[str, Any]:
    """Generate a modeling code draft without executing the training script."""
    ctx = ensure_analysis_contract(ctx)
    analysis_contract = ctx.get("analysis_contract") or {}
    if not analysis_contract.get("valid", False):
        return {
            "code": "",
            "error": format_contract_violations(list(analysis_contract.get("issues") or [])),
            "_ctx": dict(ctx),
        }

    working_ctx = dict(ctx)
    working_ctx.setdefault("model_suggestion", "")
    q_sys = render_file("modeling/get_query_llm_sys.txt", working_ctx)
    q_user = render_file("modeling/get_query_llm_user.txt", working_ctx)
    rag_query = chat(q_sys, q_user, name="model.get_query", temperature=0).strip()
    recall_results = retrieve(rag_query, top_k=3)
    recall_results = _filter_recall_results_for_contract(recall_results, analysis_contract)
    working_ctx["knowledge_results"] = format_recall(output_list=recall_results)["knowledge_results"]

    cg_sys = render_file("modeling/sec4_code_generation_llm_sys.txt", working_ctx, strict=True)
    cg_user = render_file("modeling/sec4_code_generation_llm_user.txt", working_ctx, strict=True)
    return {
        "code": _unwrap_code_block(chat_code(cg_sys, cg_user, name="model.code_generation").strip()),
        "error": "",
        "_ctx": working_ctx,
    }


def repair_modeling_code(*, ctx: dict[str, Any], code: str, error: str) -> str:
    contract_ctx = ensure_analysis_contract(ctx)
    fix_ctx = {
        **contract_ctx,
        "code": code,
        "code_modeling": code,
        "error_msg": error,
        "error": error,
        "model_suggestion": str(contract_ctx.get("model_suggestion") or ""),
    }
    fix_sys = render_file("modeling/sec4_code_fixed_llm_sys.txt", fix_ctx, strict=True)
    fix_user = render_file("modeling/sec4_code_fixed_llm_user.txt", fix_ctx, strict=True)
    return _unwrap_code_block(
        chat_code(fix_sys, fix_user, name="model.manual_code_fixer", temperature=0.3).strip()
    )


def _recall_item_contract_text(item: Any) -> str:
    if isinstance(item, dict):
        parts = [
            item.get("code"),
            item.get("output"),
            item.get("content"),
            item.get("text"),
        ]
        return "\n".join(str(part) for part in parts if part)
    return str(item or "")


def _filter_recall_results_for_contract(
    recall_results: list[dict[str, Any]],
    analysis_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Drop retrieved code templates that contain operations forbidden by the active contract."""
    if not recall_results:
        return recall_results
    filtered: list[dict[str, Any]] = []
    for item in recall_results:
        issues = validate_code_against_contract(
            code=_recall_item_contract_text(item),
            contract=analysis_contract,
        )
        if issues:
            continue
        filtered.append(item)
    return filtered


def validate_modeling_code(
    *,
    ctx: dict[str, Any],
    data: str,
    df_head: str,
    initial_code: str = "",
) -> dict[str, Any]:
    """Run the shared legacy five-attempt generation/fix loop without publishing analysis text."""
    working_ctx = ensure_analysis_contract(ctx)
    analysis_contract = working_ctx.get("analysis_contract") or {}
    if not analysis_contract.get("valid", False):
        return {
            "code": "",
            "success": False,
            "error": format_contract_violations(list(analysis_contract.get("issues") or [])),
            "contract_violations": list(analysis_contract.get("issues") or []),
            "result_json": {},
            "result_stdout": "",
            "attempts": 0,
            "_ctx": working_ctx,
        }

    if initial_code:
        current_code = _unwrap_code_block(initial_code)
    else:
        generation = generate_modeling_code(ctx=working_ctx)
        if generation.get("error"):
            return {
                "code": "",
                "success": False,
                "error": str(generation["error"]),
                "contract_violations": [],
                "result_json": {},
                "result_stdout": "",
                "attempts": 0,
                "_ctx": working_ctx,
            }
        working_ctx = generation["_ctx"]
        current_code = generation["code"]

    success = False
    last_error = ""
    final_result_json: dict[str, Any] = {}
    final_result_str = ""
    contract_violations: list[str] = []
    attempts = 0
    n_rows = infer_data_row_count(data)
    for attempt in range(MAX_FIX_ATTEMPTS):
        attempts = attempt + 1
        static_contract_issues = validate_code_against_contract(
            code=current_code,
            contract=analysis_contract,
        )
        if static_contract_issues:
            contract_violations = static_contract_issues
            last_error = format_contract_violations(static_contract_issues)
            run_result = None
        else:
            compatibility_issues = validate_modeling_runtime_compatibility(
                current_code,
                n_rows=n_rows,
            )
            if compatibility_issues:
                last_error = "Modeling runtime compatibility validation failed:\n- " + "\n- ".join(
                    compatibility_issues
                )
                run_result = None
            else:
                run_result = _run_modeling_code(
                    code=current_code,
                    data=data,
                    n_rows=n_rows,
                )
        if run_result and run_result["is_success"]:
            final_result_str = run_result.get("stdout", "")
            final_result_json = run_result.get("result_json", {})
            contract_violations = validate_modeling_result(
                code=current_code,
                result_json=final_result_json,
                contract=analysis_contract,
            )
            candidate_table = build_model_comparison_table_bundle(
                final_result_json,
                target=working_ctx.get("target", ""),
                user_input=working_ctx.get("user_input", ""),
                additional_preference=working_ctx.get("additional_preference", ""),
                language=working_ctx.get("language", "zh"),
            )
            association_primary_result = (
                str(analysis_contract.get("task_type") or "").strip().lower() == "association_inference"
                and has_primary_analysis_outputs(final_result_json)
            )
            if not candidate_table.get("has_table") and not association_primary_result:
                contract_violations.append(
                    "result_dict must contain model metrics or primary analysis tables that can produce reportable results."
                )
            if not contract_violations:
                success = True
                break
            last_error = format_contract_violations(contract_violations)
        elif run_result:
            last_error = run_result.get("error", "")

        if attempt >= MAX_FIX_ATTEMPTS - 1:
            break
        fixed = repair_modeling_code(ctx=working_ctx, code=current_code, error=last_error)
        if fixed:
            current_code = fixed

    return {
        "code": current_code,
        "success": success,
        "error": last_error,
        "contract_violations": contract_violations,
        "result_json": final_result_json,
        "result_stdout": final_result_str,
        "attempts": attempts,
        "_ctx": working_ctx,
    }


def run_modeling_phase2(
    *,
    ctx: dict[str, Any],
    data: str,
    df_head: str,
) -> dict[str, Any]:
    """Phase 2: RAG + 代码生成 + 验证修复 + 结果格式化 + 摘要。依赖 phase1 产出的 ctx。"""
    ctx = ensure_analysis_contract(ctx)
    model_suggestion = ctx.get("model_suggestion", "")
    refined_suggestions = ctx.get("refined_suggestions", "")
    analysis_contract = ctx.get("analysis_contract") or {}

    if not analysis_contract.get("valid", False):
        error_text = format_contract_violations(list(analysis_contract.get("issues") or []))
        english = is_english_language(ctx.get("language"))
        return {
            "summary_4": {
                "title": "Modeling Analysis" if english else "建模分析",
                "desc": error_text,
                "result": "",
                "code": "",
                "analysis_contract": analysis_contract,
            },
            "abstract_4": error_text,
            "model_suggestion": model_suggestion,
            "_status": "failed",
            "_contract_violations": list(analysis_contract.get("issues") or []),
        }

    validation = validate_modeling_code(ctx=ctx, data=data, df_head=df_head)
    ctx = validation["_ctx"]
    final_code = validation["code"]
    success = bool(validation["success"])
    last_error = validation["error"]
    final_result_json = validation["result_json"]
    final_result_str = validation["result_stdout"]
    contract_violations = validation["contract_violations"]
    attempt = max(0, int(validation["attempts"]) - 1)

    if not success:
        english = is_english_language(ctx.get("language"))
        error_desc = (
            f"Modeling code execution failed: {last_error[:500]}"
            if english
            else f"建模代码执行失败：{last_error[:500]}"
        )
        error_abstract = (
            f"Modeling code execution failed: {last_error[:200]}"
            if english
            else f"建模代码执行失败：{last_error[:200]}"
        )
        return {
            "summary_4": {
                "title": "Modeling Analysis" if english else "建模分析",
                "desc": error_desc,
                "result": "", "code": final_code,
                "table_title": "", "table_markdown": "", "table_html": "",
                "analysis_contract": analysis_contract,
            },
            "abstract_4": error_abstract,
            "model_suggestion": model_suggestion,
            "_status": "failed",
            "_contract_violations": contract_violations,
            "_fix_attempts": validation["attempts"],
        }

    # ---------- 结果格式化 ----------
    # 表格/内部逻辑继续使用 raw result；LLM prompt 只接收 compact evidence。
    artifact_metadata = collect_modeling_artifact_metadata(final_result_json)
    report_artifacts = build_modeling_report_artifacts(
        final_result_json,
        target=ctx.get("target", ""),
        language=ctx.get("language", "zh"),
    )
    compact_result = compact_modeling_result(
        final_result_json,
        target=ctx.get("target", ""),
        artifact_metadata=artifact_metadata,
    )
    compact_result_text = json.dumps(compact_result, ensure_ascii=False, indent=2, default=str)
    compact_code = compact_modeling_code_for_prompt(final_code)
    compact_stdout = compact_text(final_result_str, 1200)

    ctx["final_code"] = final_code
    ctx["modeling_code"] = compact_code
    ctx["code"] = compact_code
    ctx["result_json"] = compact_result_text
    ctx["modeling_result_evidence"] = compact_result_text
    ctx["modeling_report_artifacts"] = json.dumps(
        report_artifacts,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    ctx["modeling_artifact_metadata"] = artifact_metadata
    ctx["execution_stdout"] = compact_stdout
    ctx["result"] = compact_stdout
    ctx["analysis_contract"] = analysis_contract
    ctx["analysis_contract_json"] = contract_as_prompt(analysis_contract)
    table_bundle = build_model_comparison_table_bundle(
        final_result_json,
        target=ctx.get("target", ""),
        user_input=ctx.get("user_input", ""),
        additional_preference=ctx.get("additional_preference", ""),
        language=ctx.get("language", "zh"),
    )
    ctx["comparison_table_title"] = table_bundle.get("title", "")
    ctx["comparison_table_markdown"] = table_bundle.get("markdown_table", "")
    ctx["comparison_table_html"] = table_bundle.get("html_table", "")

    rfp_sys = render_file("modeling/sec4_result_format_prompt_llm_sys.txt", ctx)
    rfp_user = render_file("modeling/sec4_result_format_prompt_llm_user.txt", ctx)
    result_format = chat(rfp_sys, rfp_user, name="model.result_format").strip()
    ctx["result_format"] = result_format
    ctx["result"] = result_format

    # ---------- 章节正文 + 摘要 并行 ----------
    sh_sys = render_file("modeling/sec4_summary_html_llm_sys.txt", ctx)
    sh_user = render_file("modeling/sec4_summary_html_llm_user.txt", ctx)
    ab_sys = render_file("modeling/sec4_check_abstract_llm_sys.txt", ctx)
    ab_user = render_file("modeling/sec4_check_abstract_llm_user.txt", ctx)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_desc = submit_with_context(pool, chat, sh_sys, sh_user, name="model.summary_html")
        f_abs = submit_with_context(pool, chat, ab_sys, ab_user, name="model.check_abstract")
        desc = f_desc.result().strip()
        abstract_4 = f_abs.result().strip()

    composed = sec4_composer(
        code=final_code, desc=desc, result=result_format,
        table_title=table_bundle.get("title", ""),
        table_markdown=table_bundle.get("markdown_table", ""),
        table_html=table_bundle.get("html_table", ""),
    )
    if is_english_language(ctx.get("language")):
        composed["summary_4"]["title"] = "Modeling Analysis"
    composed["summary_4"]["analysis_contract"] = analysis_contract
    composed["summary_4"]["report_artifacts"] = report_artifacts

    return {
        "summary_4": composed["summary_4"],
        "abstract_4": abstract_4,
        "model_suggestion": model_suggestion,
        "_refined_suggestions": refined_suggestions,
        "_final_code": final_code,
        "_modeling_result_evidence": compact_result,
        "_modeling_report_artifacts": report_artifacts,
        "_modeling_artifact_metadata": artifact_metadata,
        "_fix_attempts": attempt + 1 if success else MAX_FIX_ATTEMPTS,
        "_status": "succeeded",
        "_analysis_contract": analysis_contract,
        "_contract_violations": [],
    }


def run_modeling_workflow(
    *,
    data: str,
    df_head: str,
    columns: list,
    modeling_auto: bool = True,
    target: str = "",
    train_code: str = "",
    user_input: str = "",
    user_prompt: str = "",
    add_preference: str = "",
    preference_selected: str = "",
    ref_context: str = "",
    language: str = "zh",
    task_type: str = "auto",
) -> dict[str, Any]:
    """完整执行（兼容旧调用方式，顺序执行 phase1 + phase2）。"""
    if not modeling_auto:
        return _empty_modeling_result()

    p1 = run_modeling_phase1(
        data=data, df_head=df_head, columns=columns,
        modeling_auto=modeling_auto, target=target, train_code=train_code,
        user_input=user_input, user_prompt=user_prompt,
        add_preference=add_preference, preference_selected=preference_selected,
        ref_context=ref_context, language=language,
        task_type=task_type,
    )
    if p1.get("_status") == "suggestion_incomplete":
        english = is_english_language(language)
        error_text = str(p1.get("_error") or "")
        desc = (
            f"Modeling suggestion generation did not complete: {error_text}"
            if english
            else f"建模建议生成未完成：{error_text}"
        )
        return {
            "summary_4": {
                "title": "Modeling Analysis" if english else "建模分析",
                "desc": desc,
                "result": "",
                "code": "",
                "table_title": "",
                "table_markdown": "",
                "table_html": "",
            },
            "abstract_4": desc,
            "model_suggestion": "",
            "_status": "suggestion_incomplete",
            "_error": error_text,
            "_contract_violations": [error_text] if error_text else [],
        }
    ctx = p1.get("_ctx")
    if not ctx:
        return _empty_modeling_result()

    return run_modeling_phase2(ctx=ctx, data=data, df_head=df_head)


def collect_modeling_artifact_metadata(result_json: Any) -> dict[str, Any]:
    payload, _ = _coerce_modeling_payload(result_json)
    items: list[dict[str, Any]] = []
    _collect_artifact_items(payload, path=[], items=items)
    return {
        "present": bool(items),
        "omitted_from_prompt": bool(items),
        "items": items[:50],
        "omitted_item_count": max(0, len(items) - 50),
    }


def compact_modeling_result(
    result_json: Any,
    *,
    target: str = "",
    artifact_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload, raw_text = _coerce_modeling_payload(result_json)
    if not isinstance(payload, dict):
        return {
            "available": False,
            "raw_json_chars": len(raw_text),
            "note": "Modeling result payload could not be parsed as a dict.",
        }

    artifacts = artifact_metadata or collect_modeling_artifact_metadata(payload)
    out: dict[str, Any] = {
        "available": True,
        "raw_json_chars": len(raw_text),
        "target": compact_text(target, 200),
    }

    for key in ("dataset", "task", "task_type", "type"):
        if key in payload:
            out[key] = _compact_for_llm(payload.get(key), key=key)

    if "analysis_manifest" in payload:
        out["analysis_manifest"] = _compact_for_llm(
            payload.get("analysis_manifest"),
            key="analysis_manifest",
        )

    models = payload.get("models")
    if isinstance(models, list):
        out["models"] = [_compact_model_entry(item) for item in models[:MODELING_MODEL_LIST_MAX_ITEMS]]
        out["model_count"] = len(models)
        if len(models) > MODELING_MODEL_LIST_MAX_ITEMS:
            out["omitted_model_count"] = len(models) - MODELING_MODEL_LIST_MAX_ITEMS
    elif models is not None:
        out["models"] = _compact_for_llm(models, key="models")

    if "best_model" in payload:
        out["best_model"] = _compact_for_llm(payload.get("best_model"), key="best_model")

    if "metrics" in payload:
        out["metrics"] = _compact_for_llm(payload.get("metrics"), key="metrics")
    if "score" in payload:
        out["score"] = _compact_for_llm(payload.get("score"), key="score")

    report_artifacts = build_modeling_report_artifacts(payload, target=target)
    out["report_artifacts"] = report_artifacts

    interpretability = _extract_interpretability(payload)
    if interpretability:
        out["interpretability"] = interpretability

    if "intermediate" in payload:
        out["intermediate_summary"] = _compact_for_llm(
            payload.get("intermediate"),
            key="intermediate",
        )

    if "artifact_warning" in payload:
        out["artifact_warning"] = _compact_for_llm(
            payload.get("artifact_warning"),
            key="artifact_warning",
        )
    out["artifacts"] = artifacts

    additional: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _CORE_RESULT_KEYS or _is_artifact_key(key) or _is_base64_key(key):
            continue
        if len(additional) >= 12:
            additional["omitted_additional_field_count"] = (
                len([k for k in payload if k not in _CORE_RESULT_KEYS]) - len(additional)
            )
            break
        additional[key] = _compact_for_llm(value, key=key)
    if additional:
        out["additional_fields"] = additional

    return out


def compact_modeling_code_for_prompt(code: Any) -> str:
    text = to_str(code).strip()
    if not text:
        return ""
    if len(text) <= MODELING_CODE_PROMPT_MAX_CHARS:
        return text
    head_chars = int(MODELING_CODE_PROMPT_MAX_CHARS * 0.72)
    tail_chars = MODELING_CODE_PROMPT_MAX_CHARS - head_chars
    omitted = len(text) - head_chars - tail_chars
    return (
        text[:head_chars].rstrip()
        + f"\n\n...[建模代码过长，已省略 {omitted} 字符]...\n\n"
        + text[-tail_chars:].lstrip()
    )


def compact_text(value: Any, max_chars: int = MODELING_LONG_STRING_MAX_CHARS) -> str:
    text = to_str(value).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"...[截断，原始长度 {len(text)} 字符]"


def _coerce_modeling_payload(result_json: Any) -> tuple[Any, str]:
    if isinstance(result_json, str):
        raw_text = result_json
        try:
            return json.loads(result_json), raw_text
        except Exception:
            return result_json, raw_text
    try:
        raw_text = json.dumps(result_json, ensure_ascii=False, default=str)
    except Exception:
        raw_text = to_str(result_json)
    return result_json, raw_text


def _compact_for_llm(value: Any, *, key: str = "", depth: int = 0) -> Any:
    key_text = key.lower()
    if _is_base64_key(key_text) or _is_artifact_key(key_text):
        return _artifact_value_metadata(key, value)

    if value is None or isinstance(value, (bool, int, float, str)):
        return _compact_scalar(value, key=key)

    if isinstance(value, dict):
        if depth >= 5:
            return _summarize_container(value)
        out: dict[str, Any] = {}
        for child_key, child_value in value.items():
            child_key_text = str(child_key)
            if _is_base64_key(child_key_text) or _is_artifact_key(child_key_text):
                out[child_key_text] = _artifact_value_metadata(child_key_text, child_value)
            else:
                out[child_key_text] = _compact_for_llm(
                    child_value,
                    key=child_key_text,
                    depth=depth + 1,
                )
        return out

    if isinstance(value, (list, tuple)):
        return _compact_list(list(value), key=key, depth=depth)

    return compact_text(value)


def _compact_model_entry(value: Any) -> Any:
    if not isinstance(value, dict):
        return _compact_for_llm(value, key="model")

    out: dict[str, Any] = {}
    for key in ("name", "model", "model_name", "type", "task_type"):
        if key in value:
            out[key] = _compact_scalar(value.get(key), key=key)
    if "metrics" in value:
        out["metrics"] = _compact_for_llm(value.get("metrics"), key="metrics")
    for key in ("score", "rank", "selected", "notes"):
        if key in value:
            out[key] = _compact_for_llm(value.get(key), key=key)

    for key, child_value in value.items():
        if key in out or key in {"metrics"}:
            continue
        if _is_importance_key(key):
            out[key] = _compact_importance(child_value, key=key)
        elif key not in {"artifacts"} and not _is_artifact_key(key) and not _is_base64_key(key):
            compacted = _compact_for_llm(child_value, key=key)
            if not _is_empty_compact_value(compacted):
                out[key] = compacted
    return out


def _compact_list(values: list[Any], *, key: str = "", depth: int = 0) -> Any:
    if not values:
        return []
    if _is_importance_key(key):
        return _compact_importance(values, key=key)
    if _is_large_record_key(key):
        return {
            "count": len(values),
            "sample": [_compact_for_llm(item, key=key, depth=depth + 1) for item in values[:3]],
            "omitted_count": max(0, len(values) - 3),
        }

    scalar_values = list(_iter_leaf_scalars(values, limit=5000))
    if scalar_values:
        numeric_summary = _numeric_summary(scalar_values)
        if numeric_summary and len(values) > MODELING_SAMPLE_VALUES:
            leaf_count = _count_leaf_values(values)
            if len(scalar_values) < leaf_count:
                numeric_summary["computed_from_sample_count"] = len(scalar_values)
            sample_source = (
                values
                if all(not isinstance(item, (dict, list, tuple)) for item in values)
                else scalar_values
            )
            return {
                "count": leaf_count,
                "sample": _sample_values(list(sample_source), MODELING_SAMPLE_VALUES),
                "numeric_summary": numeric_summary,
            }

    if len(values) <= MODELING_GENERIC_LIST_MAX_ITEMS:
        return [_compact_for_llm(item, key=key, depth=depth + 1) for item in values]

    head_count = MODELING_GENERIC_LIST_MAX_ITEMS // 2
    tail_count = MODELING_GENERIC_LIST_MAX_ITEMS - head_count
    sample_items = values[:head_count] + values[-tail_count:]
    return {
        "count": len(values),
        "sample": [_compact_for_llm(item, key=key, depth=depth + 1) for item in sample_items],
        "omitted_count": len(values) - len(sample_items),
    }


def _extract_interpretability(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if _is_importance_key(key):
            out[key] = _compact_importance(value, key=key)
    intermediate = payload.get("intermediate")
    if isinstance(intermediate, dict):
        for key, value in intermediate.items():
            if _is_importance_key(key):
                out[key] = _compact_importance(value, key=key)
    return out


def _compact_importance(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        rows = []
        for feature, score in value.items():
            numeric_score = _to_finite_float(score)
            rows.append(
                {
                    "feature": compact_text(feature, 120),
                    "value": numeric_score if numeric_score is not None else _compact_scalar(score, key=key),
                }
            )
        return _top_importance_rows(rows, total_count=len(rows))

    if isinstance(value, list):
        rows = []
        for item in value:
            row = _importance_row_from_item(item, key=key)
            if row:
                rows.append(row)
        if rows:
            return _top_importance_rows(rows, total_count=len(value))
    return _compact_for_llm(value, key="values")


def _importance_row_from_item(item: Any, *, key: str = "") -> dict[str, Any] | None:
    if isinstance(item, dict):
        feature = (
            item.get("feature")
            or item.get("name")
            or item.get("variable")
            or item.get("column")
            or item.get("term")
        )
        score = None
        for score_key in (
            "importance",
            "feature_importance",
            "coefficient",
            "coef",
            "value",
            "score",
            "weight",
        ):
            if score_key in item:
                score = item.get(score_key)
                break
        if feature is None and score is None:
            return None
        numeric_score = _to_finite_float(score)
        return {
            "feature": compact_text(feature, 120),
            "value": numeric_score if numeric_score is not None else _compact_scalar(score, key=key),
        }
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        numeric_score = _to_finite_float(item[1])
        return {
            "feature": compact_text(item[0], 120),
            "value": numeric_score if numeric_score is not None else _compact_scalar(item[1], key=key),
        }
    return None


def _top_importance_rows(rows: list[dict[str, Any]], *, total_count: int) -> dict[str, Any]:
    def sort_value(row: dict[str, Any]) -> float:
        value = row.get("value")
        number = _to_finite_float(value)
        return abs(number) if number is not None else 0.0

    top_rows = sorted(rows, key=sort_value, reverse=True)[:MODELING_IMPORTANCE_TOP_K]
    return {
        "count": total_count,
        "top": top_rows,
        "omitted_count": max(0, total_count - len(top_rows)),
    }


def _collect_artifact_items(value: Any, *, path: list[str], items: list[dict[str, Any]]) -> None:
    if len(items) > 200:
        return
    key = path[-1] if path else ""

    if isinstance(value, dict):
        if _is_artifact_key(key):
            items.append(_artifact_item_metadata(path, value))
        for child_key, child_value in value.items():
            child_path = path + [str(child_key)]
            child_key_text = str(child_key)
            if _is_base64_key(child_key_text) or _is_artifact_key(child_key_text):
                items.append(_artifact_item_metadata(child_path, child_value))
            elif isinstance(child_value, (dict, list, tuple)):
                _collect_artifact_items(child_value, path=child_path, items=items)
            elif isinstance(child_value, str) and _looks_like_base64(child_value, key=child_key_text):
                items.append(_artifact_item_metadata(child_path, child_value))
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(list(value)[:20]):
            _collect_artifact_items(item, path=path + [str(index)], items=items)


def _artifact_value_metadata(key: str, value: Any) -> dict[str, Any]:
    metadata = _artifact_item_metadata([key] if key else [], value)
    metadata["omitted_from_prompt"] = True
    return metadata


def _artifact_item_metadata(path: list[str], value: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": ".".join(path) if path else "",
        "type": type(value).__name__,
        "omitted_from_prompt": True,
    }
    if isinstance(value, str):
        item["chars"] = len(value)
        item["base64_like"] = _looks_like_base64(value, key=path[-1] if path else "")
    elif isinstance(value, dict):
        item["keys"] = list(value.keys())[:20]
        item["key_count"] = len(value)
    elif isinstance(value, (list, tuple)):
        item["count"] = len(value)
    else:
        item["value_preview"] = compact_text(value, 120)
    return item


def _compact_scalar(value: Any, *, key: str = "") -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return ""
        return round(number, 6) if isinstance(value, float) else value
    text = to_str(value).strip()
    if _looks_like_base64(text, key=key):
        return _artifact_value_metadata(key, text)
    return compact_text(text, MODELING_LONG_STRING_MAX_CHARS)


def _numeric_summary(values: list[Any]) -> dict[str, Any]:
    numeric = [_to_finite_float(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return {}
    sorted_numeric = sorted(numeric)
    count = len(sorted_numeric)
    mean = sum(sorted_numeric) / count
    variance = sum((value - mean) ** 2 for value in sorted_numeric) / count
    return {
        "count": count,
        "min": round(sorted_numeric[0], 6),
        "p25": round(_percentile(sorted_numeric, 0.25), 6),
        "median": round(_percentile(sorted_numeric, 0.5), 6),
        "p75": round(_percentile(sorted_numeric, 0.75), 6),
        "max": round(sorted_numeric[-1], 6),
        "mean": round(mean, 6),
        "std": round(math.sqrt(variance), 6),
    }


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _iter_leaf_scalars(values: Any, *, limit: int) -> Any:
    stack = [values]
    count = 0
    while stack and count < limit:
        current = stack.pop(0)
        if isinstance(current, (list, tuple)):
            stack = list(current[:limit]) + stack
            continue
        if isinstance(current, dict):
            continue
        yield current
        count += 1


def _count_leaf_values(values: Any) -> int:
    if not isinstance(values, (list, tuple)):
        return 1
    total = 0
    stack = [values]
    while stack:
        current = stack.pop()
        if isinstance(current, (list, tuple)):
            stack.extend(current)
        else:
            total += 1
    return total


def _sample_values(values: list[Any], max_items: int) -> list[Any]:
    if len(values) <= max_items:
        sample = values
    else:
        head_count = max_items // 2
        tail_count = max_items - head_count
        sample = values[:head_count] + values[-tail_count:]
    return [_compact_scalar(value) for value in sample]


def _summarize_container(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "dict", "key_count": len(value), "keys": list(value.keys())[:20]}
    if isinstance(value, (list, tuple)):
        return {"type": "list", "count": len(value)}
    return {"type": type(value).__name__, "preview": compact_text(value, 200)}


def _to_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _looks_like_base64(value: Any, *, key: str = "") -> bool:
    text = to_str(value).strip()
    key_text = key.lower()
    if len(text) < MODELING_BASE64_MIN_CHARS:
        return False
    if _is_base64_key(key_text):
        return True
    if not _is_artifact_key(key_text):
        return False
    compact = re.sub(r"\s+", "", text)
    return bool(re.fullmatch(r"[A-Za-z0-9+/=_-]+", compact))


def _is_base64_key(key: str) -> bool:
    key_text = key.lower()
    return any(hint in key_text for hint in _BASE64_KEY_HINTS)


def _is_artifact_key(key: str) -> bool:
    key_text = key.lower()
    if key_text == "artifact_warning":
        return False
    return any(hint in key_text for hint in _ARTIFACT_KEY_HINTS)


def _is_importance_key(key: str) -> bool:
    key_text = key.lower()
    return any(hint in key_text for hint in _IMPORTANCE_KEY_HINTS)


def _is_large_record_key(key: str) -> bool:
    key_text = key.lower()
    return any(hint in key_text for hint in _LARGE_RECORD_KEY_HINTS)


def _is_empty_compact_value(value: Any) -> bool:
    return value in ("", None, [], {})


# ===================================================================
# 建模代码专用 runner —— 比 preprocessing 多一个 result_json 输出
# ===================================================================


def _records_json_to_dataframe(value: Any) -> pd.DataFrame:
    raw = to_str(value) or "[]"
    records = json.loads(raw)
    if not isinstance(records, (list, dict)):
        raise ValueError("data must be a JSON records list or object.")
    return pd.DataFrame(records)


def _clean_modeling_transport_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, pd.DataFrame):
        return {
            "rows": len(value),
            "columns": [str(column) for column in value.columns],
            "sample": value.head(10).to_dict(orient="records"),
        }
    if isinstance(value, pd.Series):
        if len(value) <= 50:
            return value.tolist()
        return {"count": len(value), "sample": value.head(10).tolist()}
    if isinstance(value, np.ndarray):
        values = value.tolist()
        if value.size <= 50:
            return values
        flat_values = value.reshape(-1).tolist()
        return _summarize_numeric_list(flat_values) if _looks_numeric_list(flat_values) else {
            "shape": list(value.shape),
            "sample": flat_values[:10],
        }
    if isinstance(value, dict):
        return {str(key): _clean_modeling_transport_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_modeling_transport_value(child) for child in value]
    return value


def _looks_numeric_list(values: Any) -> bool:
    if not isinstance(values, list) or not values:
        return False
    sample = values[:50]
    return all(isinstance(value, (int, float, np.integer, np.floating)) for value in sample)


def _summarize_numeric_list(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    summary: dict[str, Any] = {"count": int(arr.size), "sample": values[:10]}
    if finite.size:
        summary.update(
            {
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
            }
        )
    return summary


def _strip_modeling_transport_heavy_values(value: Any, key: str = "") -> Any:
    key_text = str(key).lower()
    heavy_key_hints = (
        "artifact",
        "b64",
        "base64",
        "pickle",
        "gzip",
        "blob",
        "bytes",
        "model_object",
        "serialized",
        "raw_model",
        "fitted_model",
    )
    if any(hint in key_text for hint in heavy_key_hints):
        return {"stripped": True, "reason": "model artifact removed before report transport"}
    if isinstance(value, dict):
        return {
            str(child_key): _strip_modeling_transport_heavy_values(child_value, str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        if len(value) > 50 and _looks_numeric_list(value):
            return _summarize_numeric_list(value)
        if len(value) > 50:
            return {
                "count": len(value),
                "sample": [_strip_modeling_transport_heavy_values(child, key) for child in value[:10]],
            }
        return [_strip_modeling_transport_heavy_values(child, key) for child in value]
    if isinstance(value, str) and len(value) > 4000:
        return {
            "stripped": True,
            "chars": len(value),
            "reason": "large string removed before report transport",
        }
    return value


def _modeling_result_for_transport(result_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned = _strip_modeling_transport_heavy_values(
        _clean_modeling_transport_value(result_dict)
    )
    try:
        return json.loads(json.dumps(cleaned, ensure_ascii=False, default=str))
    except Exception:
        return {
            "serialization_error": "result_dict could not be serialized for workflow transport"
        }


def _run_modeling_code(
    *,
    code: str,
    data: str,
    timeout_seconds: int = 300,
    n_rows: int | None = None,
) -> dict[str, Any]:
    """
    执行建模训练代码。
    约定用户代码必须设置 result_dict 变量，与前端执行器保持一致。
    """
    user_code = to_str(code).strip()
    if not user_code:
        return {"is_success": False, "error": "空代码", "stdout": "", "result_json": {}}

    compatibility_issues = validate_modeling_runtime_compatibility(
        user_code,
        n_rows=infer_data_row_count(data) if n_rows is None else int(n_rows),
    )
    if compatibility_issues:
        return {
            "is_success": False,
            "error": "Modeling runtime compatibility validation failed:\n- " + "\n- ".join(
                compatibility_issues
            ),
            "stdout": "",
            "result_json": {},
        }

    try:
        dataframe = _records_json_to_dataframe(data)
    except Exception as exc:
        return {
            "is_success": False,
            "error": f"输入数据解析失败：{exc}",
            "stdout": "",
            "result_json": {},
        }

    execution_result = run_bounded_safe_exec(
        kind="modeling",
        code=user_code,
        dataframe=dataframe,
        timeout_seconds=timeout_seconds,
    )
    if not execution_result.get("is_success"):
        return {
            "is_success": False,
            "error": str(execution_result.get("error") or ""),
            "stdout": str(execution_result.get("stdout") or ""),
            "result_json": {},
        }

    result_dict = execution_result.get("value")
    if not isinstance(result_dict, dict):
        return {
            "is_success": False,
            "error": "代码必须定义 dict 类型的 result_dict",
            "stdout": str(execution_result.get("stdout") or ""),
            "result_json": {},
        }

    return {
        "is_success": True,
        "error": "",
        "stdout": str(execution_result.get("stdout") or ""),
        "result_json": _modeling_result_for_transport(result_dict),
    }


def _unwrap_code_block(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return t


# ---------- CLI 测试入口 ----------

if __name__ == "__main__":
    import sys

    import pandas as pd

    from workflows._plugins import df_to_meta

    if len(sys.argv) < 3:
        print("用法: python -m workflows.modeling <csv_path> <target_column> [user_input]")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    target = sys.argv[2]
    user_input = sys.argv[3] if len(sys.argv) > 3 else ""
    print(f"✓ 读取 {sys.argv[1]}: {df.shape}  target={target}")

    meta = df_to_meta(df)
    result = run_modeling_workflow(
        data=meta["df"],
        df_head=meta["head_dict_str"],
        columns=list(df.columns),
        modeling_auto=True,
        target=target,
        user_input=user_input,
    )

    print("\n===== model_suggestion =====")
    print(result["model_suggestion"][:500])
    print("\n===== summary_4.desc =====")
    print(result["summary_4"]["desc"][:500])
    print("\n===== summary_4.result =====")
    print(result["summary_4"]["result"][:500])
    print("\n===== abstract_4 =====")
    print(result["abstract_4"])
