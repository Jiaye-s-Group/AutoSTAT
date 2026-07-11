"""Generate modeling recommendations, executable code, metrics, and summaries."""
from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.llm_client import chat
from core.modeling_table_utils import (
    build_model_comparison_table_bundle,
)
from core.prompt_template import render_file
from core.rag_retriever import retrieve
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
    "coefficient",
    "coefficients",
    "coef",
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
) -> dict[str, Any]:
    """Build shared prompt context for modeling steps."""
    prompt_columns = _compact_columns_for_prompt(columns, target=target)
    return {
        "data": _truncate_prompt_text(data, MODELING_PROMPT_DATA_MAX_CHARS),
        "df_head": _compact_df_head_for_prompt(
            df_head,
            prompt_columns=prompt_columns,
            max_chars=MODELING_PROMPT_HEAD_MAX_CHARS,
        ),
        "columns": prompt_columns,
        "target": target or "",
        "train_code": _truncate_prompt_text(train_code or "", MODELING_CODE_PROMPT_MAX_CHARS),
        "user_input": _truncate_prompt_text(user_input or "", MODELING_PROMPT_USER_TEXT_MAX_CHARS),
        "user_prompt": _truncate_prompt_text(user_prompt or user_input or "", MODELING_PROMPT_USER_TEXT_MAX_CHARS),
        "add_preference": _truncate_prompt_text(add_preference or "", MODELING_PROMPT_USER_TEXT_MAX_CHARS),
        "additional_preference": _truncate_prompt_text(add_preference or "", MODELING_PROMPT_USER_TEXT_MAX_CHARS),
        "preference_selected": _truncate_prompt_text(preference_selected or "", MODELING_PROMPT_USER_TEXT_MAX_CHARS),
        "preference_select": _truncate_prompt_text(preference_selected or "", MODELING_PROMPT_USER_TEXT_MAX_CHARS),
        "ref_context": _truncate_prompt_text(ref_context or "（无参考资料）", MODELING_PROMPT_CONTEXT_MAX_CHARS),
    }


def _truncate_prompt_text(value: Any, max_chars: int) -> str:
    text = to_str(value).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n...[truncated for modeling prompt; original chars={len(text)}]"


def _compact_columns_for_prompt(columns: Any, *, target: str = "") -> list[str]:
    raw_columns = [str(col) for col in (columns or [])]
    if len(raw_columns) <= MODELING_PROMPT_MAX_COLUMNS:
        return raw_columns

    target_text = str(target or "").strip()
    kept: list[str] = []
    if target_text and target_text in raw_columns:
        kept.append(target_text)
    for col in raw_columns:
        if col in kept:
            continue
        kept.append(col)
        if len(kept) >= MODELING_PROMPT_MAX_COLUMNS:
            break
    kept.append(f"...[{len(raw_columns) - len(kept)} omitted columns; full df is still available at runtime]")
    return kept


def _compact_df_head_for_prompt(df_head: Any, *, prompt_columns: list[str], max_chars: int) -> str:
    text = to_str(df_head).strip()
    if len(text) <= max_chars:
        return text

    try:
        parsed = json.loads(text)
    except Exception:
        return _truncate_prompt_text(text, max_chars)

    if isinstance(parsed, dict):
        keep = {col for col in prompt_columns if not col.startswith("...[")}
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
) -> dict[str, Any]:
    """Generate modeling recommendations and concise implementation requirements."""
    if not modeling_auto:
        return {"model_suggestion": "", "refined_suggestions": "", "_ctx": {}}

    ctx = _build_modeling_ctx(
        data=data, df_head=df_head, columns=columns, target=target,
        train_code=train_code, user_input=user_input, user_prompt=user_prompt,
        add_preference=add_preference, preference_selected=preference_selected,
        ref_context=ref_context,
    )

    sug_sys = render_file("modeling/sec4_get_model_suggestion_llm_sys.txt", ctx)
    sug_user = render_file("modeling/sec4_get_model_suggestion_llm_user.txt", ctx)
    model_suggestion = chat(sug_sys, sug_user, name="model.get_suggestion").strip()
    ctx["model_suggestion"] = model_suggestion

    ref_sys = render_file("modeling/sec4_refine_suggestion_llm_sys.txt", ctx)
    ref_user = render_file("modeling/sec4_refine_suggestion_llm_user.txt", ctx)
    refined_suggestions = chat(ref_sys, ref_user, name="model.refine").strip()
    ctx["refined_suggestions"] = refined_suggestions
    ctx["refine_suggestion"] = refined_suggestions

    return {
        "model_suggestion": model_suggestion,
        "refined_suggestions": refined_suggestions,
        "_ctx": ctx,
    }


def run_modeling_phase2(
    *,
    ctx: dict[str, Any],
    data: str,
    df_head: str,
) -> dict[str, Any]:
    """Run RAG, code generation, repair, result formatting, and summaries."""
    model_suggestion = ctx.get("model_suggestion", "")
    refined_suggestions = ctx.get("refined_suggestions", "")

    # ---------- RAG ----------
    q_sys = render_file("modeling/get_query_llm_sys.txt", ctx)
    q_user = render_file("modeling/get_query_llm_user.txt", ctx)
    rag_query = chat(q_sys, q_user, name="model.get_query", temperature=0).strip()

    recall_results = retrieve(rag_query, top_k=3)
    ctx["knowledge_results"] = format_recall(output_list=recall_results)["knowledge_results"]

    # Generate modeling code from the refined requirements.
    cg_sys = render_file("modeling/sec4_code_generation_llm_sys.txt", ctx)
    cg_user = render_file("modeling/sec4_code_generation_llm_user.txt", ctx)
    generated_code = chat(cg_sys, cg_user, name="model.code_generation").strip()
    generated_code = _unwrap_code_block(generated_code)

    # Execute and repair generated code until it yields a valid metrics table.
    current_code = generated_code
    success = False
    last_error = ""
    final_result_json: dict = {}
    final_result_str = ""

    for attempt in range(MAX_FIX_ATTEMPTS):
        run_result = _run_modeling_code(code=current_code, data=data)
        if run_result["is_success"]:
            candidate_result_json = run_result.get("result_json", {})
            candidate_table = build_model_comparison_table_bundle(
                candidate_result_json,
                target=ctx.get("target", ""),
                user_input=ctx.get("user_input", ""),
                additional_preference=ctx.get("additional_preference", ""),
            )
            if candidate_table.get("has_table"):
                success = True
                final_result_str = run_result.get("stdout", "")
                final_result_json = candidate_result_json
                break

            last_error = (
                "代码执行成功，但 result_dict 无法生成模型对比表。"
                "result_dict 必须包含可解析的 models 列表；每个模型条目必须包含 name/type/metrics，"
                "metrics 中至少包含一个数值型评价指标。"
            )
            if attempt >= MAX_FIX_ATTEMPTS - 1:
                break
        else:
            last_error = run_result.get("error", "")

        if attempt >= MAX_FIX_ATTEMPTS - 1:
            break

        fix_ctx = {
            **ctx,
            "code": current_code,
            "code_modeling": current_code,
            "error_msg": last_error,
            "error": last_error,
        }
        fix_sys = render_file("modeling/sec4_code_fixed_llm_sys.txt", fix_ctx)
        fix_user = render_file("modeling/sec4_code_fixed_llm_user.txt", fix_ctx)
        fixed = chat(
            fix_sys, fix_user, name=f"model.code_fixed.{attempt+1}", temperature=0.3
        ).strip()
        fixed = _unwrap_code_block(fixed)
        if fixed:
            current_code = fixed

    final_code = current_code

    if not success:
        return {
            "summary_4": {
                "title": "建模分析",
                "desc": f"建模代码执行失败：{last_error[:500]}",
                "result": "", "code": final_code,
                "table_title": "", "table_markdown": "", "table_html": "",
            },
            "abstract_4": f"建模代码执行失败：{last_error[:200]}",
            "model_suggestion": model_suggestion,
            "_code_success": False,
            "_code_error": last_error,
            "_fix_attempts": MAX_FIX_ATTEMPTS,
        }

    # Use the raw result for deterministic tables; pass compact evidence to the LLM.
    artifact_metadata = collect_modeling_artifact_metadata(final_result_json)
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
    ctx["modeling_artifact_metadata"] = artifact_metadata
    ctx["execution_stdout"] = compact_stdout
    ctx["result"] = compact_stdout
    table_bundle = build_model_comparison_table_bundle(
        final_result_json,
        target=ctx.get("target", ""),
        user_input=ctx.get("user_input", ""),
        additional_preference=ctx.get("additional_preference", ""),
    )
    ctx["comparison_table_title"] = table_bundle.get("title", "")
    ctx["comparison_table_markdown"] = table_bundle.get("markdown_table", "")
    ctx["comparison_table_html"] = table_bundle.get("html_table", "")

    rfp_sys = render_file("modeling/sec4_result_format_prompt_llm_sys.txt", ctx)
    rfp_user = render_file("modeling/sec4_result_format_prompt_llm_user.txt", ctx)
    result_format = chat(rfp_sys, rfp_user, name="model.result_format").strip()
    ctx["result_format"] = result_format
    ctx["result"] = result_format

    # Build the report body and abstract in parallel.
    sh_sys = render_file("modeling/sec4_summary_html_llm_sys.txt", ctx)
    sh_user = render_file("modeling/sec4_summary_html_llm_user.txt", ctx)
    ab_sys = render_file("modeling/sec4_check_abstract_llm_sys.txt", ctx)
    ab_user = render_file("modeling/sec4_check_abstract_llm_user.txt", ctx)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_desc = pool.submit(chat, sh_sys, sh_user, name="model.summary_html")
        f_abs = pool.submit(chat, ab_sys, ab_user, name="model.check_abstract")
        desc = f_desc.result().strip()
        abstract_4 = f_abs.result().strip()

    composed = sec4_composer(
        code=final_code, desc=desc, result=result_format,
        table_title=table_bundle.get("title", ""),
        table_markdown=table_bundle.get("markdown_table", ""),
        table_html=table_bundle.get("html_table", ""),
    )

    return {
        "summary_4": composed["summary_4"],
        "abstract_4": abstract_4,
        "model_suggestion": model_suggestion,
        "_refined_suggestions": refined_suggestions,
        "_final_code": final_code,
        "_modeling_result_evidence": compact_result,
        "_modeling_artifact_metadata": artifact_metadata,
        "_code_success": True,
        "_code_error": "",
        "_fix_attempts": attempt + 1 if success else MAX_FIX_ATTEMPTS,
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
) -> dict[str, Any]:
    """Run modeling recommendation and code-generation phases in sequence."""
    if not modeling_auto:
        return _empty_modeling_result()

    p1 = run_modeling_phase1(
        data=data, df_head=df_head, columns=columns,
        modeling_auto=modeling_auto, target=target, train_code=train_code,
        user_input=user_input, user_prompt=user_prompt,
        add_preference=add_preference, preference_selected=preference_selected,
        ref_context=ref_context,
    )
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


# Modeling code runner with structured result capture.


def _run_modeling_code(*, code: str, data: str, timeout_seconds: int = 300) -> dict[str, Any]:
    """Run generated modeling code and collect its `result_dict` output."""
    import json
    import os
    import subprocess
    import sys
    import tempfile
    import textwrap

    user_code = to_str(code).strip()
    if not user_code:
        return {"is_success": False, "error": "空代码", "stdout": "", "result_json": {}}

    script = '''import json, sys, traceback
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder as _SklearnOneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

if not hasattr(pd.DataFrame, "concat"):
    pd.DataFrame.concat = staticmethod(pd.concat)

def pd_isna_like(value):
    return pd.isna(value)

def OneHotEncoder(*args, **kwargs):
    if "sparse" in kwargs and "sparse_output" not in kwargs:
        kwargs["sparse_output"] = kwargs.pop("sparse")
    try:
        return _SklearnOneHotEncoder(*args, **kwargs)
    except TypeError:
        if "sparse_output" in kwargs:
            kwargs["sparse"] = kwargs.pop("sparse_output")
        return _SklearnOneHotEncoder(*args, **kwargs)

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

_RECORDS = json.loads(sys.stdin.read())
df = pd.DataFrame(_RECORDS)

try:
__USER_CODE__
except Exception as e:
    print("__AUTOSTAT_ERROR__", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(2)

# Collect result_dict using the same contract as the frontend runner.
_out = locals().get("result_dict")
if not isinstance(_out, dict):
    print("__AUTOSTAT_ERROR__", file=sys.stderr)
    print("代码必须定义 dict 类型的 result_dict", file=sys.stderr)
    sys.exit(3)

# Convert numpy and pandas values into JSON-friendly Python objects.
def _clean(o):
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            return str(o)
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o

def _looks_numeric_list(values):
    if not isinstance(values, list) or not values:
        return False
    sample = values[:50]
    return all(isinstance(v, (int, float, np.integer, np.floating)) for v in sample)

def _summarize_numeric_list(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    summary = {"count": int(arr.size), "sample": values[:10]}
    if finite.size:
        summary.update({
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        })
    return summary

def _strip_transport_heavy_values(o, key=""):
    key_text = str(key).lower()
    if any(hint in key_text for hint in (
        "artifact", "b64", "base64", "pickle", "gzip", "blob", "bytes",
        "model_object", "serialized", "raw_model", "fitted_model",
    )):
        return {"stripped": True, "reason": "model artifact removed before report transport"}

    if isinstance(o, dict):
        return {str(k): _strip_transport_heavy_values(v, k) for k, v in o.items()}

    if isinstance(o, list):
        if len(o) > 50 and _looks_numeric_list(o):
            return _summarize_numeric_list(o)
        if len(o) > 50:
            return {"count": len(o), "sample": [_strip_transport_heavy_values(v, key) for v in o[:10]]}
        return [_strip_transport_heavy_values(v, key) for v in o]

    if isinstance(o, str) and len(o) > 4000:
        compact = "".join(o.split())
        if len(compact) > 4000:
            return {"stripped": True, "chars": len(o), "reason": "large string removed before report transport"}
    return o

_cleaned = _strip_transport_heavy_values(_clean(_out))
print("__AUTOSTAT_RESULT__:" + json.dumps(_cleaned, ensure_ascii=False))
'''

    indented = textwrap.indent(user_code, " " * 4)
    full_script = script.replace("__USER_CODE__", indented)

    try:
        env = dict(os.environ)
        env.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        completed = subprocess.run(
            [sys.executable, "-c", full_script],
            input=to_str(data) or "[]",
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "is_success": False,
            "error": f"代码执行超时（>{timeout_seconds}s）",
            "stdout": "",
            "result_json": {},
        }

    if completed.returncode != 0:
        return {
            "is_success": False,
            "error": (completed.stderr or "")[:6000],
            "stdout": completed.stdout or "",
            "result_json": {},
        }

    # Separate user stdout from the structured result payload.
    result_json: dict = {}
    lines = completed.stdout.splitlines()
    stdout_clean_lines = []
    for line in lines:
        if line.startswith("__AUTOSTAT_RESULT__:"):
            try:
                result_json = json.loads(line[len("__AUTOSTAT_RESULT__:"):])
            except Exception:
                result_json = {}
        else:
            stdout_clean_lines.append(line)

    return {
        "is_success": True,
        "error": "",
        "stdout": "\n".join(stdout_clean_lines),
        "result_json": result_json,
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


# CLI smoke-test entry point.

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
