"""
Preprocessing workflow 本地实现。

Workflow stages:
    Start → Condition(prep_auto==True)
      → get_preprocessing_suggestions(plugin)              [统计 df 基本信息]
      → get_preprocessing_suggestions2(LLM)                [生成初版建议]
      → refine_suggestions(LLM)                            [精炼建议]
      → get_query(LLM) → Knowledge(RAG) → format_recall    [召回相关算法]
      → code_generation(LLM)                               [生成预处理代码]
      → Variable assign: code_prep = code
      → Loop(max 5):                                       [修复循环]
          ├─ code_runner(plugin)      [执行]
          ├─ if success: break
          └─ Code_Fixer(LLM)          [修]
      → final_list(plugin)                                 [取最后一次结果]
      → CHAP2_summary_html(LLM)                            [章节正文]
      → ABS2_check_abstract(LLM)                           [摘要]
      → summary2_composer(plugin)                          [组装]
      → Code(兜底) → End

输出:
    {
      "summary_2": {title, desc, processed_df, code},
      "abstract_2": "...",
      "suggestion": "初版建议"  # 给前端展示
    }
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.llm_client import (
    LLMOutputIncompleteError,
    LLMOutputTruncatedError,
    chat,
    chat_code,
    chat_suggestion,
    submit_with_context,
)
from core.code_runtime_profile import build_code_runtime_constraints
from core.prompt_template import render_file
from core.preprocessing_contract import (
    build_preprocessing_contract,
    contract_as_prompt as preprocessing_contract_as_prompt,
)
from core.rag_retriever import retrieve
from core.report_language import (
    app_language_instruction,
    app_language_name,
    app_language_ref_context_empty,
    is_english_language,
    normalize_app_language,
)
from core.suggestion_revision import normalize_suggestion_output, revise_suggestion
from core.workflow_runner import to_str
from workflows._plugins import (
    get_preprocessing_suggestions,
    format_recall,
    code_runner,
    final_list,
    summary2_composer,
)

MAX_FIX_ATTEMPTS = 5


def ensure_preprocessing_contract(ctx: dict[str, Any]) -> dict[str, Any]:
    """Backfill a contract for contexts created before semantic validation existed."""
    working_ctx = dict(ctx)
    if isinstance(working_ctx.get("preprocessing_contract"), dict):
        working_ctx.setdefault(
            "preprocessing_contract_json",
            preprocessing_contract_as_prompt(working_ctx["preprocessing_contract"]),
        )
        return working_ctx
    contract = build_preprocessing_contract(
        columns=list(working_ctx.get("columns") or []),
        user_input=str(working_ctx.get("user_input") or ""),
        add_preference=str(working_ctx.get("add_preference") or ""),
        suggestion=str(working_ctx.get("suggestion") or ""),
        refined_suggestions=str(working_ctx.get("refined_suggestions") or ""),
    )
    working_ctx["preprocessing_contract"] = contract
    working_ctx["preprocessing_contract_json"] = preprocessing_contract_as_prompt(contract)
    return working_ctx


def run_preprocessing_phase1(
    *,
    df: str,
    shape_0: int,
    shape_1: int,
    dtype_info_str: str,
    head_dict_str: str,
    prep_auto: bool = True,
    user_input: str = "",
    add_preference: str = "",
    preference_selected: str = "",
    ref_context: str = "",
    language: str = "zh",
) -> dict[str, Any]:
    if not prep_auto:
        return {"suggestion": "", "refined_suggestions": "", "_ctx": {}}

    language = normalize_app_language(language)
    stats = get_preprocessing_suggestions(df=df)
    if not stats["is_success"]:
        return {
            "suggestion": "",
            "refined_suggestions": "",
            "_ctx": {},
            "_status": "failed",
            "_error": str(stats.get("error") or "Unable to inspect the dataset."),
        }

    ctx: dict[str, Any] = {
        "shape_0": shape_0,
        "shape_1": shape_1,
        "dtype_info_str": dtype_info_str,
        "head_dict_str": head_dict_str,
        "df": df,
        "user_input": user_input or "",
        "add_preference": add_preference or "",
        "preference_selected": preference_selected or "",
        "ref_context": ref_context or app_language_ref_context_empty(language),
        "language": language,
        "language_name": app_language_name(language),
        "language_instruction": app_language_instruction(language),
        "n_rows": stats["n_rows"],
        "n_cols": stats["n_cols"],
        "dtype_counts": stats["dtype_counts"],
        "missing_total": stats["missing_total"],
        "missing_by_col": stats["missing_by_col"],
        "num_cols": stats["num_cols"],
        "columns": stats["columns"],
        "runtime_constraints_json": build_code_runtime_constraints(
            df,
            fallback_rows=shape_0,
            fallback_columns=shape_1,
        ),
    }
    try:
        sug_sys = render_file("preprocessing/get_preprocessing_suggestions2_llm_sys.txt", ctx)
        sug_user = render_file("preprocessing/get_preprocessing_suggestions2_llm_user.txt", ctx)
        suggestion = normalize_suggestion_output(
            chat_suggestion(sug_sys, sug_user, name="prep.get_suggestions")
        )
        ctx["suggestion"] = suggestion

        ref_sys = render_file("preprocessing/refine_suggestions_llm_sys.txt", ctx)
        ref_user = render_file("preprocessing/refine_suggestions_llm_user.txt", ctx)
        refined = normalize_suggestion_output(
            chat_suggestion(ref_sys, ref_user, name="prep.refine")
        )
    except (LLMOutputIncompleteError, LLMOutputTruncatedError) as exc:
        return {
            "suggestion": "",
            "refined_suggestions": "",
            "_ctx": ctx,
            "_status": "suggestion_incomplete",
            "_error": str(exc),
        }
    ctx["refined_suggestions"] = refined
    contract = build_preprocessing_contract(
        columns=list(ctx.get("columns") or []),
        user_input=str(ctx.get("user_input") or ""),
        add_preference=str(ctx.get("add_preference") or ""),
        suggestion=suggestion,
        refined_suggestions=refined,
    )
    ctx["preprocessing_contract"] = contract
    ctx["preprocessing_contract_json"] = preprocessing_contract_as_prompt(contract)
    return {"suggestion": suggestion, "refined_suggestions": refined, "_ctx": ctx}


def revise_preprocessing_phase1(
    *,
    ctx: dict[str, Any],
    original_requirements: str,
    revision_instruction: str,
) -> dict[str, Any]:
    revised_ctx = dict(ctx)
    try:
        revised = revise_suggestion(
            stage_label="preprocessing",
            original_requirements=original_requirements,
            current_suggestion=str(ctx.get("suggestion") or ""),
            revision_instruction=revision_instruction,
            hard_constraints=f"Available columns: {ctx.get('columns', [])}",
            language_instruction=str(ctx.get("language_instruction") or ""),
        )
        revised_ctx["suggestion"] = revised
        ref_sys = render_file("preprocessing/refine_suggestions_llm_sys.txt", revised_ctx)
        ref_user = render_file("preprocessing/refine_suggestions_llm_user.txt", revised_ctx)
        refined = normalize_suggestion_output(
            chat_suggestion(ref_sys, ref_user, name="prep.refine_revision")
        )
    except (LLMOutputIncompleteError, LLMOutputTruncatedError) as exc:
        return {
            "suggestion": "",
            "refined_suggestions": "",
            "_ctx": revised_ctx,
            "_status": "suggestion_incomplete",
            "_error": str(exc),
        }
    revised_ctx["refined_suggestions"] = refined
    contract = build_preprocessing_contract(
        columns=list(revised_ctx.get("columns") or []),
        user_input=str(revised_ctx.get("user_input") or original_requirements or ""),
        add_preference=str(revised_ctx.get("add_preference") or ""),
        suggestion=revised,
        refined_suggestions=refined,
    )
    revised_ctx["preprocessing_contract"] = contract
    revised_ctx["preprocessing_contract_json"] = preprocessing_contract_as_prompt(contract)
    return {"suggestion": revised, "refined_suggestions": refined, "_ctx": revised_ctx}


def generate_preprocessing_code(*, ctx: dict[str, Any]) -> dict[str, Any]:
    """Generate a preprocessing code draft without executing user data."""
    working_ctx = ensure_preprocessing_contract(ctx)
    q_sys = render_file("preprocessing/get_query_llm_sys.txt", working_ctx)
    q_user = render_file("preprocessing/get_query_llm_user.txt", working_ctx)
    rag_query = chat(q_sys, q_user, name="prep.get_query", temperature=0).strip()
    working_ctx["rag_query"] = rag_query
    recall_results = retrieve(rag_query, top_k=3)
    working_ctx["knowledge_results"] = format_recall(output_list=recall_results)["knowledge_results"]

    cg_sys = render_file("preprocessing/code_generation_llm_sys.txt", working_ctx)
    cg_user = render_file("preprocessing/code_generation_llm_user.txt", working_ctx)
    code = _unwrap_code_block(
        chat_code(cg_sys, cg_user, name="prep.code_generation").strip()
    )
    return {"code": code, "_ctx": working_ctx, "_rag_query": rag_query}


def repair_preprocessing_code(*, ctx: dict[str, Any], code: str, error: str) -> str:
    contract_ctx = ensure_preprocessing_contract(ctx)
    fix_ctx = {
        **contract_ctx,
        "code": code,
        "code_prep": code,
        "error": error,
        "error_msg": error,
        "suggestion": str(contract_ctx.get("suggestion") or ""),
        "preprocessing_contract_json": str(contract_ctx.get("preprocessing_contract_json") or "{}"),
    }
    fix_sys = render_file("preprocessing/code_fixer_llm_sys.txt", fix_ctx, strict=True)
    fix_user = render_file("preprocessing/code_fixer_llm_user.txt", fix_ctx, strict=True)
    return _unwrap_code_block(
        chat_code(fix_sys, fix_user, name="prep.manual_code_fixer", temperature=0.3).strip()
    )


def validate_preprocessing_code(
    *,
    ctx: dict[str, Any],
    df: str,
    initial_code: str = "",
) -> dict[str, Any]:
    """Run the shared legacy five-attempt generation/fix loop without publishing results."""
    working_ctx = ensure_preprocessing_contract(ctx)
    if initial_code:
        current_code = _unwrap_code_block(initial_code)
    else:
        generation = generate_preprocessing_code(ctx=working_ctx)
        working_ctx = generation["_ctx"]
        current_code = generation["code"]

    processed_df = ""
    processed_df_head = ""
    last_error = ""
    success = False
    attempts = 0
    for attempt in range(MAX_FIX_ATTEMPTS):
        attempts = attempt + 1
        run_result = code_runner(
            code=current_code,
            df=df,
            preprocessing_contract=working_ctx.get("preprocessing_contract") or {},
        )
        if run_result["is_success"]:
            processed_df = run_result["processed_df"]
            processed_df_head = run_result["processed_df_head"]
            success = True
            break

        last_error = run_result.get("error", "")
        if attempt >= MAX_FIX_ATTEMPTS - 1:
            break
        fixed = repair_preprocessing_code(ctx=working_ctx, code=current_code, error=last_error)
        if fixed:
            current_code = fixed

    return {
        "code": current_code,
        "processed_df": processed_df,
        "processed_df_head": processed_df_head,
        "qc_summary": run_result.get("qc_summary", {}) if success else {},
        "error": last_error,
        "success": success,
        "attempts": attempts,
        "_ctx": working_ctx,
    }


def run_preprocessing_workflow(
    *,
    df: str,
    shape_0: int,
    shape_1: int,
    dtype_info_str: str,
    head_dict_str: str,
    prep_auto: bool = True,
    user_input: str = "",
    add_preference: str = "",
    preference_selected: str = "",
    ref_context: str = "",
    language: str = "zh",
    phase1_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # ---------- Condition: prep_auto ----------
    if not prep_auto:
        return {
            "summary_2": {"title": "", "desc": "", "processed_df": to_str(df), "code": ""},
            "abstract_2": "",
            "suggestion": "",
        }

    if phase1_ctx:
        ctx = dict(phase1_ctx)
        phase1_status = ""
        phase1_error = ""
    else:
        phase1 = run_preprocessing_phase1(
            df=df,
            shape_0=shape_0,
            shape_1=shape_1,
            dtype_info_str=dtype_info_str,
            head_dict_str=head_dict_str,
            prep_auto=prep_auto,
            user_input=user_input,
            add_preference=add_preference,
            preference_selected=preference_selected,
            ref_context=ref_context,
            language=language,
        )
        phase1_status = str(phase1.get("_status") or "")
        phase1_error = str(phase1.get("_error") or "")
        ctx = dict(phase1.get("_ctx") or {})
    if phase1_status == "suggestion_incomplete":
        english = is_english_language(language)
        desc = (
            f"Preprocessing suggestion generation did not complete: {phase1_error}"
            if english
            else f"预处理建议生成未完成：{phase1_error}"
        )
        return {
            "summary_2": {"title": "Preprocessing" if english else "数据预处理", "desc": desc, "code": "", "status": "failed"},
            "abstract_2": desc,
            "suggestion": "",
            "_status": "suggestion_incomplete",
            "_code_success": False,
            "_error": phase1_error,
        }
    if not ctx:
        return {
            "summary_2": {"title": "数据预处理", "desc": "预处理建议生成失败。", "code": "", "status": "failed"},
            "abstract_2": "预处理建议生成失败。",
            "suggestion": "",
            "_status": "failed",
            "_code_success": False,
        }
    language = normalize_app_language(str(ctx.get("language") or language))
    suggestion = str(ctx.get("suggestion") or "")
    refined_suggestions = str(ctx.get("refined_suggestions") or "")

    # ---------- 节点 4-6: RAG + code_generation + legacy fix loop ----------
    validation = validate_preprocessing_code(ctx=ctx, df=df)
    ctx = validation["_ctx"]
    current_code = validation["code"]
    last_error = validation["error"]
    success = bool(validation["success"])
    attempts = int(validation["attempts"])
    rag_query = str(ctx.get("rag_query") or "")

    # ---------- 节点 7: final_list (plugin) ----------
    if success:
        final = final_list(
            processed_df_head_list=[validation["processed_df_head"]],
            processed_df_list=[validation["processed_df"]],
        )
        processed_df = final["processed_df"]
        processed_df_head = final["processed_df_head"]
    else:
        english = is_english_language(language)
        error_text = last_error or (
            "Unknown preprocessing execution error"
            if english
            else "未知预处理执行错误"
        )
        return {
            "summary_2": {
                "title": "Data Preprocessing" if english else "数据预处理",
                "desc": (
                    f"Preprocessing failed and no processed dataset was produced: {error_text[:500]}"
                    if english
                    else f"预处理执行失败，未生成处理后数据：{error_text[:500]}"
                ),
                "code": current_code,
                "status": "failed",
                "error": error_text,
            },
            "abstract_2": (
                f"Preprocessing failed: {error_text[:200]}"
                if english
                else f"预处理失败：{error_text[:200]}"
            ),
            "suggestion": suggestion,
            "_status": "failed",
            "_refined_suggestions": refined_suggestions,
            "_rag_query": rag_query,
            "_code_success": False,
            "_code_error": error_text,
            "_fix_attempts": attempts,
        }

    ctx["code"] = current_code
    ctx["processed_df"] = processed_df
    ctx["processed_df_head"] = processed_df_head
    ctx["qc_summary"] = validation.get("qc_summary", {})

    # ---------- 节点 8 & 9: 章节正文 + 摘要 并行 ----------
    chap_sys = render_file("preprocessing/chap2_summary_html_llm_sys.txt", ctx)
    chap_user = render_file("preprocessing/chap2_summary_html_llm_user.txt", ctx)
    abs_sys = render_file("preprocessing/abs2_check_abstract_llm_sys.txt", ctx)
    abs_user = render_file("preprocessing/abs2_check_abstract_llm_user.txt", ctx)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_desc = submit_with_context(pool, chat, chap_sys, chap_user, name="prep.chap2_summary_html")
        f_abs = submit_with_context(pool, chat, abs_sys, abs_user, name="prep.abs2_check_abstract")
        desc = f_desc.result().strip()
        abstract_2 = f_abs.result().strip()

    # ---------- 节点 10: summary2_composer (plugin) ----------
    composed = summary2_composer(code=current_code, desc=desc, processed_df=processed_df)
    composed["summary_2"]["qc_summary"] = validation.get("qc_summary", {})
    if is_english_language(language):
        composed["summary_2"]["title"] = "Data Preprocessing"

    return {
        "summary_2": composed["summary_2"],
        "abstract_2": abstract_2,
        "suggestion": suggestion,
        # 额外字段（给前端展示）
        "_refined_suggestions": refined_suggestions,
        "_rag_query": rag_query,
        "_code_success": success,
        "_code_error": last_error if not success else "",
        "_fix_attempts": attempts,
        "_preprocessing_contract": ctx.get("preprocessing_contract") or {},
        "_status": "succeeded",
    }


def _unwrap_code_block(text: str) -> str:
    """
    LLM 经常把代码包在 ```python ... ``` 里，去掉这层包装。
    """
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```"):
        # 去掉首行 ``` 或 ```python
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

    if len(sys.argv) < 2:
        print("用法: python -m workflows.preprocessing <csv_path> [user_input]")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    user_input = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"✓ 读取 {sys.argv[1]}: {df.shape}")

    meta = df_to_meta(df)
    result = run_preprocessing_workflow(
        df=meta["df"],
        shape_0=meta["shape_0"],
        shape_1=meta["shape_1"],
        dtype_info_str=meta["dtype_info_str"],
        head_dict_str=meta["head_dict_str"],
        prep_auto=True,
        user_input=user_input,
    )

    print("\n===== suggestion =====")
    print(result["suggestion"])
    print("\n===== summary_2.desc (章节正文) =====")
    print(result["summary_2"]["desc"])
    print("\n===== summary_2.code =====")
    print(result["summary_2"]["code"])
    print("\n===== abstract_2 =====")
    print(result["abstract_2"])
    print(f"\n===== 代码执行：{'✅ 成功' if result['_code_success'] else '❌ 失败'} =====")
    if not result["_code_success"]:
        print(f"错误: {result['_code_error']}")
