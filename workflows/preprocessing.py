"""
Preprocessing workflow 本地实现。

原 Coze 流程（简化）：
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

from core.llm_client import chat
from core.prompt_template import render_file
from core.rag_retriever import retrieve
from core.workflow_runner import to_str
from workflows._plugins import (
    get_preprocessing_suggestions,
    format_recall,
    code_runner,
    final_list,
    summary2_composer,
)

MAX_FIX_ATTEMPTS = 5


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
) -> dict[str, Any]:
    # ---------- Condition: prep_auto ----------
    if not prep_auto:
        return {
            "summary_2": {"title": "", "desc": "", "processed_df": to_str(df), "code": ""},
            "abstract_2": "",
            "suggestion": "",
        }

    # ---------- 节点 1: get_preprocessing_suggestions (plugin) ----------
    stats = get_preprocessing_suggestions(df=df)
    if not stats["is_success"]:
        return {
            "summary_2": {"title": "", "desc": "", "processed_df": to_str(df), "code": ""},
            "abstract_2": "",
            "suggestion": "",
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
        "ref_context": ref_context or "（无参考资料）",
        "n_rows": stats["n_rows"],
        "n_cols": stats["n_cols"],
        "dtype_counts": stats["dtype_counts"],
        "missing_total": stats["missing_total"],
        "missing_by_col": stats["missing_by_col"],
        "num_cols": stats["num_cols"],
        "columns": stats["columns"],
    }

    # ---------- 节点 2: get_preprocessing_suggestions2 (LLM) ----------
    sug_sys = render_file("preprocessing/get_preprocessing_suggestions2_llm_sys.txt", ctx)
    sug_user = render_file("preprocessing/get_preprocessing_suggestions2_llm_user.txt", ctx)
    suggestion = chat(sug_sys, sug_user, name="prep.get_suggestions").strip()
    ctx["suggestion"] = suggestion

    # ---------- 节点 3: refine_suggestions (LLM) ----------
    ref_sys = render_file("preprocessing/refine_suggestions_llm_sys.txt", ctx)
    ref_user = render_file("preprocessing/refine_suggestions_llm_user.txt", ctx)
    refined_suggestions = chat(ref_sys, ref_user, name="prep.refine").strip()
    ctx["refined_suggestions"] = refined_suggestions

    # ---------- 节点 4: RAG ----------
    #   get_query(LLM) → 生成检索 query
    q_sys = render_file("preprocessing/get_query_llm_sys.txt", ctx)
    q_user = render_file("preprocessing/get_query_llm_user.txt", ctx)
    rag_query = chat(q_sys, q_user, name="prep.get_query", temperature=0).strip()

    #   Knowledge retrieval + format_recall
    recall_results = retrieve(rag_query, top_k=3)
    ctx["knowledge_results"] = format_recall(output_list=recall_results)["knowledge_results"]

    # ---------- 节点 5: code_generation (LLM) ----------
    cg_sys = render_file("preprocessing/code_generation_llm_sys.txt", ctx)
    cg_user = render_file("preprocessing/code_generation_llm_user.txt", ctx)
    initial_code = chat(cg_sys, cg_user, name="prep.code_generation").strip()
    initial_code = _unwrap_code_block(initial_code)

    # ---------- 节点 6: Loop (修复循环，最多 5 次) ----------
    current_code = initial_code
    processed_df_list: list[str] = []
    processed_df_head_list: list[str] = []
    last_error = ""
    success = False

    for attempt in range(MAX_FIX_ATTEMPTS):
        run_result = code_runner(code=current_code, df=df)

        if run_result["is_success"]:
            processed_df_list.append(run_result["processed_df"])
            processed_df_head_list.append(run_result["processed_df_head"])
            success = True
            break

        last_error = run_result.get("error", "")
        if attempt >= MAX_FIX_ATTEMPTS - 1:
            break

        # ---------- 节点 6.x: Code_Fixer LLM ----------
        # 截断长字段，防止 prompt 超出 LLM 上下文窗口
        MAX_ERR_CHARS = 8000
        MAX_CODE_CHARS = 30000
        _err_trunc = last_error[-MAX_ERR_CHARS:] if len(last_error) > MAX_ERR_CHARS else last_error
        _code_trunc = current_code[:MAX_CODE_CHARS] if len(current_code) > MAX_CODE_CHARS else current_code
        fix_ctx = {
            **ctx,
            "code": _code_trunc,
            "code_prep": _code_trunc,
            "error": _err_trunc,
            "error_msg": _err_trunc,
        }
        fix_sys = render_file("preprocessing/code_fixer_llm_sys.txt", fix_ctx)
        fix_user = render_file("preprocessing/code_fixer_llm_user.txt", fix_ctx)
        fixed = chat(
            fix_sys, fix_user, name=f"prep.code_fixer.{attempt+1}", temperature=0.3
        ).strip()
        fixed = _unwrap_code_block(fixed)
        if fixed:
            current_code = fixed

    # ---------- 节点 7: final_list (plugin) ----------
    if processed_df_list:
        final = final_list(
            processed_df_head_list=processed_df_head_list,
            processed_df_list=processed_df_list,
        )
        processed_df = final["processed_df"]
        processed_df_head = final["processed_df_head"]
    else:
        # 全部尝试都失败：退化为原始 df
        processed_df = df
        processed_df_head = head_dict_str

    ctx["code"] = current_code
    ctx["processed_df"] = processed_df
    ctx["processed_df_head"] = processed_df_head

    # ---------- 节点 8 & 9: 章节正文 + 摘要 并行 ----------
    chap_sys = render_file("preprocessing/chap2_summary_html_llm_sys.txt", ctx)
    chap_user = render_file("preprocessing/chap2_summary_html_llm_user.txt", ctx)
    abs_sys = render_file("preprocessing/abs2_check_abstract_llm_sys.txt", ctx)
    abs_user = render_file("preprocessing/abs2_check_abstract_llm_user.txt", ctx)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_desc = pool.submit(chat, chap_sys, chap_user, name="prep.chap2_summary_html")
        f_abs = pool.submit(chat, abs_sys, abs_user, name="prep.abs2_check_abstract")
        desc = f_desc.result().strip()
        abstract_2 = f_abs.result().strip()

    # ---------- 节点 10: summary2_composer (plugin) ----------
    composed = summary2_composer(code=current_code, desc=desc, processed_df=processed_df)

    return {
        "summary_2": composed["summary_2"],
        "abstract_2": abstract_2,
        "suggestion": suggestion,
        # 额外字段（给前端展示）
        "_refined_suggestions": refined_suggestions,
        "_rag_query": rag_query,
        "_code_success": success,
        "_code_error": last_error if not success else "",
        "_fix_attempts": len(processed_df_list) + (0 if success else MAX_FIX_ATTEMPTS),
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
