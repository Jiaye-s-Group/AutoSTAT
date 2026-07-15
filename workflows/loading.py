"""
Loading workflow 本地实现。

Workflow stages:
    Start → Condition(loading_auto==True) ─┬─► do_data_description(LLM)
                                           │       ├─► ABS1_check_abstract(LLM) ─► abstract_1
                                           │       └─► CHAP1_summary_html(LLM) ─► summary1_composer ─► summary_1
                                           └─► (else) fall through to Code with empty values
    Code(兜底) → End {summary_1, abstract_1}

输出:
    {
      "summary_1": {"title": "...", "desc": "...", "df": "..."},
      "abstract_1": "一段式摘要..."
    }
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.llm_client import chat, submit_with_context
from core.prompt_template import render_file
from core.report_language import (
    app_language_instruction,
    app_language_name,
    app_language_ref_context_empty,
    is_english_language,
    normalize_app_language,
)
from core.suggestion_revision import revise_suggestion
from workflows._plugins import summary1_composer


def run_loading_workflow(
    *,
    shape_0: int,
    shape_1: int,
    dtype_info_str: str,
    head_dict_str: str,
    loading_auto: bool = True,
    user_input: str = "",
    add_preference: str = "",
    preference_selected: str = "",
    ref_context: str = "",
    language: str = "zh",
) -> dict[str, Any]:
    # ---------- Condition: loading_auto ----------
    if not loading_auto:
        return {
            "summary_1": {"title": "", "desc": "", "df": ""},
            "abstract_1": "",
        }

    language = normalize_app_language(language)
    ctx = {
        "shape_0": shape_0,
        "shape_1": shape_1,
        "dtype_info_str": dtype_info_str,
        "head_dict_str": head_dict_str,
        "user_input": user_input or "",
        "add_preference": add_preference or "",
        "preference_selected": preference_selected or "",
        "ref_context": ref_context or app_language_ref_context_empty(language),
        "language": language,
        "language_name": app_language_name(language),
        "language_instruction": app_language_instruction(language),
    }

    # 三个 LLM call 都只依赖 ctx 中的原始元信息，互不依赖 → 全并行
    desc_sys = render_file("loading/do_data_description__llm_sys.txt", ctx)
    desc_user = render_file("loading/do_data_description__llm_user.txt", ctx)
    chap_sys = render_file("loading/chap1_summary_html_llm_sys.txt", ctx)
    chap_user = render_file("loading/chap1_summary_html_llm_user.txt", ctx)
    abs_sys = render_file("loading/abs1_check_abstract_llm_sys.txt", ctx)
    abs_user = render_file("loading/abs1_check_abstract_llm_user.txt", ctx)

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_desc = submit_with_context(pool, chat, desc_sys, desc_user, name="loading.do_data_description")
        f_chap = submit_with_context(pool, chat, chap_sys, chap_user, name="loading.chap1_summary_html")
        f_abs = submit_with_context(pool, chat, abs_sys, abs_user, name="loading.abs1_check_abstract")
        description = f_desc.result()
        chap_desc = f_chap.result()
        abstract_1 = f_abs.result().strip()

    # ---------- summary1_composer ----------
    composed = summary1_composer(desc=chap_desc, head_dict_str=head_dict_str)
    summary_1 = composed["summary_1"]
    fallback_title = (
        "Data Overview and Field Meaning Analysis"
        if is_english_language(language)
        else "数据概览与数据含义分析"
    )

    return {
        "summary_1": {
            "title": fallback_title if is_english_language(language) else summary_1.get("title") or fallback_title,
            "desc": summary_1.get("desc") or "",
            "df": summary_1.get("df") or "",
        },
        "abstract_1": abstract_1 or "",
        "_description": description,
    }


def revise_loading_result(
    *,
    current_result: dict[str, Any],
    original_requirements: str,
    revision_instruction: str,
    language_instruction: str = "",
) -> dict[str, Any]:
    summary = dict(current_result.get("summary_1") or {})
    current_text = str(summary.get("desc") or current_result.get("_description") or "")
    revised = revise_suggestion(
        stage_label="data_understanding",
        original_requirements=original_requirements,
        current_suggestion=current_text,
        revision_instruction=revision_instruction,
        hard_constraints="Keep the dataset schema and observed values unchanged.",
        language_instruction=language_instruction,
    )
    summary["desc"] = revised
    return {
        **current_result,
        "summary_1": summary,
        "abstract_1": revised,
        "_description": revised,
    }


# ---------- CLI 测试入口 ----------

if __name__ == "__main__":
    import sys

    import pandas as pd

    from workflows._plugins import df_to_meta

    if len(sys.argv) < 2:
        print("用法: python -m workflows.loading <csv_path> [user_input]")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    user_input = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"✓ 读取 {sys.argv[1]}: {df.shape}")

    meta = df_to_meta(df)
    result = run_loading_workflow(
        shape_0=meta["shape_0"],
        shape_1=meta["shape_1"],
        dtype_info_str=meta["dtype_info_str"],
        head_dict_str=meta["head_dict_str"],
        loading_auto=True,
        user_input=user_input,
    )

    print("\n===== summary_1.title =====")
    print(result["summary_1"]["title"])
    print("\n===== summary_1.desc =====")
    print(result["summary_1"]["desc"])
    print("\n===== abstract_1 =====")
    print(result["abstract_1"])
