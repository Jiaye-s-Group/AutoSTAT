"""
Planning workflow 本地实现。

Workflow stages:
    Start(data, preferences)
      → Data_to_url(Code) → Loading_Data(plugin) → planner(LLM) → analysis_path(LLM) → End

本地流程（输入直接是 DataFrame）：
    df → df_to_meta → planner(LLM, JSON) → analysis_path(LLM) → 返回

Returns a stable workflow payload:
    {
      loading_auto, prep_auto, vis_auto, modeling_auto, report_auto,
      plan,
      shape_0, shape_1, dtype_info_str, head_dict_str, df
    }
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.llm_client import chat, chat_json
from core.planning_contract import DEFAULT_STAGE_PLAN
from core.prompt_template import render_file
from core.report_language import (
    app_language_instruction,
    app_language_name,
    app_language_ref_context_empty,
    is_english_language,
    normalize_app_language,
)
from core.workflow_runner import as_bool
from workflows._plugins import df_to_meta


def run_planning_workflow(
    *,
    df: pd.DataFrame,
    add_preference: str = "",
    preference_selected: str = "",
    ref_context: str = "",
    language: str = "zh",
) -> dict[str, Any]:
    """
    Run Planning: 根据 df 元信息让 LLM 决定 5 个阶段开关 + 生成分析路径说明。
    """
    language = normalize_app_language(language)

    # ---------- 节点 1: df_to_meta（替代 Data_to_url + Loading_Data） ----------
    meta = df_to_meta(df)
    if not meta["is_success"]:
        error_text = (
            f"Data loading failed: {meta['error']}"
            if is_english_language(language)
            else f"数据加载失败：{meta['error']}"
        )
        return {
            "loading_auto": False,
            "prep_auto": False,
            "vis_auto": False,
            "modeling_auto": False,
            "report_auto": False,
            "plan": error_text,
            "shape_0": 0,
            "shape_1": 0,
            "dtype_info_str": "",
            "head_dict_str": "",
            "df": "",
        }

    ctx = {
        "shape_0": meta["shape_0"],
        "shape_1": meta["shape_1"],
        "dtype_info_str": meta["dtype_info_str"],
        "head_dict_str": meta["head_dict_str"],
        "add_preference": add_preference or "",
        "preference_selected": preference_selected or "",
        # analysis_path prompt 里写的是 preference_select（少了 ed）
        "preference_select": preference_selected or "",
        "ref_context": ref_context or app_language_ref_context_empty(language),
        "language": language,
        "language_name": app_language_name(language),
        "language_instruction": app_language_instruction(language),
    }

    # ---------- 节点 2: planner LLM（输出 5 个开关 JSON） ----------
    planner_sys = render_file("planning/planner_llm_sys.txt", ctx)
    planner_user = render_file("planning/planner_llm_user.txt", ctx)
    planner_result = chat_json(
        planner_sys,
        planner_user,
        name="planning.planner",
        temperature=1.0,
    )

    loading_auto = as_bool(planner_result.get("loading_auto"), default=DEFAULT_STAGE_PLAN["loading_auto"])
    prep_auto = as_bool(planner_result.get("prep_auto"), default=DEFAULT_STAGE_PLAN["prep_auto"])
    vis_auto = as_bool(planner_result.get("vis_auto"), default=DEFAULT_STAGE_PLAN["vis_auto"])
    modeling_auto = as_bool(planner_result.get("modeling_auto"), default=DEFAULT_STAGE_PLAN["modeling_auto"])
    report_auto = as_bool(planner_result.get("report_auto"), default=DEFAULT_STAGE_PLAN["report_auto"])

    # ---------- 节点 3: analysis_path LLM（写一段分析路径说明） ----------
    ctx.update(
        {
            "loading_auto": loading_auto,
            "prep_auto": prep_auto,
            "vis_auto": vis_auto,
            "modeling_auto": modeling_auto,
            "report_auto": report_auto,
        }
    )
    path_sys = render_file("planning/analysis_path_llm_sys.txt", ctx)
    path_user = render_file("planning/analysis_path_llm_user.txt", ctx)
    plan_text = chat(
        path_sys,
        path_user,
        name="planning.analysis_path",
        temperature=0.5,
        max_tokens=8192,
    )

    return {
        "loading_auto": loading_auto,
        "prep_auto": prep_auto,
        "vis_auto": vis_auto,
        "modeling_auto": modeling_auto,
        "report_auto": report_auto,
        "plan": plan_text.strip(),
        "shape_0": meta["shape_0"],
        "shape_1": meta["shape_1"],
        "dtype_info_str": meta["dtype_info_str"],
        "head_dict_str": meta["head_dict_str"],
        "df": meta["df"],
    }


# ---------- CLI 测试入口 ----------

if __name__ == "__main__":
    import sys

    import pandas as pd

    if len(sys.argv) < 2:
        print("用法: python -m workflows.planning <csv_path>")
        print("示例: python -m workflows.planning data/iris.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    user_input = sys.argv[2] if len(sys.argv) > 2 else ""
    df = pd.read_csv(csv_path)
    print(f"✓ 读取 {csv_path}: {df.shape}")

    result = run_planning_workflow(df=df, add_preference=user_input)

    print("\n===== Planning 结果 =====")
    print(f"loading_auto:  {result['loading_auto']}")
    print(f"prep_auto:     {result['prep_auto']}")
    print(f"vis_auto:      {result['vis_auto']}")
    print(f"modeling_auto: {result['modeling_auto']}")
    print(f"report_auto:   {result['report_auto']}")
    print("\n----- plan -----")
    print(result["plan"])
