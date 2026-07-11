"""Plan which AutoSTAT stages should run for a dataset."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.llm_client import chat, chat_json
from core.prompt_template import render_file
from core.workflow_runner import as_bool
from workflows._plugins import df_to_meta


def run_planning_workflow(
    *,
    df: pd.DataFrame,
    add_preference: str = "",
    preference_selected: str = "",
    ref_context: str = "",
) -> dict[str, Any]:
    """Return stage switches, dataset metadata, and a short analysis plan."""
    # Build compact dataset metadata for the planner prompt.
    meta = df_to_meta(df)
    if not meta["is_success"]:
        return {
            "loading_auto": False,
            "prep_auto": False,
            "vis_auto": False,
            "modeling_auto": False,
            "report_auto": False,
            "plan": f"数据加载失败：{meta['error']}",
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
        # Keep the legacy prompt variable name for compatibility.
        "preference_select": preference_selected or "",
        "ref_context": ref_context or "（无参考资料）",
    }

    # Ask the LLM for the stage switches as strict JSON.
    planner_sys = render_file("planning/planner_llm_sys.txt", ctx)
    planner_user = render_file("planning/planner_llm_user.txt", ctx)
    planner_result = chat_json(
        planner_sys,
        planner_user,
        name="planning.planner",
        temperature=1.0,
    )

    loading_auto = as_bool(planner_result.get("loading_auto"), default=True)
    prep_auto = as_bool(planner_result.get("prep_auto"), default=True)
    vis_auto = as_bool(planner_result.get("vis_auto"), default=True)
    modeling_auto = as_bool(planner_result.get("modeling_auto"), default=False)
    report_auto = as_bool(planner_result.get("report_auto"), default=True)

    # Generate the user-facing analysis plan text.
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


# CLI smoke-test entry point.

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
