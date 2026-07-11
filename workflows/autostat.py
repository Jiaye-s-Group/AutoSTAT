"""
Top-level AutoSTAT workflow orchestration.

The local engine runs six workflow stages in order:

    Planning → Loading → Preprocessing → Visualizing → Modeling → Reporting

Each stage follows the switches returned by Planning and can report progress
through the optional `on_step` callback.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import pandas as pd

from workflows.planning import run_planning_workflow
from workflows.loading import run_loading_workflow
from workflows.preprocessing import run_preprocessing_workflow
from workflows.visualizing import run_visualizing_workflow
from workflows.modeling import run_modeling_workflow
from workflows.reporting_toc import run_reporting_toc_workflow
from workflows.reporting_partly import run_reporting_partly_workflow
from workflows.reporting_reference_context import build_stage_reference_contexts


def run_autostat(
    df: pd.DataFrame,
    *,
    user_input_pre: str = "",
    user_input_load: str = "",
    user_input_vis: str = "",
    user_input_model: str = "",
    user_input_report: str = "",
    add_preference: str = "",
    preference_selected: str = "",
    target_column: str = "",
    visualization_color: str = "",
    outline_length: str = "标准",
    ref_retriever=None,
    on_step: Callable[[str, dict], None] | None = None,
) -> dict[str, Any]:
    """Run the full AutoSTAT workflow once."""

    def _get_ref(query: str) -> str:
        """Retrieve reference-document snippets for a workflow prompt."""
        if ref_retriever is None or ref_retriever.is_empty:
            return ""
        return ref_retriever.retrieve_and_format(query, top_k=3)

    workflow_started = time.perf_counter()
    runtime_events: list[dict[str, Any]] = []

    def run_timed_stage(step: str, func: Callable[[], dict]) -> dict:
        stage_started = time.perf_counter()
        payload = func()
        stage_finished = time.perf_counter()
        event = {
            "step": step,
            "runtime_seconds": round(stage_finished - stage_started, 3),
            "started_at_offset_seconds": round(stage_started - workflow_started, 3),
            "finished_at_offset_seconds": round(stage_finished - workflow_started, 3),
        }
        runtime_events.append(event)
        if isinstance(payload, dict):
            payload["_stage_runtime_seconds"] = event["runtime_seconds"]
            payload["_stage_started_at_offset_seconds"] = event["started_at_offset_seconds"]
            payload["_stage_finished_at_offset_seconds"] = event["finished_at_offset_seconds"]
        return payload

    def notify(step: str, payload: dict) -> None:
        if on_step:
            try:
                on_step(step, payload)
            except Exception:
                pass

    # 1. Planning.
    plan = run_timed_stage(
        "planning",
        lambda: run_planning_workflow(
            df=df,
            add_preference=add_preference,
            preference_selected=preference_selected,
            ref_context=_get_ref(f"数据分析 业务背景 {add_preference}"),
        ),
    )
    notify("planning", plan)

    # Stop early when planning cannot read the dataset.
    if not plan.get("shape_0"):
        return {
            "plan": plan,
            "error": "Planning 失败：数据加载异常",
            "final_html": "",
            "final_html_parts": [],
            "title": "",
        }

    # Keep planning outputs as the shared stage context.
    base = {
        "shape_0": plan["shape_0"],
        "shape_1": plan["shape_1"],
        "dtype_info_str": plan["dtype_info_str"],
        "head_dict_str": plan["head_dict_str"],
        "df": plan["df"],
    }

    # 2 & 3. Loading and preprocessing can run independently.
    from concurrent.futures import ThreadPoolExecutor

    def _run_loading():
        return run_timed_stage(
            "loading",
            lambda: run_loading_workflow(
                shape_0=base["shape_0"],
                shape_1=base["shape_1"],
                dtype_info_str=base["dtype_info_str"],
                head_dict_str=base["head_dict_str"],
                loading_auto=plan["loading_auto"],
                user_input=user_input_load,
                add_preference=add_preference,
                preference_selected=preference_selected,
                ref_context=_get_ref(f"字段含义 数据说明 {base['dtype_info_str'][:200]}"),
            ),
        )

    def _run_prep():
        def _run() -> dict:
            if plan["prep_auto"]:
                return run_preprocessing_workflow(
                    df=base["df"],
                    shape_0=base["shape_0"],
                    shape_1=base["shape_1"],
                    dtype_info_str=base["dtype_info_str"],
                    head_dict_str=base["head_dict_str"],
                    prep_auto=True,
                    user_input=user_input_pre,
                    add_preference=add_preference,
                    preference_selected=preference_selected,
                    ref_context=_get_ref(f"数据预处理 缺失值 异常值 {add_preference}"),
                )
            return {
                "summary_2": {"title": "", "desc": "", "processed_df": base["df"], "code": ""},
                "abstract_2": "",
                "suggestion": "",
            }

        return run_timed_stage("preprocessing", _run)

    with ThreadPoolExecutor(max_workers=2) as pool:
        load_future = pool.submit(_run_loading)
        prep_future = pool.submit(_run_prep)
        loading = load_future.result()
        prep = prep_future.result()

    notify("loading", loading)
    notify("preprocessing", prep)

    # Use processed data when available, otherwise fall back to the input data.
    _prep_summary = prep.get("summary_2", {})
    _processed_df_str = _prep_summary.get("processed_df") or base["df"]
    next_data = _processed_df_str

    # Parse processed data to update downstream columns and preview rows.
    try:
        import json as _json
        _records = _json.loads(str(next_data))
        _next_df = pd.DataFrame(_records)
        next_cols = list(_next_df.columns.astype(str))
        next_head = _json.dumps(
            _next_df.head(5).to_dict(orient="list"), ensure_ascii=False
        )
    except Exception:
        # Fall back to planning metadata if the processed data is not JSON.
        next_cols = list(df.columns.astype(str))
        next_head = base["head_dict_str"]

    # 4 & 5. Visualizing and modeling can run independently.
    from concurrent.futures import ThreadPoolExecutor

    def _run_viz():
        return run_timed_stage(
            "visualizing",
            lambda: run_visualizing_workflow(
                data=next_data,
                shape0=base["shape_0"],
                shape1=len(next_cols),
                cols=next_cols,
                def_head=next_head,
                vis_auto=plan["vis_auto"],
                color=visualization_color,
                user_input=user_input_vis,
                add_preference=add_preference,
                preference_selected=preference_selected,
                ref_context=_get_ref(f"可视化 图表 数据分布 {add_preference}"),
            ),
        )

    def _run_model():
        return run_timed_stage(
            "modeling",
            lambda: run_modeling_workflow(
                data=next_data,
                df_head=next_head,
                columns=next_cols,
                modeling_auto=plan["modeling_auto"],
                target=target_column,
                user_input=user_input_model,
                user_prompt=user_input_model or add_preference,
                add_preference=add_preference,
                preference_selected=preference_selected,
                ref_context=_get_ref(f"建模 算法 {target_column} {add_preference}"),
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        viz_future = pool.submit(_run_viz)
        model_future = pool.submit(_run_model)
        viz = viz_future.result()
        model = model_future.result()

    notify("visualizing", viz)
    notify("modeling", model)

    # 6. Reporting.
    if plan["report_auto"]:
        stage_reference_contexts = build_stage_reference_contexts(
            plan=plan,
            loading=loading,
            prep=prep,
            viz=viz,
            model=model,
            next_cols=next_cols,
            next_head=next_head,
        )
        toc = run_timed_stage(
            "reporting_toc",
            lambda: run_reporting_toc_workflow(
                load_summary=loading.get("summary_1", {}),
                preproc_summary=prep.get("summary_2", {}),
                visual_summary=viz.get("summary_3", {}),
                coding_summary=model.get("summary_4", {}),
                load_abstract=loading.get("abstract_1", ""),
                preproc_abstract=prep.get("abstract_2", ""),
                visual_abstract=viz.get("abstract_3", ""),
                coding_abstract=model.get("abstract_4", ""),
                selected_full_conten=viz.get("full", ""),
                toc_md=[],
                outline_length=outline_length,
                report_auto=True,
                user_input=user_input_report,
                add_preference=add_preference,
                preference_selected=preference_selected,
                ref_context=_get_ref(f"报告 分析结论 业务背景 {add_preference}"),
            ),
        )
        notify("reporting_toc", toc)

        partly = run_timed_stage(
            "reporting_partly",
            lambda: run_reporting_partly_workflow(
                toc_text=toc["toc_text"],
                selected_full_conten=toc["selected_full_conten"],
                load_abstract=toc["load_abstract"],
                preproc_abstract=toc["preproc_abstract"],
                visual_abstract=toc["visual_abstract"],
                coding_abstract=toc["coding_abstract"],
                user_input=user_input_report,
                add_preference=toc.get("add_preference", add_preference),
                preference_select=toc.get("preference_select", preference_selected),
                ref_context=toc.get("ref_context", "") or _get_ref(f"报告撰写 {add_preference}"),
                stage_reference_contexts=stage_reference_contexts,
            ),
        )
        notify("reporting_partly", partly)

        final_html = partly["final_html"]
        final_html_parts = partly["final_html_parts"]
        title = partly["title"]
    else:
        toc = {}
        partly = {}
        final_html = ""
        final_html_parts = []
        title = ""

    return {
        "plan": plan,
        "loading": loading,
        "preprocessing": prep,
        "visualizing": viz,
        "modeling": model,
        "reporting_toc": toc,
        "reporting_partly": partly,
        "runtime_events": sorted(runtime_events, key=lambda item: item["started_at_offset_seconds"]),
        "final_html": final_html,
        "final_html_parts": final_html_parts,
        "title": title,
    }


# CLI smoke-test entry point.

if __name__ == "__main__":
    import sys
    import pandas as pd

    if len(sys.argv) < 2:
        print("用法: python -m workflows.autostat <csv_path> [target_column]")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    target = sys.argv[2] if len(sys.argv) > 2 else ""

    def print_step(step: str, _payload: dict) -> None:
        print(f"✓ 完成步骤: {step}")

    result = run_autostat(df, target_column=target, on_step=print_step)

    print("\n==================== 最终报告 ====================")
    print(f"title: {result['title']}")
    print(f"final_html 长度: {len(result['final_html'])} 字符")
    print(f"chunk 数量: {len(result['final_html_parts'])}")
    if result["final_html"]:
        out = "autostat_report.html"
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{result['title']}</title></head><body>")
            f.write(result["final_html"])
            f.write("</body></html>")
        print(f"已写入 {out}")
