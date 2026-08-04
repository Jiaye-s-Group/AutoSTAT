"""
AutoSTAT 总编排 workflow。

Coordinates the AutoSTAT analysis stages.
但本地版本不做单次大调用，而是按顺序串起 6 个子 workflow：

    Planning → Loading → Preprocessing → Visualizing → Modeling → Reporting

每步都会尊重 Planning 的 auto 开关（False 时跳过对应阶段）。
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
from core.report_language import is_english_language, normalize_app_language


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
    outline_length: str = "标准",
    language: str = "zh",
    ref_retriever=None,
    on_step: Callable[[str, dict], None] | None = None,
) -> dict[str, Any]:
    """
    一次性跑完整流程。

    on_step(step_name, payload) 是可选回调，让前端可以在每阶段结束立即显示进度。
    ref_retriever: RefDocRetriever 实例，用于从用户上传的参考资料中检索相关内容。
    """
    language = normalize_app_language(language)
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

    def sorted_runtime_events() -> list[dict[str, Any]]:
        return sorted(runtime_events, key=lambda item: item["started_at_offset_seconds"])

    def _get_ref(query: str) -> str:
        """按 query 从参考资料中检索相关 chunks，格式化为 prompt 文本。"""
        retriever = getattr(ref_retriever, "_ref_retriever", ref_retriever)
        if retriever is None:
            return ""
        is_empty = getattr(retriever, "is_empty", False)
        if callable(is_empty):
            try:
                is_empty = is_empty()
            except Exception:
                is_empty = False
        if is_empty:
            return ""
        try:
            return retriever.retrieve_and_format(query, top_k=5, min_score=0.0)
        except TypeError:
            return retriever.retrieve_and_format(query, top_k=5)
        except Exception:
            return ""

    def _query(zh: str, en: str) -> str:
        return en if is_english_language(language) else zh

    def notify(step: str, payload: dict) -> None:
        if on_step:
            try:
                on_step(step, payload)
            except Exception:
                pass

    # ========== 1. Planning ==========
    plan = run_timed_stage(
        "planning",
        lambda: run_planning_workflow(
            df=df,
            add_preference=add_preference,
            preference_selected=preference_selected,
            ref_context=_get_ref(_query(
                f"数据分析 业务背景 {add_preference}",
                f"data analysis business context {add_preference}",
            )),
            language=language,
        ),
    )
    notify("planning", plan)

    # 可能 Planning 失败
    if not plan.get("shape_0"):
        return {
            "plan": plan,
            "error": (
                "Planning failed: data loading error"
                if is_english_language(language)
                else "Planning 失败：数据加载异常"
            ),
            "final_html": "",
            "final_html_parts": [],
            "title": "",
            "runtime_events": sorted_runtime_events(),
        }

    # 把 Planning 的产物塞到 ctx
    base = {
        "shape_0": plan["shape_0"],
        "shape_1": plan["shape_1"],
        "dtype_info_str": plan["dtype_info_str"],
        "head_dict_str": plan["head_dict_str"],
        "data_profile_str": plan.get("data_profile_str", ""),
        "df": plan["df"],
    }

# ========== 2 & 3. Loading + Preprocessing 并行 ==========
    from concurrent.futures import ThreadPoolExecutor
    from core.llm_client import submit_with_context

    def _run_loading():
        return run_timed_stage(
            "loading",
            lambda: run_loading_workflow(
                shape_0=base["shape_0"],
                shape_1=base["shape_1"],
                dtype_info_str=base["dtype_info_str"],
                head_dict_str=base["head_dict_str"],
                data_profile_str=base.get("data_profile_str", ""),
                loading_auto=plan["loading_auto"],
                user_input=user_input_load,
                add_preference=add_preference,
                preference_selected=preference_selected,
                ref_context=_get_ref(_query(
                    f"数据字典 字段说明 变量含义 单位 编码 取值方向 缺失值 {base['dtype_info_str'][:1200]} {user_input_load} {add_preference}",
                    f"data dictionary field descriptions variable meanings units coding value direction missing values {base['dtype_info_str'][:1200]} {user_input_load} {add_preference}",
                )),
                language=language,
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
                    ref_context=_get_ref(_query(
                        f"数据预处理 缺失值 异常值 {add_preference}",
                        f"data preprocessing missing values outliers {add_preference}",
                    )),
                    language=language,
                )
            return {
                "summary_2": {
                    "title": "",
                    "desc": "",
                    "processed_df": base["df"],
                    "code": "",
                    "status": "skipped",
                    "data_source": "raw",
                },
                "abstract_2": "",
                "suggestion": "",
                "_status": "skipped",
            }

        return run_timed_stage("preprocessing", _run)

    with ThreadPoolExecutor(max_workers=2) as pool:
        load_future = submit_with_context(pool, _run_loading)
        prep_future = submit_with_context(pool, _run_prep)
        loading = load_future.result()
        prep = prep_future.result()

    notify("loading", loading)
    notify("preprocessing", prep)

    if plan["prep_auto"] and prep.get("_status") != "succeeded":
        return {
            "plan": plan,
            "loading": loading,
            "preprocessing": prep,
            "visualizing": {},
            "modeling": {},
            "reporting_toc": {},
            "reporting_partly": {},
            "error": prep.get("_code_error") or "Preprocessing failed",
            "final_html": "",
            "final_html_parts": [],
            "title": "",
            "runtime_events": sorted_runtime_events(),
        }

    # ── 从 preprocessing 输出中提取下游数据 ──────────────────────
    # 预处理成功时使用 processed_df；仅当该阶段明确跳过时使用原始数据。
    _prep_summary = prep.get("summary_2", {})
    _processed_df_str = _prep_summary.get("processed_df") or base["df"]
    next_data = _processed_df_str

    # 解析 processed_df 以获取最新列名和 head
    try:
        import json as _json
        _records = _json.loads(str(next_data))
        _next_df = pd.DataFrame(_records)
        next_cols = list(_next_df.columns.astype(str))
        next_head = _json.dumps(
            _next_df.head(5).to_dict(orient="list"), ensure_ascii=False
        )
    except Exception:
        # 解析失败时退化为 Planning 阶段的元信息
        next_cols = list(df.columns.astype(str))
        next_head = base["head_dict_str"]

# ========== 4 & 5. Visualizing + Modeling 并行 ==========
    from concurrent.futures import ThreadPoolExecutor
    from core.llm_client import submit_with_context

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
                user_input=user_input_vis,
                add_preference=add_preference,
                preference_selected=preference_selected,
                ref_context=_get_ref(_query(
                    f"可视化 图表 数据分布 {add_preference}",
                    f"visualization charts data distribution {add_preference}",
                )),
                language=language,
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
                ref_context=_get_ref(_query(
                    f"建模 算法 {target_column} {add_preference}",
                    f"modeling algorithm target {target_column} {add_preference}",
                )),
                language=language,
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        viz_future = submit_with_context(pool, _run_viz)
        model_future = submit_with_context(pool, _run_model)
        viz = viz_future.result()
        model = model_future.result()

    notify("visualizing", viz)
    notify("modeling", model)

    failed_analysis_stages = []
    if plan["vis_auto"] and viz.get("_status") != "succeeded":
        failed_analysis_stages.append("visualization")
    if plan["modeling_auto"] and model.get("_status") != "succeeded":
        failed_analysis_stages.append("modeling")
    if failed_analysis_stages:
        return {
            "plan": plan,
            "loading": loading,
            "preprocessing": prep,
            "visualizing": viz,
            "modeling": model,
            "reporting_toc": {},
            "reporting_partly": {},
            "error": f"Analysis stage failed: {', '.join(failed_analysis_stages)}",
            "final_html": "",
            "final_html_parts": [],
            "title": "",
            "runtime_events": sorted_runtime_events(),
        }

    # ========== 6. Reporting ==========
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
                ref_context=_get_ref(_query(
                    f"报告 分析结论 业务背景 {add_preference}",
                    f"report findings analysis conclusions business context {add_preference}",
                )),
                report_language=language,
            ),
        )
        notify("reporting_toc", toc)

        partly = run_timed_stage(
            "reporting_partly",
            lambda: run_reporting_partly_workflow(
                toc_text=toc["toc_text"],
                selected_full_conten=toc["selected_full_conten"],
                figure_artifacts=viz.get("figure_artifacts"),
                load_abstract=toc["load_abstract"],
                preproc_abstract=toc["preproc_abstract"],
                visual_abstract=toc["visual_abstract"],
                coding_abstract=toc["coding_abstract"],
                user_input=user_input_report,
                add_preference=toc.get("add_preference", add_preference),
                preference_select=toc.get("preference_select", preference_selected),
                ref_context=toc.get("ref_context", "") or _get_ref(_query(
                    f"报告撰写 {add_preference}",
                    f"report writing {add_preference}",
                )),
                stage_reference_contexts=stage_reference_contexts,
                report_language=language,
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
        "runtime_events": sorted_runtime_events(),
        "final_html": final_html,
        "final_html_parts": final_html_parts,
        "title": title,
    }


# ---------- CLI 测试入口 ----------

if __name__ == "__main__":
    import sys

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
