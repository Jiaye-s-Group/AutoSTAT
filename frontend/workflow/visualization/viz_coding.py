import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import streamlit as st
from stqdm import stqdm
from streamlit_ace import st_ace
import streamlit_antd_components as sac

from core.safe_code import safe_exec
from utils.i18n import bt
from utils.sanitize_code import sanitize_visualization_code
from utils.suggestion_state import (
    begin_code_execution,
    can_auto_repair,
    finish_code_execution,
    get_suggestion_state,
    mark_code_draft,
    record_execution_failure,
)
from utils.workflow_state import (
    current_dataset_fingerprint,
    invalidate_from,
    record_stage_status,
    stable_fingerprint,
    stage_is_current,
)
from workflow.visualization.viz_color import apply_palette_to_figure


def _show_execution_error(message: str, error_text: str) -> None:
    st.error(message)
    if error_text:
        st.code(error_text, language="text")


def _record_visualization_failure(agent, state, error_text: str) -> None:
    record_execution_failure(state, error_text)
    if not agent.load_fig():
        record_stage_status(
            st.session_state,
            "visualization",
            "failed",
            input_fingerprint=str(
                st.session_state.get("analysis_dataset_fingerprint")
                or current_dataset_fingerprint(st.session_state)
            ),
            error=error_text,
        )


def _load_workflow_visualization_code(agent) -> bool:
    workflow_code = sanitize_visualization_code(st.session_state.get("final_code") or "")
    if not workflow_code:
        return False

    agent.save_code(workflow_code)
    return True


def _summary_3_fig_analysis(summary_3):
    if not isinstance(summary_3, dict):
        return []

    fig_analysis = summary_3.get("fig_analysis")
    if not isinstance(fig_analysis, list):
        return []

    normalized = []
    for item in fig_analysis:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "fig": str(item.get("fig", "")).strip(),
                "analysis": str(item.get("analysis", "")).strip(),
                "title": str(item.get("title", "")).strip(),
            }
        )
    return normalized


def _match_fig_analysis(fig_analysis_items, fig_key, index):
    # Prefer lightweight key-based matching before positional fallback.
    for item in fig_analysis_items:
        fig_field = item.get("fig", "")
        if not fig_field:
            continue
        # Exact match (legacy path)
        if fig_field == fig_key:
            if item.get("analysis"):
                return item["analysis"]
        # fig_key is the dict key (e.g. "fig_1") – check if it appears
        # in the fig JSON title or as a substring of the serialised figure
        if fig_key and fig_key in fig_field:
            if item.get("analysis"):
                return item["analysis"]

    # Fallback: match by position index
    if index < len(fig_analysis_items):
        item = fig_analysis_items[index]
        if item.get("analysis"):
            return item["analysis"]

    return None


def _normalize_figure(fig):
    if isinstance(fig, go.Figure):
        return fig

    if isinstance(fig, str):
        try:
            return pio.from_json(fig)
        except Exception:
            return None

    if isinstance(fig, dict):
        try:
            return go.Figure(fig)
        except Exception:
            return None

    return None


def generate_visualization_code_once(agent) -> bool:
    return _load_workflow_visualization_code(agent)


def execute_visualization_code_once(agent, code_override=None) -> bool:
    df = agent.load_df()
    code = code_override if code_override is not None else agent.load_code()
    code = sanitize_visualization_code(code)

    if df is None or not code:
        return False

    exec_ns = {
        "df": df,
        "np": np,
        "pd": pd,
        "px": px,
        "go": go,
    }

    agent.save_code(code)
    agent.save_fig([])
    try:
        with st.spinner(bt("正在运行可视化脚本...", "Running the visualization script...")):
            safe_exec(code, exec_ns)
    except Exception:
        error_text = traceback.format_exc()
        agent.save_error(error_text)
        _show_execution_error(
            bt(
                "当前代码出现执行错误，请点击清除可视化分析，重新生成可视化推荐。",
                "The current code failed to execute. Clear the visualization analysis and regenerate the recommendation.",
            ),
            error_text,
        )
        return False

    fig_dict = exec_ns.get("fig_dict")
    if not fig_dict or not isinstance(fig_dict, dict):
        error_text = bt(
            "可视化脚本执行完成，但未产出有效的 `fig_dict`。\n"
            "请确保代码中创建 `fig_dict`，且其类型为 dict，例如：fig_dict = {'fig_1': fig}。",
            "The visualization script finished, but did not produce a valid `fig_dict`.\n"
            "Make sure the code creates `fig_dict` as a dict, for example: fig_dict = {'fig_1': fig}.",
        )
        agent.save_error(error_text)
        _show_execution_error(
            bt(
                "可视化脚本未产出有效的 `fig_dict`，请先前往可视化页面检查代码。",
                "The visualization script did not produce a valid `fig_dict`. Check the code on the visualization page first.",
            ),
            error_text,
        )
        return False

    summary_3 = st.session_state.get("summary_3")
    fig_analysis_items = _summary_3_fig_analysis(summary_3)
    tu_title_all = st.session_state.get("tu_title")
    if isinstance(tu_title_all, list):
        tu_title_list = list(tu_title_all)
    else:
        tu_title_list = []
    with st.spinner(bt("正在处理可视化结果...", "Processing visualization results...")):
        for idx, (col_name, fig) in enumerate(stqdm(fig_dict.items())):
            try:
                normalized_fig = _normalize_figure(fig)
                if normalized_fig is None:
                    continue
                base_fig = go.Figure(normalized_fig)

                dtype_info = ", ".join(
                    f"{c}: {df[c].dtype}" for c in df.columns
                )
                normalized_fig = apply_palette_to_figure(
                    normalized_fig,
                    agent.load_color() or [],
                    idx,
                )

                desc = _match_fig_analysis(fig_analysis_items, col_name, idx)
                if desc is None:
                    try:
                        desc = agent.desc_fig(normalized_fig, dtype_info)
                    except Exception:
                        desc = None

                # Bundle the matching title so it stays aligned with this figure
                # even when other figures are skipped.
                fig_title = tu_title_list[idx] if idx < len(tu_title_list) else ""

                agent.add_fig(normalized_fig, desc, base_fig=base_fig.to_json(), title=fig_title)
            except Exception:
                continue

    return bool(agent.load_fig())


def vis_code_gen(agent, debug = False, auto = False) -> None:

    suggest = agent.load_suggestion()
    current_code = agent.load_code()
    control_slot = st.empty()

    chat_history = agent.load_memory()
    already_generated = any(
        entry["role"] == "assistant"
        and any(
            message in str(entry["content"])
            for message in (
                "训练脚本已更新！请重新运行代码！",
                "可视化代码已从工作流加载，请在下方执行。",
                "Visualization code has been loaded from the workflow. Run it below.",
            )
        )
        for entry in chat_history
    )

    workflow_code = st.session_state.get("final_code")
    if workflow_code:
        code_is_loaded = str(current_code or "").strip() == str(workflow_code).strip()
        analyze_btn = False
        with control_slot.container():
            if not code_is_loaded:
                analyze_btn = st.button(
                    bt("🔧 生成可视化代码", "🔧 Generate Visualization Code"),
                    key="viz_code_workflow",
                )
        if analyze_btn or (auto and not code_is_loaded):
            control_slot.empty()
            _load_workflow_visualization_code(agent)
            loaded_message = bt(
                "可视化代码已从工作流加载，请在下方执行。",
                "Visualization code has been loaded from the workflow. Run it below.",
            )
            st.chat_message("assistant").write(loaded_message)
            agent.add_memory({"role": "assistant", "content": loaded_message})
            st.rerun()
        return

    if suggest is not None:
        code_is_loaded = bool(agent.load_code())
        analyze_btn = False
        with control_slot.container():
            if not code_is_loaded:
                analyze_btn = st.button(
                    bt("🔧 生成可视化代码", "🔧 Generate Visualization Code"),
                    key="viz_code_generate",
                )
        if analyze_btn or (auto and not code_is_loaded):
            if (
                isinstance(st.session_state.get("_viz_phase1_ctx"), dict)
                and isinstance(st.session_state.get("_viz_phase2_inputs"), dict)
            ):
                st.session_state._viz_phase2_requested = True
            else:
                st.error(
                    bt(
                        "当前可视化建议上下文已失效，请清除后重新生成建议。",
                        "The visualization recommendation context has expired. Clear it and generate a new recommendation.",
                    )
                )
                return
            control_slot.empty()
            st.rerun()
            

def vis_execution(agent, auto = False):
    df = agent.load_df()
    code = agent.load_code()
    edited = st_ace(value=code, height=450, theme="tomorrow_night", language="python", auto_update=True)
    state = get_suggestion_state(st.session_state, "visualization")
    tracked_execution = bool(state.get("executed_code_fingerprint"))
    result_is_current = bool(
        tracked_execution and state.get("current_code_fingerprint") == state.get("executed_code_fingerprint")
    )
    if edited is not None:
        _, result_is_current = mark_code_draft(state, edited)
        if agent.load_fig() and tracked_execution and not result_is_current:
            st.warning(bt(
                "代码已修改，下方保留的是上一次成功代码生成的图表。",
                "The code has changed. The figures below are from the last successful code.",
            ))

    if "viz_desc_switch" not in st.session_state:
        st.session_state["viz_desc_switch"] = False
    if "viz_desc_switch_widget" not in st.session_state:
        st.session_state["viz_desc_switch_widget"] = st.session_state["viz_desc_switch"]
    desc_switch = sac.switch(label=bt("附加分析", "Additional Analysis"), key="viz_desc_switch_widget", off_label="Off")
    st.session_state["viz_desc_switch"] = bool(st.session_state.get("viz_desc_switch_widget", desc_switch))

    if code is not None and (
        st.button(bt("▶️ 执行可视化", "▶️ Run Visualization"))
        or (auto and not agent.load_fig())
    ):
        current_code = sanitize_visualization_code(edited)
        run_id = begin_code_execution(state, current_code)
        agent.save_code(current_code)
        exec_ns = {"df": df, "np": np, "pd": pd, "px": px, "go": go}
        try:
            with st.spinner(bt("正在运行可视化脚本...", "Running the visualization script...")):
                safe_exec(current_code, exec_ns)
            fig_dict = exec_ns.get("fig_dict")
            if not isinstance(fig_dict, dict) or not fig_dict:
                raise ValueError(bt(
                    "可视化脚本未产出有效的 `fig_dict`。",
                    "The visualization script did not produce a valid `fig_dict`.",
                ))

            summary_3 = st.session_state.get("summary_3")
            fig_analysis_items = _summary_3_fig_analysis(summary_3)
            titles = st.session_state.get("tu_title")
            tu_title_list = list(titles) if isinstance(titles, list) else []
            new_figures = []
            with st.spinner(bt("正在处理可视化结果...", "Processing visualization results...")):
                for idx, (col_name, fig) in enumerate(stqdm(fig_dict.items())):
                    normalized_fig = _normalize_figure(fig)
                    if normalized_fig is None:
                        continue
                    base_fig = go.Figure(normalized_fig)
                    normalized_fig = apply_palette_to_figure(normalized_fig, agent.load_color() or [], idx)
                    desc = _match_fig_analysis(fig_analysis_items, col_name, idx)
                    if desc is None:
                        try:
                            desc = agent.desc_fig(normalized_fig, ", ".join(f"{c}: {df[c].dtype}" for c in df.columns))
                        except Exception:
                            desc = None
                    new_figures.append({
                        "fig": normalized_fig,
                        "base_fig": base_fig.to_json(),
                        "desc": desc,
                        "title": tu_title_list[idx] if idx < len(tu_title_list) else "",
                    })
            if not new_figures:
                raise ValueError(bt(
                    "可视化代码运行后没有得到可展示图表。",
                    "The visualization code ran but produced no displayable charts.",
                ))
        except Exception:
            error_text = traceback.format_exc()
            finish_code_execution(state, run_id, success=False)
            agent.save_error(error_text)
            _record_visualization_failure(agent, state, error_text)
        else:
            invalidate_from(st.session_state, "visualization", reason="visualization result replaced")
            agent.save_fig(new_figures)
            finish_code_execution(state, run_id, success=True)
            st.session_state.pop("visualization_failure", None)
            record_stage_status(
                st.session_state,
                "visualization",
                "succeeded",
                input_fingerprint=str(st.session_state.get("analysis_dataset_fingerprint") or current_dataset_fingerprint(st.session_state)),
                output_fingerprint=stable_fingerprint(current_code, len(new_figures)),
            )
            agent.finish_auto()
            st.rerun()

    error_text = str(state.get("last_execution_error") or "")
    if error_text:
        st.error(bt("执行失败", "Execution failed"))
        with st.expander(bt("查看错误详情", "View error details")):
            st.code(error_text, language="text")
        attempts = int(state.get("auto_repair_attempts") or 0)
        if can_auto_repair(state):
            repair_slot = st.empty()
            with repair_slot.container():
                repair_requested = st.button(
                    bt("自动修复代码", "Auto-fix Code"),
                    key="viz_auto_fix_code",
                )
                if attempts:
                    st.caption(bt(f"本轮已自动修复 {attempts} 次，最多 5 次。", f"This round has used {attempts} of 5 automatic repairs."))
            if repair_requested:
                repair_slot.empty()
                st.session_state._viz_code_repair_requested = True
                st.rerun()
        else:
            st.caption(bt("已达到自动修复上限，请手动修改代码或调整建议后重新生成。", "The automatic repair limit has been reached. Edit the code or revise the suggestion before generating again."))
