import time
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

from utils.sanitize_code import sanitize_code, sanitize_visualization_code
from workflow.visualization.viz_color import apply_palette_to_figure


def _load_workflow_visualization_code(agent) -> bool:
    workflow_code = sanitize_visualization_code(st.session_state.get("final_code") or "")
    if not workflow_code:
        return False

    agent.save_code(workflow_code)
    return True


def _warn_missing_workflow_visualization_code() -> None:
    st.warning("当前未获取到工作流生成的可视化代码，请先重新执行可视化推荐。")


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
            }
        )
    return normalized


def _match_fig_analysis(fig_analysis_items, fig_key, index):
    if index < len(fig_analysis_items):
        item = fig_analysis_items[index]
        if item.get("analysis"):
            return item["analysis"]

    for item in fig_analysis_items:
        if item.get("fig") == fig_key and item.get("analysis"):
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


def _regenerate_visualization_code(agent, df, suggest, rerun=True) -> None:
    if not _load_workflow_visualization_code(agent):
        _warn_missing_workflow_visualization_code()
        return

    st.chat_message("assistant").write("训练脚本已更新！请重新运行代码！")
    agent.add_memory({"role": "assistant", "content": "训练脚本已更新！请重新运行代码！"})
    if rerun:
        st.rerun()


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
        with st.spinner("正在运行可视化脚本..."):
            exec(code, exec_ns)
    except Exception:
        error_text = traceback.format_exc()
        st.error("当前代码出现执行错误，请点击清除可视化分析，重新生成可视化推荐。")
        agent.save_error(error_text)
        return False

    fig_dict = exec_ns.get("fig_dict")
    if not fig_dict or not isinstance(fig_dict, dict):
        st.error("可视化脚本未产出有效的 `fig_dict`，请先前往可视化页面检查代码。")
        agent.save_error("fig_dict missing or invalid")
        return False

    summary_3 = st.session_state.get("summary_3")
    fig_analysis_items = _summary_3_fig_analysis(summary_3)
    with st.spinner("正在处理可视化结果..."):
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

                agent.add_fig(normalized_fig, desc, base_fig=base_fig.to_json())
            except Exception:
                continue

    return bool(agent.load_fig())


def vis_code_gen(agent, debug = False, auto = False) -> None:

    suggest = agent.load_suggestion()
    current_code = agent.load_code()

    chat_history = agent.load_memory()
    already_generated = any(
        entry["role"] == "assistant" and "训练脚本已更新！请重新运行代码！" in str(entry["content"])
        for entry in chat_history
    )

    workflow_code = st.session_state.get("final_code")
    if workflow_code:
        analyze_btn = st.button("🔧 生成可视化代码", key="viz_code_workflow")
        if analyze_btn or (auto and not current_code):
            _load_workflow_visualization_code(agent)
            st.chat_message("assistant").write("可视化代码已从工作流加载，请在下方执行。")
            agent.add_memory({"role": "assistant", "content": "可视化代码已从工作流加载，请在下方执行。"})
            st.rerun()
        return

    if suggest is not None:
        analyze_btn = st.button("🔧 生成可视化代码", key="viz_code_generate")
        if analyze_btn or (debug == True or (auto and not current_code)):
            _warn_missing_workflow_visualization_code()
            

def vis_execution(agent, auto = False):

    df = agent.load_df()

    exec_ns = {
        "df": df,
        "np": np,
        "pd": pd,
        "px": px,
        "go": go,
    }

    code = agent.load_code()
    edited = st_ace(
            value=code,
            height=450,
            theme="tomorrow_night",
            language="python",
            auto_update=True
        )
    desc_switch = sac.switch(
        label='附加分析',
        value=st.session_state.get("viz_desc_switch", False),
        key="viz_desc_switch",
        off_label='Off',
    )
    if code is not None:
        not_executed = agent.load_fig() == []
        # 当点击按钮，或者 auto=True 且尚未执行过时才执行
        if st.button("▶️ 执行可视化") or (auto and not_executed):
            code = sanitize_visualization_code(edited)
            agent.save_code(code)
            agent.save_fig([])
            try:
                with st.spinner("正在运行可视化脚本..."):
                    exec(code, exec_ns)
            except Exception as exc:
                agent.save_error(traceback.format_exc())
                st.error("当前代码出现执行错误，请点击清除可视化分析，重新生成可视化推荐。")
            else:
                fig_dict = exec_ns.get("fig_dict")
                if not fig_dict or not isinstance(fig_dict, dict):
                    agent.save_error(traceback.format_exc())
                    st.error("当前代码出现执行错误，请点击清除可视化分析，重新生成可视化推荐。")
                else:
                    summary_3 = st.session_state.get("summary_3")
                    fig_analysis_items = _summary_3_fig_analysis(summary_3)
                    with st.spinner("正在处理可视化结果..."):
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
                                agent.add_fig(normalized_fig, desc, base_fig=base_fig.to_json())
                            except Exception:
                                continue
                        agent.finish_auto()
                        st.rerun()
