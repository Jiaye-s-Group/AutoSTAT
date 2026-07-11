"""
Bridge Streamlit page inputs to local workflow functions.

Render pages build dictionaries from session state. This module normalizes those
inputs, calls the corresponding workflow, and returns the workflow result to the
UI layer.
"""
from __future__ import annotations

import json
from typing import Any

import streamlit as st

def _err(msg: str) -> None:
    st.error(msg)

def _get_ref_context(query: str = "") -> str:
    """Return formatted reference-document snippets from session state."""
    retriever = st.session_state.get("ref_retriever")
    if retriever is None or retriever.is_empty:
        return ""
    return retriever.retrieve_and_format(query or "数据分析", top_k=3)




def call_loading_bridge(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Call the loading workflow from UI inputs."""
    from workflows.loading import run_loading_workflow

    try:
        return run_loading_workflow(
            shape_0=int(inputs.get("shape_0", 0)),
            shape_1=int(inputs.get("shape_1", 0)),
            dtype_info_str=str(inputs.get("dtype_info_str", "")),
            head_dict_str=str(inputs.get("head_dict_str", "")),
            loading_auto=bool(inputs.get("loading_auto", True)),
            user_input=str(inputs.get("user_input", "")),
            add_preference=str(inputs.get("add_preference", "")),
            preference_selected=str(inputs.get("preference_selected", "")),
            ref_context=_get_ref_context(f"字段含义 数据说明 {inputs.get('dtype_info_str', '')[:200]}"),
        )
    except Exception as e:
        _err(f"本地 Loading workflow 失败：{e}")
        return None


def call_preprocessing_bridge(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Call the preprocessing workflow from UI inputs."""
    from workflows.preprocessing import run_preprocessing_workflow

    try:
        return run_preprocessing_workflow(
            df=str(inputs.get("df", "")),
            shape_0=int(inputs.get("shape_0", 0)),
            shape_1=int(inputs.get("shape_1", 0)),
            dtype_info_str=str(inputs.get("dtype_info_str", "")),
            head_dict_str=str(inputs.get("head_dict_str", "")),
            prep_auto=bool(inputs.get("prep_auto", True)),
            user_input=str(inputs.get("user_input", "")),
            add_preference=str(inputs.get("add_preference", "")),
            preference_selected=str(inputs.get("preference_selected", "")),
            ref_context=_get_ref_context(f"数据预处理 缺失值 异常值 {inputs.get('add_preference', '')}"),
        )
    except Exception as e:
        _err(f"本地 Preprocessing workflow 失败：{e}")
        return None


def call_visualizing_bridge(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Call the visualization workflow from UI inputs."""
    from workflows.visualizing import run_visualizing_workflow

    try:
        cols_val = inputs.get("cols", [])
        if isinstance(cols_val, str):
            try:
                cols_val = json.loads(cols_val)
            except Exception:
                cols_val = [cols_val]
        return run_visualizing_workflow(
            data=str(inputs.get("data", "")),
            shape0=int(inputs.get("shape0", 0) or inputs.get("shape_0", 0)),
            shape1=int(inputs.get("shape1", 0) or inputs.get("shape_1", 0)),
            cols=list(cols_val or []),
            def_head=str(inputs.get("def_head", "")),
            vis_auto=bool(inputs.get("vis_auto", True)),
            color=str(inputs.get("color", "")),
            user_input=str(inputs.get("user_input", "")),
            add_preference=str(inputs.get("add_preference", "")),
            preference_selected=str(inputs.get("preference_selected", "")),
            ref_context=_get_ref_context(f"可视化 图表 {inputs.get('add_preference', '')}"),
        )
    except Exception as e:
        _err(f"本地 Visualizing workflow 失败：{e}")
        return None


def call_visualizing_phase1_bridge(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Run visualization phase 1 for recommendation preview."""
    from workflows.visualizing import run_visualizing_phase1

    try:
        cols_val = inputs.get("cols", [])
        if isinstance(cols_val, str):
            try:
                cols_val = json.loads(cols_val)
            except Exception:
                cols_val = [cols_val]
        return run_visualizing_phase1(
            data=str(inputs.get("data", "")),
            shape0=int(inputs.get("shape0", 0) or inputs.get("shape_0", 0)),
            shape1=int(inputs.get("shape1", 0) or inputs.get("shape_1", 0)),
            cols=list(cols_val or []),
            def_head=str(inputs.get("def_head", "")),
            vis_auto=bool(inputs.get("vis_auto", True)),
            color=str(inputs.get("color", "")),
            user_input=str(inputs.get("user_input", "")),
            add_preference=str(inputs.get("add_preference", "")),
            preference_selected=str(inputs.get("preference_selected", "")),
            ref_context=_get_ref_context(f"可视化 图表 {inputs.get('add_preference', '')}"),
        )
    except Exception as e:
        _err(f"本地 Visualizing phase1 失败：{e}")
        return None


def call_visualizing_phase2_bridge(inputs: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Run visualization phase 2 for code generation and chart analysis."""
    from workflows.visualizing import run_visualizing_phase2

    try:
        cols_val = inputs.get("cols", [])
        if isinstance(cols_val, str):
            try:
                cols_val = json.loads(cols_val)
            except Exception:
                cols_val = [cols_val]
        return run_visualizing_phase2(
            ctx=ctx,
            data=str(inputs.get("data", "")),
            cols=list(cols_val or []),
            def_head=str(inputs.get("def_head", "")),
        )
    except Exception as e:
        _err(f"本地 Visualizing phase2 失败：{e}")
        return None


def call_modeling_bridge(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Call the modeling workflow from UI inputs."""
    from workflows.modeling import run_modeling_workflow

    try:
        cols_val = inputs.get("columns", [])
        if isinstance(cols_val, str):
            try:
                cols_val = json.loads(cols_val)
            except Exception:
                cols_val = [cols_val]
        return run_modeling_workflow(
            data=str(inputs.get("data", "")),
            df_head=str(inputs.get("df_head", "")),
            columns=list(cols_val or []),
            modeling_auto=bool(inputs.get("modeling_auto", True)),
            target=str(inputs.get("target", "")),
            train_code=str(inputs.get("train_code", "")),
            user_input=str(inputs.get("user_input", "")),
            user_prompt=str(inputs.get("user_prompt", "")),
            add_preference=str(inputs.get("add_preference", "")),
            preference_selected=str(inputs.get("preference_selected", "")),
            ref_context=_get_ref_context(f"建模 算法 {inputs.get('target', '')} {inputs.get('add_preference', '')}"),
        )
    except Exception as e:
        _err(f"本地 Modeling workflow 失败：{e}")
        return None


def call_modeling_phase1_bridge(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Run modeling phase 1 for recommendation preview."""
    from workflows.modeling import run_modeling_phase1

    try:
        cols_val = inputs.get("columns", [])
        if isinstance(cols_val, str):
            try:
                cols_val = json.loads(cols_val)
            except Exception:
                cols_val = [cols_val]
        return run_modeling_phase1(
            data=str(inputs.get("data", "")),
            df_head=str(inputs.get("df_head", "")),
            columns=list(cols_val or []),
            modeling_auto=bool(inputs.get("modeling_auto", True)),
            target=str(inputs.get("target", "")),
            train_code=str(inputs.get("train_code", "")),
            user_input=str(inputs.get("user_input", "")),
            user_prompt=str(inputs.get("user_prompt", "")),
            add_preference=str(inputs.get("add_preference", "")),
            preference_selected=str(inputs.get("preference_selected", "")),
            ref_context=_get_ref_context(f"建模 算法 {inputs.get('target', '')} {inputs.get('add_preference', '')}"),
        )
    except Exception as e:
        _err(f"本地 Modeling phase1 失败：{e}")
        return None


def call_modeling_phase2_bridge(inputs: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Run modeling phase 2 for RAG, code generation, and result analysis."""
    from workflows.modeling import run_modeling_phase2

    try:
        return run_modeling_phase2(
            ctx=ctx,
            data=str(inputs.get("data", "")),
            df_head=str(inputs.get("df_head", "")),
        )
    except Exception as e:
        _err(f"本地 Modeling phase2 失败：{e}")
        return None


def call_reporting_toc_bridge(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Call the report-outline workflow from UI inputs."""
    from workflows.reporting_toc import run_reporting_toc_workflow

    try:
        def _as_dict(x):
            if isinstance(x, dict):
                return x
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except Exception:
                    return {}
            return {}

        toc_md = inputs.get("toc_md", [])
        if isinstance(toc_md, str):
            try:
                toc_md = json.loads(toc_md)
            except Exception:
                toc_md = [toc_md]
        if not isinstance(toc_md, list):
            toc_md = []

        return run_reporting_toc_workflow(
            load_summary=_as_dict(inputs.get("load_summary")),
            preproc_summary=_as_dict(inputs.get("preproc_summary")),
            visual_summary=_as_dict(inputs.get("visual_summary")),
            coding_summary=_as_dict(inputs.get("coding_summary")),
            load_abstract=str(inputs.get("load_abstract", "")),
            preproc_abstract=str(inputs.get("preproc_abstract", "")),
            visual_abstract=str(inputs.get("visual_abstract", "")),
            coding_abstract=str(inputs.get("coding_abstract", "")),
            selected_full_conten=str(inputs.get("selected_full_conten", "")),
            toc_md=toc_md,
            outline_length=str(inputs.get("outline_length", "标准")),
            report_auto=bool(inputs.get("report_auto", True)),
            user_input=str(inputs.get("user_input", "")),
            add_preference=str(inputs.get("add_preference", "")),
            preference_selected=str(inputs.get("preference_selected", "")),
            ref_context=_get_ref_context(f"报告 分析结论 {inputs.get('add_preference', '')}"),
        )
    except Exception as e:
        _err(f"本地 Reporting_toc workflow 失败：{e}")
        return None


def call_reporting_partly_bridge(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Call the report-writing workflow from UI inputs."""
    from workflows.reporting_partly import ReportGenerationCancelled, run_reporting_partly_workflow

    try:
        cancel_check = inputs.get("_report_cancel_check")
        return run_reporting_partly_workflow(
            toc_text=str(inputs.get("toc_text", "")),
            selected_full_conten=str(inputs.get("selected_full_conten", "")),
            load_abstract=str(inputs.get("load_abstract", "")),
            preproc_abstract=str(inputs.get("preproc_abstract", "")),
            visual_abstract=str(inputs.get("visual_abstract", "")),
            coding_abstract=str(inputs.get("coding_abstract", "")),
            user_input=str(inputs.get("user_input", "")),
            add_preference=str(inputs.get("add_preference", "")),
            preference_select=str(inputs.get("preference_select", "")),
            ref_context=_get_ref_context(f"报告撰写 业务背景 {inputs.get('add_preference', '')}"),
            stage_reference_contexts=inputs.get("stage_reference_contexts") if isinstance(inputs.get("stage_reference_contexts"), dict) else None,
            cancel_check=cancel_check if callable(cancel_check) else None,
        )
    except ReportGenerationCancelled:
        return None
    except Exception as e:
        _err(f"本地 Reporting_partly workflow 失败：{e}")
        return None
