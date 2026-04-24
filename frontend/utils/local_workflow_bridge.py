"""
前端 render 页面 → 本地 workflow 的桥接层。

原 render 页面构造好 inputs dict 后走 Coze HTTP，这里把它替换成本地调用，
返回结构**严格对齐原 Coze** 返回的 data 字段。
"""
from __future__ import annotations

import json
from typing import Any

import streamlit as st

import pandas as pd


def _err(msg: str) -> None:
    st.error(msg)

def _get_ref_context(query: str = "") -> str:
    """从 session_state 中获取参考资料检索结果。"""
    retriever = st.session_state.get("ref_retriever")
    if retriever is None or retriever.is_empty:
        return ""
    return retriever.retrieve_and_format(query or "数据分析", top_k=3)




def call_loading_bridge(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """对应 Coze Loading workflow (7598094351072526389)"""
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
    """对应 Coze Preprocessing workflow (7604840478119706677)"""
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
    """对应 Coze Visualizing workflow (7628850702967930885)"""
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
    """Phase 1: 仅生成 visual_recommendation + refined_suggestions，快速返回。"""
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
    """Phase 2: 代码生成 + 验证 + 图表分析，依赖 phase1 的 ctx。"""
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
    """对应 Coze Modeling workflow (7605874583226056709)"""
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
    """Phase 1: 仅生成 model_suggestion + refined_suggestions，快速返回。"""
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
    """Phase 2: RAG + 代码生成 + 验证 + 结果分析，依赖 phase1 的 ctx。"""
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
    """对应 Coze Reporting_toc workflow (7619618199978508341)"""
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
    """对应 Coze Reporting_partly workflow (7619618317418446901)"""
    from workflows.reporting_partly import run_reporting_partly_workflow

    try:
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
        )
    except Exception as e:
        _err(f"本地 Reporting_partly workflow 失败：{e}")
        return None
