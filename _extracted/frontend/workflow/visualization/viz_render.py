import json
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import requests
import streamlit as st
import streamlit_antd_components as sac

from utils.coze_runtime import resolve_coze_runtime
from utils.page_paths import page_file
from workflow.visualization.viz_coding import vis_code_gen, vis_execution
from workflow.visualization.viz_color import apply_palette_to_figure, vis_palette

COZE_SPACE_ID = "7594748927577554949"
WORKFLOW_ID = "7628850702967930885"
BOT_ID = "7595403958269575173"
DEFAULT_COZE_URL = "https://api.coze.com/v1/workflow/run"
CONNECT_TIMEOUT_SECONDS = 30
WORKFLOW_TIMEOUT_SECONDS = 600


def _maybe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return value

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _stringify_content(value: Any) -> str:
    value = _maybe_json_loads(value)

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return json.dumps(value, ensure_ascii=False)


def clean_and_parse(raw_data: Any):
    if isinstance(raw_data, list):
        return raw_data
    if not isinstance(raw_data, str):
        return None

    content = raw_data.strip()
    try:
        return json.loads(content)
    except Exception:
        try:
            cleaned = content.replace('\\"', '"')
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = json.loads(cleaned)
            return json.loads(cleaned)
        except Exception:
            return None


def _serialize_dataframe_for_workflow(df: pd.DataFrame) -> str:
    safe_df = df.copy()

    for column in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[column]):
            safe_df[column] = safe_df[column].astype(str)

    return safe_df.to_json(orient="records", force_ascii=False)


def _find_nested_field(data: Any, field_name: str) -> Any:
    if isinstance(data, dict):
        if field_name in data:
            return data[field_name]

        for value in data.values():
            nested = _find_nested_field(value, field_name)
            if nested is not None:
                return nested

    if isinstance(data, list):
        for item in data:
            nested = _find_nested_field(item, field_name)
            if nested is not None:
                return nested

    return None


def _normalize_visualization_titles(raw_titles: Any) -> list[str]:
    parsed_titles = _maybe_json_loads(raw_titles)

    if parsed_titles is None:
        return []

    if isinstance(parsed_titles, str):
        text = parsed_titles.strip()
        if not text:
            return []
        return [line.strip() for line in text.splitlines() if line.strip()]

    if isinstance(parsed_titles, dict):
        candidate_keys = ("tu_title", "titles", "data", "items")
        for key in candidate_keys:
            if key in parsed_titles:
                return _normalize_visualization_titles(parsed_titles.get(key))
        return [
            str(value).strip()
            for value in parsed_titles.values()
            if str(value).strip()
        ]

    if isinstance(parsed_titles, list):
        normalized_titles: list[str] = []
        for item in parsed_titles:
            if isinstance(item, dict):
                candidate = (
                    item.get("tu_title")
                    or item.get("name")
                    or item.get("label")
                    or item.get("text")
                )
            else:
                candidate = item

            candidate_text = str(candidate).strip() if candidate is not None else ""
            if candidate_text:
                normalized_titles.append(candidate_text)
        return normalized_titles

    fallback = str(parsed_titles).strip()
    return [fallback] if fallback else []


def _set_visualization_titles(title_items: list[str]) -> None:
    normalized_titles = [str(item).strip() for item in title_items]
    st.session_state.tu_title = normalized_titles

    workflow_result = st.session_state.get("viz_workflow_result")
    if isinstance(workflow_result, dict):
        workflow_result["tu_title"] = normalized_titles


def _extract_title_from_figure(fig: go.Figure) -> str:
    if not isinstance(fig, go.Figure):
        return ""

    try:
        title_obj = fig.layout.title
        if title_obj is None:
            return ""
        title_text = getattr(title_obj, "text", "") or ""
        return str(title_text).strip()
    except Exception:
        return ""


def _safe_download_name(text: str, fallback: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", str(text or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._")
    return cleaned or fallback


def _clear_visualization_title_inputs() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("viz_title_input_"):
            st.session_state.pop(key, None)


def _has_usable_data(source: Any) -> bool:
    if source is None:
        return False

    if isinstance(source, pd.DataFrame):
        return not source.empty

    if isinstance(source, np.ndarray):
        return source.size > 0

    if isinstance(source, str):
        return bool(source.strip())

    if isinstance(source, (list, dict)):
        return bool(source)

    return True


def _resolve_visualization_source(preproc_agent, load_agent) -> tuple[Any, str | None]:
    processed_df = preproc_agent.load_processed_df()
    if _has_usable_data(processed_df):
        return processed_df, "processed"

    summary_2 = st.session_state.get("summary_2")
    if isinstance(summary_2, dict):
        summary_processed_df = summary_2.get("processed_df")
        if _has_usable_data(summary_processed_df):
            return summary_processed_df, "processed"

    cached_processed_df = st.session_state.get("prep_result_from_summary_2")
    if _has_usable_data(cached_processed_df):
        return cached_processed_df, "processed"

    raw_df = load_agent.load_df()
    if _has_usable_data(raw_df):
        return raw_df, "raw"

    return None, None


def _source_to_dataframe(source: Any) -> pd.DataFrame | None:
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, np.ndarray):
        return pd.DataFrame(source)

    if isinstance(source, str):
        records = clean_and_parse(source)
        if records is None:
            return None
        try:
            return pd.DataFrame(records)
        except Exception:
            return None

    return None


def _build_visualization_inputs(
    source_data: Any,
    agent,
    user_input: str = "",
    vis_auto: bool = True,
) -> dict[str, Any] | None:
    if isinstance(source_data, pd.DataFrame):
        data_str = _serialize_dataframe_for_workflow(source_data)
        df_obj = source_data.copy()
    elif isinstance(source_data, np.ndarray):
        df_obj = pd.DataFrame(source_data)
        data_str = _serialize_dataframe_for_workflow(df_obj)
    elif isinstance(source_data, str):
        data_str = source_data
        records = clean_and_parse(source_data)
        if records is None:
            return None
        df_obj = pd.DataFrame(records)
    else:
        return None

    columns = df_obj.columns.astype(str).tolist()
    head_dict_str = json.dumps(df_obj.head(5).to_dict(orient="list"), ensure_ascii=False)

    preference_selected = st.session_state.get("preference_selected")
    add_preference = st.session_state.get("add_preference")
    color = agent.load_color()

    return {
        "data": data_str,
        "user_input": user_input or "",
        "preference_selected": _stringify_content(preference_selected),
        "add_preference": add_preference or "",
        "color": _stringify_content(color),
        "shape0": int(df_obj.shape[0]),
        "shape1": int(df_obj.shape[1]),
        "cols": columns,
        "def_head": head_dict_str,
        "vis_auto": bool(vis_auto),
    }


def _normalize_visualization_workflow_result(result: Any) -> dict[str, Any] | None:
    result = _maybe_json_loads(result)
    if not isinstance(result, dict):
        return None

    normalized = dict(result)
    normalized["tu_title"] = _stringify_content(
        _find_nested_field(result, "tu_title")
    )
    normalized["full"] = _stringify_content(_find_nested_field(result, "full"))
    normalized["abstract_3"] = _stringify_content(_find_nested_field(result, "abstract_3"))
    normalized["summary_3"] = _maybe_json_loads(_find_nested_field(result, "summary_3"))
    normalized["visual_recommendatio"] = _stringify_content(
        _find_nested_field(result, "visual_recommendatio")
    )
    normalized["final_code"] = _stringify_content(_find_nested_field(result, "final_code"))
    return normalized


def call_coze_workflow_visualization(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """本地化版本：改走本地 Visualizing workflow。"""
    from utils.local_workflow_bridge import call_visualizing_bridge

    # 补齐本地 workflow 可能需要的偏好字段
    inputs = dict(inputs)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    inputs.setdefault("preference_selected", st.session_state.get("preference_selected") or "")

    result = call_visualizing_bridge(inputs)
    if result is None:
        return None

    normalized = _normalize_visualization_workflow_result(result)
    if normalized is None:
        st.error("可视化工作流返回结构异常，未解析到有效结果。")
        return None
    return normalized


def vis_result(agent) -> None:
    fig_desc_list = agent.load_fig()
    total = len(fig_desc_list)
    if total == 0:
        return

    title_items = _normalize_visualization_titles(st.session_state.get("tu_title"))
    if len(title_items) < total:
        title_items.extend([""] * (total - len(title_items)))
    show_analysis = st.session_state.get("viz_desc_switch", False)
    current_page_key = "viz_current_page"

    if current_page_key not in st.session_state:
        st.session_state[current_page_key] = 1

    st.session_state[current_page_key] = max(1, min(int(st.session_state[current_page_key]), total))

    page_size = 1
    st.markdown(
        """
        <style>
        .ant-pagination {
            display: flex !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            justify-content: center !important;
            white-space: nowrap !important;
        }
        .ant-pagination-item,
        .ant-pagination-prev,
        .ant-pagination-next,
        .ant-pagination-jump-prev,
        .ant-pagination-jump-next {
            flex: 0 0 auto !important;
        }
        .viz-page-indicator {
            text-align: center;
            color: #374151;
            font-size: 1rem;
            line-height: 1;
            margin-top: -0.45rem;
        }
        div[data-testid="stTextInput"] {
            margin-top: -0.55rem !important;
            margin-bottom: 0.2rem !important;
        }
        div[data-testid="stTextInput"] input {
            text-align: center !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    selected_page = sac.pagination(
        total=total,
        index=st.session_state[current_page_key],
        page_size=page_size,
        align="center",
        jump=False,
        show_total=False,
        variant="filled",
        color="#44658C",
        key="viz_pagination",
    )

    current_page = int(selected_page)
    if current_page != st.session_state[current_page_key]:
        st.session_state[current_page_key] = current_page
        st.rerun()

    st.markdown(
        f'<div class="viz-page-indicator">{current_page}-{total}</div>',
        unsafe_allow_html=True,
    )

    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total)
    for idx, item in enumerate(fig_desc_list[start_idx:end_idx], start=start_idx):
        fig = item.get("base_fig", item["fig"])
        desc = item["desc"]

        if isinstance(fig, str):
            try:
                fig = pio.from_json(fig)
            except Exception:
                continue

        if not isinstance(fig, go.Figure):
            continue

        colors = agent.load_color() or []
        base_display_fig = apply_palette_to_figure(fig, colors, idx) if colors else go.Figure(fig)
        display_fig = go.Figure(base_display_fig)
        display_title = _extract_title_from_figure(display_fig) or _extract_title_from_figure(fig)
        if not title_items[idx] and display_title:
            title_items[idx] = display_title
            _set_visualization_titles(title_items)

        input_key = f"viz_title_input_{idx}"
        default_title = (title_items[idx] or display_title).strip()
        if input_key not in st.session_state or not str(st.session_state.get(input_key, "")).strip():
            st.session_state[input_key] = default_title
        edited_title = st.session_state.get(input_key, default_title)
        if edited_title != title_items[idx]:
            title_items[idx] = edited_title
            _set_visualization_titles(title_items)
        if edited_title.strip():
            display_fig.update_layout(title=edited_title.strip())

        st.plotly_chart(display_fig, use_container_width=True, key=f"fig_{idx}")

        title_columns = st.columns([1.2, 4.6, 1.2])
        with title_columns[1]:
            edited_title = st.text_input(
                f"图表标题 {idx + 1}",
                key=input_key,
                label_visibility="collapsed",
                placeholder="请输入图表标题",
            )

        if edited_title != title_items[idx]:
            title_items[idx] = edited_title
            _set_visualization_titles(title_items)
            display_fig.update_layout(title=edited_title.strip())

        download_name = _safe_download_name(
            edited_title or f"visualization_{idx + 1}",
            fallback=f"visualization_{idx + 1}",
        )
        download_bytes = pio.to_image(
            display_fig,
            format="png",
            width=1800,
            height=1200,
            scale=2,
        )
        download_columns = st.columns([1.2, 1.6, 1.2])
        with download_columns[1]:
            st.download_button(
                "下载图片",
                data=download_bytes,
                file_name=f"{download_name}.png",
                mime="image/png",
                key=f"viz_download_{idx}",
                use_container_width=True,
            )

        if show_analysis and desc is not None:
            st.write(desc)


def _clear_visualization_workflow_state(agent) -> None:
    agent.clear_memory()
    agent.suggestion = None
    agent.code = None
    agent.user_input = None
    agent.error = None
    agent.fig_desc_list = []
    agent.save_fig([])
    agent.finish_auto_task = False
    _clear_visualization_title_inputs()
    st.session_state.pop("viz_workflow_result", None)
    st.session_state.pop("viz_suggestion", None)
    st.session_state.pop("tu_title", None)
    st.session_state.pop("full", None)
    st.session_state.pop("abstract_3", None)
    st.session_state.pop("summary_3", None)
    st.session_state.pop("visual_recommendatio", None)
    st.session_state.pop("final_code", None)
    st.session_state.pop("viz_desc_switch", None)


def _reset_visualization_outputs(agent) -> None:
    agent.suggestion = None
    agent.code = None
    agent.error = None
    agent.fig_desc_list = []
    agent.save_fig([])
    agent.finish_auto_task = False
    _clear_visualization_title_inputs()
    st.session_state.pop("viz_workflow_result", None)
    st.session_state.pop("viz_suggestion", None)
    st.session_state.pop("tu_title", None)
    st.session_state.pop("full", None)
    st.session_state.pop("abstract_3", None)
    st.session_state.pop("summary_3", None)
    st.session_state.pop("visual_recommendatio", None)
    st.session_state.pop("final_code", None)
    st.session_state.pop("viz_desc_switch", None)
    st.session_state.pop("viz_current_page", None)
    st.session_state.pop("viz_pagination", None)


def _request_visualization_recommendation(agent, source_data: Any, user_input: str) -> None:
    _reset_visualization_outputs(agent)

    inputs = _build_visualization_inputs(
        source_data=source_data,
        agent=agent,
        user_input=user_input,
        vis_auto=True,
    )
    if inputs is None:
        st.error("无法从当前可用数据构造可视化工作流输入，请检查预处理结果或原始上传数据是否可解析。")
        return

    with st.spinner("正在生成可视化推荐，预计需要 2-3 分钟，请耐心等待..."):
        workflow_result = call_coze_workflow_visualization(inputs)

    if not workflow_result:
        return

    tu_title = _normalize_visualization_titles(workflow_result.get("tu_title", ""))
    full = workflow_result.get("full")
    abstract_3 = workflow_result.get("abstract_3")
    summary_3 = workflow_result.get("summary_3")
    visual_recommendatio = workflow_result.get("visual_recommendatio", "")
    final_code = workflow_result.get("final_code", "")

    st.session_state.viz_workflow_result = workflow_result
    st.session_state.tu_title = tu_title
    st.session_state.full = full
    st.session_state.abstract_3 = abstract_3
    st.session_state.summary_3 = summary_3
    st.session_state.visual_recommendatio = visual_recommendatio
    st.session_state.final_code = final_code
    st.session_state.viz_suggestion = visual_recommendatio
    agent.save_suggestion(visual_recommendatio)

    agent.add_memory(
        {
            "role": "assistant",
            "content": workflow_result,
        }
    )
    agent.finish_auto()
    st.rerun()


def _has_visualization_result(agent) -> bool:
    suggestion = (
        st.session_state.get("visual_recommendatio")
        or st.session_state.get("viz_suggestion")
        or agent.load_suggestion()
    )
    return bool(suggestion)


def _has_visualization_execution_result(agent) -> bool:
    return bool(agent.load_fig())


def vis_chat(agent, source_data: Any, auto: bool = False):
    with st.chat_message("assistant"):
        st.write(
            "我是 Autostat 数据分析助手。\n\n"
            "你可以在下方输入具体可视化需求，或者直接点击按钮获取可视化推荐。"
        )

        columns = st.columns(2)
        with columns[0]:
            analyze_clicked = st.button("🔍 可视化推荐", key="viz_suggest", use_container_width=True)
        with columns[1]:
            clear_viz_suggest = st.button("♻️ 清除可视化分析", key="clear_viz_suggest", use_container_width=True)

        if clear_viz_suggest:
            _clear_visualization_workflow_state(agent)
            st.rerun()

    chat_history = agent.load_memory()
    for idx, entry in enumerate(chat_history):
        role = entry.get("role")
        content = entry.get("content")

        with st.chat_message(role):
            if isinstance(content, str):
                st.write(content)
            elif isinstance(content, dict):
                suggestion = (
                    _stringify_content(content.get("visual_recommendatio"))
                    or _stringify_content(content.get("abstract_3"))
                    or _stringify_content(content.get("tu_title"))
                )
                if suggestion:
                    st.write(suggestion)
            elif isinstance(content, go.Figure):
                st.plotly_chart(content, use_container_width=True, key=f"chart-{idx}")

    already_generated = any(
        entry["role"] == "assistant" and isinstance(entry.get("content"), dict)
        for entry in chat_history
    )

    if auto and _has_visualization_execution_result(agent) and not agent.finish_auto_task:
        agent.finish_auto()
        st.rerun()

    if analyze_clicked or (auto and not already_generated):
        user_prompt = agent.load_user_input() or ""
        prompt_text = "请帮我做可视化分析"
        st.chat_message("user").write(prompt_text)
        agent.add_memory({"role": "user", "content": prompt_text})
        _request_visualization_recommendation(agent, source_data, user_prompt)

    user_input = st.chat_input("请输入需求，例如“请给我一些可视化建议”")
    if user_input:
        agent.save_user_input(user_input)
        st.chat_message("user").write(user_input)
        agent.add_memory({"role": "user", "content": user_input})
        _request_visualization_recommendation(agent, source_data, user_input)


if __name__ == "__main__":
    st.title("统计可视化分析")
    st.markdown("---")

    preproc_agent = st.session_state.data_preprocess_agent
    load_agent = st.session_state.data_loading_agent
    planner = st.session_state.planner_agent
    auto = bool(st.session_state.auto_mode and planner.vis_auto)

    source_data, source_kind = _resolve_visualization_source(preproc_agent, load_agent)
    df = _source_to_dataframe(source_data)

    if df is None:
        st.warning("请先在数据导入页面加载数据。")
        st.stop()

    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    agent = st.session_state.visualization_agent
    agent.add_df(df_shuffled)

    if st.session_state.auto_mode:
        if planner.vis_auto and _has_visualization_execution_result(agent):
            planner.finish_vis_auto()
            st.switch_page(page_file("modeling", "modeling_render.py"))

    code = agent.load_code()
    code_expand = code is not None

    fig = agent.load_fig()
    fig_expand = bool(fig)

    if source_kind == "raw":
        st.caption("当前未对原始数据进行预处理，后续将基于原始数据进行分析")

    columns = st.columns(2)
    with columns[0].expander("配色选择", True):
        vis_palette(agent)
    with columns[1].expander("可视化建议", True):
        vis_chat(agent, source_data, auto)
        vis_code_gen(agent, auto=auto)
    with columns[0].expander("可视化执行", code_expand):
        vis_execution(agent, auto=auto)
    with columns[0].expander("可视化结果", fig_expand):
        vis_result(agent)
