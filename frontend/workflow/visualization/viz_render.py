import base64
import hashlib
import json
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import streamlit as st
import streamlit_antd_components as sac

from core.figure_artifacts import (
    VISUALIZATION_FIGURE_ARTIFACTS_KEY,
    successful_figure_artifacts,
)
from core.plotly_serialization import figure_to_json
from utils.i18n import bt, get_language
from utils.page_paths import page_file
from utils.suggestion_state import (
    add_requirement,
    base_requirements_text,
    can_auto_repair,
    clear_suggestion_state,
    confirm_active_suggestion,
    get_suggestion_state,
    mark_code_draft,
    queue_initial_request,
    queue_revision_request,
    record_auto_repair,
    record_validated_code,
    record_validation_failure,
    replace_active_suggestion,
    revision_fallback_text,
    take_pending_code_revision,
    take_pending_initial_request,
    take_pending_revision,
    visible_messages,
)
from utils.workflow_state import (
    current_dataset_fingerprint,
    invalidate_from,
    record_stage_status,
    stable_fingerprint,
    stage_is_current,
)
from workflow.visualization.viz_coding import vis_code_gen, vis_execution
from workflow.visualization.viz_color import apply_palette_to_figure, vis_palette

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


def _safe_visualization_page(value: Any, total: int) -> int:
    try:
        page = int(value)
    except (TypeError, ValueError):
        page = 1
    return max(1, min(page, max(1, total)))


def _figure_download_cache_key(fig: go.Figure) -> str:
    return hashlib.sha256(figure_to_json(fig).encode("utf-8")).hexdigest()


def _cached_figure_download_bytes(fig: go.Figure) -> bytes | None:
    cache = st.session_state.get("viz_download_image_cache")
    if not isinstance(cache, dict):
        return None
    cached = cache.get(_figure_download_cache_key(fig))
    return cached if isinstance(cached, bytes) else None


def _figure_download_bytes(fig: go.Figure) -> tuple[bytes | None, str]:
    fig_json = figure_to_json(fig)
    cache_key = _figure_download_cache_key(fig)
    cache = st.session_state.setdefault("viz_download_image_cache", {})
    cached = cache.get(cache_key)
    if isinstance(cached, bytes):
        return cached, "ok"

    try:
        from workflows.visualizing import _render_plotly_image_base64

        status, encoded = _render_plotly_image_base64(fig_json)
        image_bytes = base64.b64decode(encoded) if status == "ok" and encoded else None
    except Exception:
        status, image_bytes = "error", None

    if image_bytes:
        cache[cache_key] = image_bytes
        while len(cache) > 3:
            cache.pop(next(iter(cache)))
    return image_bytes, status


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
        return [
            cleaned
            for line in text.splitlines()
            if (cleaned := _clean_visualization_title_text(line))
        ]

    if isinstance(parsed_titles, dict):
        candidate_keys = ("tu_title", "titles", "data", "items")
        for key in candidate_keys:
            if key in parsed_titles:
                return _normalize_visualization_titles(parsed_titles.get(key))
        return [
            cleaned
            for value in parsed_titles.values()
            if (cleaned := _clean_visualization_title_text(value))
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

            candidate_text = _clean_visualization_title_text(candidate)
            if candidate_text:
                normalized_titles.append(candidate_text)
        return normalized_titles

    fallback = _clean_visualization_title_text(parsed_titles)
    return [fallback] if fallback else []


def _clean_visualization_title_text(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^\[?\s*FIG\s*[:：]\s*\d+\s*\]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^(?:图(?:表)?|figure|fig\.?|chart)\s*(?:x|\d+)\s*[|｜:：、.．\-—–]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:图(?:表)?|figure|fig\.?|chart)\s*(?:x|\d+)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^\d+\s*[|｜]\s*", "", text)
    return text.strip(" |｜:：、.．-—–\t\r\n")


def _rebuild_visualization_full_from_summary(summary_3: Any) -> str:
    if not isinstance(summary_3, dict):
        return ""
    fig_analysis = summary_3.get("fig_analysis")
    if not isinstance(fig_analysis, list):
        return ""
    parts: list[str] = []
    for index, item in enumerate(fig_analysis):
        if isinstance(item, dict):
            title = _clean_visualization_title_text(item.get("title") or item.get("tu_title") or "")
            analysis = str(item.get("analysis") or item.get("desc") or "").strip()
        else:
            title = ""
            analysis = str(item or "").strip()
        title_line = bt("图题", "Title") + f"：{title}" if title else ""
        content = "\n".join(part for part in (title_line, analysis) if part)
        parts.append(f"[FIG:{index}] {content}".strip())
    return "\n\n".join(parts)


def _set_visualization_titles(title_items: list[str]) -> None:
    normalized_titles = [_clean_visualization_title_text(item) for item in title_items]
    st.session_state.tu_title = normalized_titles

    visualization_agent = st.session_state.get("visualization_agent")
    if visualization_agent is not None:
        try:
            fig_desc_list = visualization_agent.load_fig() or []
        except Exception:
            fig_desc_list = []
        if isinstance(fig_desc_list, list):
            for index, title in enumerate(normalized_titles):
                if index >= len(fig_desc_list):
                    break
                if isinstance(fig_desc_list[index], dict):
                    fig_desc_list[index]["title"] = title
            try:
                visualization_agent.save_fig(fig_desc_list)
            except Exception:
                pass

    summary_3 = st.session_state.get("summary_3")
    if isinstance(summary_3, dict):
        summary_3 = dict(summary_3)
        fig_analysis = summary_3.get("fig_analysis")
        if isinstance(fig_analysis, list):
            updated_analysis = []
            for index, item in enumerate(fig_analysis):
                if isinstance(item, dict):
                    updated_item = dict(item)
                    if index < len(normalized_titles) and normalized_titles[index]:
                        updated_item["title"] = normalized_titles[index]
                    updated_analysis.append(updated_item)
                else:
                    updated_analysis.append(item)
            summary_3["fig_analysis"] = updated_analysis
            st.session_state.summary_3 = summary_3
            rebuilt_full = _rebuild_visualization_full_from_summary(summary_3)
            if rebuilt_full:
                st.session_state.full = rebuilt_full

    workflow_result = st.session_state.get("viz_workflow_result")
    if isinstance(workflow_result, dict):
        workflow_result["tu_title"] = normalized_titles
        if isinstance(st.session_state.get("summary_3"), dict):
            workflow_result["summary_3"] = st.session_state.get("summary_3")
        if st.session_state.get("full"):
            workflow_result["full"] = st.session_state.get("full")


def _figure_key_suffix(item: Any, fig: go.Figure) -> str:
    if isinstance(item, dict):
        fingerprint = str(item.get("figure_fingerprint") or "").strip()
        if fingerprint:
            return fingerprint[:12]
    try:
        return hashlib.sha256(figure_to_json(fig).encode("utf-8")).hexdigest()[:12]
    except Exception:
        return hashlib.sha256(str(fig).encode("utf-8", errors="ignore")).hexdigest()[:12]



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


_ANALYSIS_ECHO_MARKERS = (
    "i understand the requirement",
    "i understand your requirement",
    "as a senior data visualization expert",
    "as requested",
    "chart generated",
    "field type overview",
    "here is",
    "i will",
    "the following",
    "visualization recommendation",
    "user-facing response",
    "chart title, report excerpt",
    "polished professional english",
    "preserving all dataset field names",
    "please provide the data",
    "language requirement",
    "output language",
    "output requirement",
    "system prompt",
    "图表已生成",
    "字段类型概览",
    "可视化建议",
    "我理解",
    "以下是",
    "好的",
    "系统提示",
    "用户指令",
    "语言要求",
    "输出要求",
    "输出规则",
)
_CHINESE_LABEL_ENGLISH_REPLACEMENTS = (
    ("relationship between", "关系"),
    ("distribution of", "分布"),
    ("comparison of", "比较"),
    ("changes in", "变化"),
    ("change in", "变化"),
    ("relationship", "关系"),
    ("comparison", "比较"),
    ("distribution", "分布"),
    ("correlation", "相关性"),
    ("accuracy", "准确率"),
    ("monetary", "金额"),
    ("amount", "金额"),
    ("revenue", "收入"),
    ("income", "收入"),
    ("profit", "利润"),
    ("sales", "销售额"),
    ("price", "价格"),
    ("cost", "成本"),
    ("quantity", "数量"),
    ("frequency", "频次"),
    ("recency", "近度"),
    ("duration", "时长"),
    ("score", "得分"),
    ("gender", "性别"),
    ("category", "类别"),
    ("class", "类别"),
    ("group", "组别"),
    ("cluster", "聚类"),
    ("trend", "变化趋势"),
    ("rate", "率"),
    ("count", "计数"),
    ("value", "数值"),
    ("time", "时间"),
    ("date", "日期"),
    ("year", "年份"),
    ("month", "月份"),
    ("day", "日期"),
)


def _contains_cjk_text(value: Any) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", str(value or "")))


def _contains_ascii_letters(value: Any) -> bool:
    return bool(re.search(r"[A-Za-z]", str(value or "")))


def _replace_english_terms_for_chinese_label(value: Any) -> str:
    text = str(value or "")
    for src, dst in _CHINESE_LABEL_ENGLISH_REPLACEMENTS:
        pattern = rf"(?<![A-Za-z]){re.escape(src)}(?![A-Za-z])"
        text = re.sub(pattern, dst, text, flags=re.IGNORECASE)
    return text


def _has_disallowed_english_words_for_chinese(value: Any) -> bool:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(value or ""))
    for token in tokens:
        if len(token) == 1 and token.isupper():
            continue
        if 2 <= len(token) <= 8 and token.isupper():
            continue
        return True
    return False


def _limit_chinese_title(title: str, max_chars: int = 20) -> str:
    text = re.sub(r"\s+", "", str(title or "")).strip("，,。.;；:：")
    return text[:max_chars].strip("，,。.;；:：")


def _clean_chinese_label(value: Any) -> str:
    text = re.sub(r"<[^>]+>", "", str(value or ""))
    text = text.replace("_", " ")
    text = _replace_english_terms_for_chinese_label(text)
    text = re.sub(
        r"(频率)?分布直方图|柱状图|条形图|折线图|散点图|箱线图|小提琴图|热力图|饼图|雷达图",
        "",
        text,
    )
    text = re.sub(
        r"\b(?:bar chart|line chart|scatter plot|histogram|box plot|heatmap)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = _limit_chinese_title(text)
    if not text or _has_disallowed_english_words_for_chinese(text):
        return ""
    return text


def _chinese_title_needs_repair(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if _contains_ascii_letters(text) and not _contains_cjk_text(text):
        return True
    return _has_disallowed_english_words_for_chinese(text)


def _is_english_ui() -> bool:
    return str(get_language()).lower().startswith("en")


def _clean_english_label(value: Any) -> str:
    text = re.sub(r"<[^>]+>", "", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip(" ,.;:")
    if not text or _contains_cjk_text(text):
        return ""
    return text[:80]


def _figure_axis_title(fig: go.Figure, axis_name: str) -> str:
    if not isinstance(fig, go.Figure):
        return ""
    axis = getattr(fig.layout, axis_name, None)
    title = getattr(axis, "title", None) if axis is not None else None
    text = getattr(title, "text", "") if title is not None else ""
    return str(text or "").strip()


def _figure_trace_types(fig: go.Figure) -> set[str]:
    if not isinstance(fig, go.Figure):
        return set()
    return {str(getattr(trace, "type", "") or "").lower() for trace in fig.data}


def _first_trace_name(fig: go.Figure) -> str:
    if not isinstance(fig, go.Figure):
        return ""
    for trace in fig.data:
        label = _clean_english_label(getattr(trace, "name", ""))
        if label:
            return label
    return ""


def _first_trace_chinese_name(fig: go.Figure) -> str:
    if not isinstance(fig, go.Figure):
        return ""
    for trace in fig.data:
        label = _clean_chinese_label(getattr(trace, "name", ""))
        if label:
            return label
    return ""


def _limit_english_title_words(title: str, max_words: int = 14) -> str:
    words = re.sub(r"\s+", " ", str(title or "")).strip().split()
    return " ".join(words[:max_words]).strip(" ,.;:") if words else ""


def _fallback_english_title_from_figure(fig: go.Figure) -> str:
    existing_title = _extract_title_from_figure(fig)
    x_axis = _clean_english_label(_figure_axis_title(fig, "xaxis"))
    y_axis = _clean_english_label(_figure_axis_title(fig, "yaxis"))
    trace_types = _figure_trace_types(fig)
    loess_prefix = "LOESS-Smoothed " if "loess" in existing_title.lower() else ""

    if x_axis and y_axis:
        return _limit_english_title_words(f"{loess_prefix}{y_axis} vs {x_axis}")

    variable = y_axis or x_axis or _first_trace_name(fig)
    if variable:
        if {"scatter", "scattergl"} & trace_types:
            return _limit_english_title_words(f"{variable} Relationship")
        if {"line", "scatter"} & trace_types:
            return _limit_english_title_words(f"{variable} Trend")
        return _limit_english_title_words(f"{variable} Distribution")

    return "Chart Summary"


def _fallback_chinese_title_from_figure(fig: go.Figure) -> str:
    existing_title = _clean_chinese_label(_extract_title_from_figure(fig))
    if existing_title:
        return existing_title

    x_axis = _clean_chinese_label(_figure_axis_title(fig, "xaxis"))
    y_axis = _clean_chinese_label(_figure_axis_title(fig, "yaxis"))
    trace_types = _figure_trace_types(fig)

    variable = y_axis or x_axis or _first_trace_chinese_name(fig)
    if "histogram" in trace_types:
        variable = x_axis or y_axis or _first_trace_chinese_name(fig)
    if {"histogram", "box", "violin"} & trace_types and variable:
        return _limit_chinese_title(f"{variable}分布")
    if {"scatter", "scattergl"} & trace_types and x_axis and y_axis:
        return _limit_chinese_title(f"{x_axis}与{y_axis}关系")
    if {"line", "scatter"} & trace_types and variable:
        return _limit_chinese_title(f"{variable}变化趋势")
    if {"bar", "pie"} & trace_types and variable:
        return _limit_chinese_title(f"{variable}比较")
    if x_axis and y_axis:
        return _limit_chinese_title(f"{x_axis}与{y_axis}关系")
    if variable:
        return _limit_chinese_title(f"{variable}分布")

    return "主要变量分布"


def _is_unhelpful_analysis(value: Any, *, english_ui: bool | None = None) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not text:
        return True
    if any(marker in text for marker in _ANALYSIS_ECHO_MARKERS):
        return True
    if english_ui is True:
        return _contains_cjk_text(text)
    if english_ui is False:
        return not _contains_cjk_text(text)
    return False


def _fallback_chart_analysis_from_figure(fig: go.Figure) -> str:
    x_axis = _clean_english_label(_figure_axis_title(fig, "xaxis"))
    y_axis = _clean_english_label(_figure_axis_title(fig, "yaxis"))
    zh_x_axis = _clean_chinese_label(_figure_axis_title(fig, "xaxis"))
    zh_y_axis = _clean_chinese_label(_figure_axis_title(fig, "yaxis"))
    trace_types = _figure_trace_types(fig)
    if "histogram" in trace_types:
        variable = x_axis or y_axis or _first_trace_name(fig)
        zh_variable = zh_x_axis or zh_y_axis or _first_trace_chinese_name(fig)
    else:
        variable = y_axis or x_axis or _first_trace_name(fig)
        zh_variable = zh_y_axis or zh_x_axis or _first_trace_chinese_name(fig)

    english_text = (
        f"This chart summarizes the distribution of {variable}. "
        "It highlights the visible spread, concentration, and potential outlying values."
        if {"histogram", "box", "violin"} & trace_types and variable
        else (
            f"This chart shows {y_axis} in relation to {x_axis}. "
            "It helps identify the visible pattern, spread, and potential association between the variables."
            if x_axis and y_axis
            else (
                f"This chart summarizes {variable}. "
                "It highlights the variable's visible distribution, grouping, or trend in the generated visualization."
                if variable
                else "This chart summarizes the selected variables and highlights the visible distribution, relationship, or trend."
            )
        )
    )
    chinese_text = (
        f"这张图概括了{zh_variable}的分布特征，可用于识别数值的集中区间、离散程度和可能的极端值。"
        if {"histogram", "box", "violin"} & trace_types and zh_variable
        else (
            f"这张图展示了{zh_y_axis}与{zh_x_axis}之间的关系，可用于识别两个变量的可见模式、离散程度和潜在关联。"
            if zh_x_axis and zh_y_axis
            else (
                f"这张图概括了{zh_variable}的可见分布、分组或变化趋势，有助于理解数据中的主要特征。"
                if zh_variable
                else "这张图展示了所选变量的分布、关系或变化趋势，可用于辅助判断数据中的主要模式。"
            )
        )
    )

    return bt(
        chinese_text,
        english_text,
    )


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
    dataset_fingerprint = current_dataset_fingerprint(st.session_state)
    stage_states = st.session_state.get("workflow_stage_states") or {}
    prep_state = stage_states.get("preprocessing") if isinstance(stage_states, dict) else None
    prep_succeeded = stage_is_current(
        st.session_state,
        "preprocessing",
        input_fingerprint=dataset_fingerprint,
    )

    if prep_succeeded:
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

    if isinstance(prep_state, dict) and prep_state.get("status") in {"failed", "running"}:
        return None, "preprocessing_failed"

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
        "language": get_language(),
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


def call_visualization_workflow(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Run the local visualization workflow."""
    from utils.local_workflow_bridge import call_visualizing_bridge

    inputs = dict(inputs)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    inputs.setdefault("preference_selected", st.session_state.get("preference_selected") or "")

    result = call_visualizing_bridge(inputs)
    if result is None:
        return None

    normalized = _normalize_visualization_workflow_result(result)
    if normalized is None:
        st.error(
            bt(
                "可视化工作流返回结构异常，未解析到有效结果。",
                "The visualization workflow returned an invalid structure, and no usable result was found.",
            )
        )
        return None
    return normalized


def _call_visualization_phase1(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Phase 1: 仅生成 suggestion，快速返回给前端展示。"""
    from utils.local_workflow_bridge import call_visualizing_phase1_bridge

    inputs = dict(inputs)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    inputs.setdefault("preference_selected", st.session_state.get("preference_selected") or "")
    return call_visualizing_phase1_bridge(inputs)


def _call_visualization_phase2(inputs: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Phase 2: 代码生成 + 验证 + 分析。"""
    from utils.local_workflow_bridge import call_visualizing_phase2_bridge

    inputs = dict(inputs)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    inputs.setdefault("preference_selected", st.session_state.get("preference_selected") or "")

    result = call_visualizing_phase2_bridge(inputs, ctx)
    if result is None:
        return None

    normalized = _normalize_visualization_workflow_result(result)
    if normalized is None:
        st.error(
            bt(
                "可视化代码生成阶段返回结构异常。",
                "The visualization code-generation stage returned an invalid structure.",
            )
        )
        return None
    return normalized


def _generate_visualization_code_draft(agent) -> None:
    from utils.local_workflow_bridge import call_visualizing_validated_code_bridge

    state = get_suggestion_state(st.session_state, "visualization")
    ctx = st.session_state.get("_viz_phase1_ctx")
    inputs = st.session_state.get("_viz_phase2_inputs")
    if not isinstance(ctx, dict) or not isinstance(inputs, dict):
        st.error(bt("当前可视化建议上下文已失效，请重新生成建议。", "The visualization recommendation context has expired. Generate it again."))
        return
    with st.spinner(bt("正在生成并验证可视化代码，失败时将自动修复，最多尝试5次...", "Generating and validating visualization code; up to five repair attempts...")):
        result = call_visualizing_validated_code_bridge(inputs, ctx)
    code = str((result or {}).get("code") or "").strip()
    if not code:
        st.error(bt("未能生成可视化代码。", "No visualization code was generated."))
        return
    agent.save_code(code)
    attempts = int((result or {}).get("attempts") or 0)
    if (result or {}).get("success"):
        record_validated_code(state, code, attempts=attempts)
        if (result or {}).get("validation_status") == "partial":
            missing = list((result or {}).get("missing_charts") or [])
            st.warning(bt(
                f"代码可执行并保留可用图表，但有 {len(missing)} 张已确认图表未返回；执行后会显示具体缺失项。",
                f"The code is executable and keeps available figures, but {len(missing)} confirmed chart(s) were not returned; execution will show the missing items.",
            ))
        else:
            st.success(bt(f"代码已生成并通过验证（第{attempts}/5次）。请点击执行可视化。", f"Code passed validation on attempt {attempts}/5. Run visualization to publish the charts."))
    else:
        record_validation_failure(state, code, str((result or {}).get("error") or "未生成可执行代码。"), attempts=attempts or 5)
        st.error(bt("可视化代码连续5次未通过验证，已停止自动修复。", "Visualization code failed validation five times; auto-repair stopped."))
    st.rerun()


def _repair_visualization_code_draft(agent) -> None:
    from utils.local_workflow_bridge import call_visualizing_validated_code_bridge

    state = get_suggestion_state(st.session_state, "visualization")
    ctx = st.session_state.get("_viz_phase1_ctx")
    inputs = st.session_state.get("_viz_phase2_inputs")
    error_text = str(state.get("last_execution_error") or "")
    if not isinstance(ctx, dict) or not isinstance(inputs, dict) or not error_text or not can_auto_repair(state):
        return
    state["repair_in_progress"] = True
    with st.spinner(bt("正在自动修复代码，失败时将继续修复，最多尝试5次...", "Automatically repairing code; up to five attempts...")):
        result = call_visualizing_validated_code_bridge(inputs, ctx, str(agent.load_code() or ""))
    code = str((result or {}).get("code") or "").strip()
    if not code:
        state["repair_in_progress"] = False
        st.error(bt("未能生成修复后的代码。", "No repaired code was generated."))
        return
    agent.save_code(code)
    attempts = int((result or {}).get("attempts") or 0)
    if (result or {}).get("success"):
        record_validated_code(state, code, attempts=attempts)
        st.success(bt("代码已修复并通过验证，请点击执行可视化。", "Code was repaired and validated. Run visualization to publish the charts."))
    else:
        record_validation_failure(state, code, str((result or {}).get("error") or error_text), attempts=attempts or 5)
        st.error(bt("自动修复已达到5次，未能生成可执行代码。", "Auto-repair reached five attempts without a valid script."))
    st.rerun()


def _revise_visualization_code_draft(agent, revision_instruction: str) -> None:
    from utils.local_workflow_bridge import (
        call_visualizing_code_repair_bridge,
        call_visualizing_validated_code_bridge,
    )

    state = get_suggestion_state(st.session_state, "visualization")
    ctx = st.session_state.get("_viz_phase1_ctx")
    inputs = st.session_state.get("_viz_phase2_inputs")
    current_code = str(agent.load_code() or "").strip()
    if not isinstance(ctx, dict) or not isinstance(inputs, dict) or not current_code:
        st.error(bt("当前可视化代码上下文已失效，请重新生成代码。", "The visualization code context has expired. Generate the code again."))
        return

    repair_prompt = (
        "User requested a code revision. Modify the current code to satisfy this instruction "
        "while preserving the confirmed visualization suggestion:\n"
        f"{revision_instruction}"
    )
    with st.spinner(bt("正在按你的意见修改并验证可视化代码...", "Revising and validating visualization code...")):
        repaired = call_visualizing_code_repair_bridge(ctx, current_code, repair_prompt)
        revised_code = str((repaired or {}).get("code") or "").strip()
        if not revised_code:
            st.error(bt("未能生成修改后的可视化代码。", "No revised visualization code was generated."))
            return
        result = call_visualizing_validated_code_bridge(inputs, ctx, revised_code)

    code = str((result or {}).get("code") or revised_code).strip()
    agent.save_code(code)
    attempts = int((result or {}).get("attempts") or 0)
    if (result or {}).get("success"):
        record_validated_code(state, code, attempts=attempts)
        st.success(bt("代码已按你的意见修改并通过验证，请点击执行可视化。", "The code was revised and validated. Run visualization to publish the charts."))
    else:
        record_validation_failure(
            state,
            code,
            str((result or {}).get("error") or "修改后的代码未通过验证。"),
            attempts=attempts or 5,
        )
        st.error(bt("修改后的可视化代码未通过验证。", "The revised visualization code did not pass validation."))
    st.rerun()


def vis_result(agent) -> None:
    fig_desc_list = agent.load_fig() or []
    if not isinstance(fig_desc_list, list):
        st.warning(
            bt(
                "图表结果格式异常，请重新执行可视化。",
                "The chart result has an invalid format. Run visualization again.",
            )
        )
        return
    total = len(fig_desc_list)
    if total == 0:
        return
    state = get_suggestion_state(st.session_state, "visualization")
    executed_fingerprint = state.get("executed_code_fingerprint")
    current_fingerprint = state.get("current_code_fingerprint")
    if executed_fingerprint and current_fingerprint != executed_fingerprint:
        st.info(
            bt(
                "下方图表来自上一次成功代码；当前草稿尚未执行。",
                "The figures below are from the last successful code; the current draft has not run.",
            )
        )
    summary_3 = st.session_state.get("summary_3")
    if isinstance(summary_3, dict) and summary_3.get("validation_status") == "partial":
        missing = summary_3.get("missing_charts") or []
        st.warning(
            bt(
                f"已展示可用图表；另有 {len(missing)} 张已确认图表未返回。",
                f"Available figures are shown; {len(missing)} confirmed chart(s) were not returned.",
            )
        )
    english_ui = _is_english_ui()

    # Build title list: prefer per-figure bundled title (set during execution),
    # fall back to positional tu_title from session_state.
    raw_tu_titles = _normalize_visualization_titles(st.session_state.get("tu_title"))
    title_items: list[str] = []
    for i, item in enumerate(fig_desc_list):
        bundled = item.get("title", "").strip() if isinstance(item, dict) else ""
        if bundled:
            title_items.append(bundled)
        elif i < len(raw_tu_titles):
            title_items.append(raw_tu_titles[i])
        else:
            title_items.append("")
    show_analysis = bool(
        st.session_state.get(
            "viz_desc_switch_widget",
            st.session_state.get("viz_desc_switch", False),
        )
    )
    current_page_key = "viz_current_page"

    if current_page_key not in st.session_state:
        st.session_state[current_page_key] = 1

    st.session_state[current_page_key] = _safe_visualization_page(
        st.session_state[current_page_key], total
    )

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

    current_page = _safe_visualization_page(selected_page, total)
    if current_page != st.session_state[current_page_key]:
        st.session_state[current_page_key] = current_page

    st.markdown(
        f'<div class="viz-page-indicator">{current_page}-{total}</div>',
        unsafe_allow_html=True,
    )

    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total)
    for idx, item in enumerate(fig_desc_list[start_idx:end_idx], start=start_idx):
        if isinstance(item, dict):
            fig = item.get("base_fig")
            if fig is None:
                fig = item.get("fig")
            desc = item.get("desc", "")
        else:
            fig = item
            desc = ""

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
        figure_key_suffix = _figure_key_suffix(item, display_fig)
        display_title = _clean_visualization_title_text(
            _extract_title_from_figure(display_fig) or _extract_title_from_figure(fig)
        )
        if english_ui and _contains_cjk_text(display_title):
            display_title = _fallback_english_title_from_figure(display_fig)
        if not english_ui and _chinese_title_needs_repair(display_title):
            display_title = _fallback_chinese_title_from_figure(display_fig)
        if not title_items[idx] and display_title:
            title_items[idx] = display_title
            _set_visualization_titles(title_items)

        input_key = f"viz_title_input_{idx}_{figure_key_suffix}"
        default_title = _clean_visualization_title_text(title_items[idx] or display_title)
        if english_ui and _contains_cjk_text(default_title):
            default_title = _fallback_english_title_from_figure(display_fig)
            title_items[idx] = default_title
            _set_visualization_titles(title_items)
        if not english_ui and _chinese_title_needs_repair(default_title):
            default_title = _fallback_chinese_title_from_figure(display_fig)
            title_items[idx] = default_title
            _set_visualization_titles(title_items)

        existing_input = str(st.session_state.get(input_key, "")).strip()
        if (
            input_key not in st.session_state
            or not existing_input
            or (english_ui and _contains_cjk_text(existing_input))
            or (not english_ui and _chinese_title_needs_repair(existing_input))
        ):
            st.session_state[input_key] = default_title
        edited_title = _clean_visualization_title_text(st.session_state.get(input_key, default_title))
        if edited_title != title_items[idx]:
            title_items[idx] = edited_title
            _set_visualization_titles(title_items)
        if edited_title.strip():
            display_fig.update_layout(title=edited_title.strip())

        st.plotly_chart(display_fig, use_container_width=True, key=f"fig_{idx}_{figure_key_suffix}")

        title_columns = st.columns([1.2, 4.6, 1.2])
        with title_columns[1]:
            edited_title = st.text_input(
                bt(f"图表标题 {idx + 1}", f"Chart Title {idx + 1}"),
                key=input_key,
                label_visibility="collapsed",
                placeholder=bt("请输入图表标题", "Enter chart title"),
            )

        edited_title = _clean_visualization_title_text(edited_title)
        if edited_title != title_items[idx]:
            title_items[idx] = edited_title
            _set_visualization_titles(title_items)
            display_fig.update_layout(title=edited_title.strip())

        download_name = _safe_download_name(
            edited_title or f"visualization_{idx + 1}",
            fallback=f"visualization_{idx + 1}",
        )
        download_bytes = _cached_figure_download_bytes(display_fig)
        download_status = "ok" if download_bytes else ""
        download_columns = st.columns([1.2, 1.6, 1.2])
        with download_columns[1]:
            if not download_bytes and st.button(
                bt("生成下载图片", "Prepare Image Download"),
                key=f"viz_prepare_download_{idx}_{figure_key_suffix}",
                use_container_width=True,
            ):
                with st.spinner(bt("正在生成图片...", "Preparing image...")):
                    download_bytes, download_status = _figure_download_bytes(display_fig)

            if download_bytes:
                st.download_button(
                    bt("下载图片", "Download Image"),
                    data=download_bytes,
                    file_name=f"{download_name}.jpg",
                    mime="image/jpeg",
                    key=f"viz_download_{idx}_{figure_key_suffix}",
                    use_container_width=True,
                )
            elif download_status == "timeout":
                st.caption(
                    bt(
                        "图片导出超时，可使用图表工具栏下载。",
                        "Image export timed out; use the chart toolbar to download.",
                    )
                )
            elif download_status:
                st.caption(
                    bt(
                        "图片导出不可用，可使用图表工具栏下载。",
                        "Image export is unavailable; use the chart toolbar to download.",
                    )
                )

        analysis_text = str(desc or "").strip()
        if _is_unhelpful_analysis(analysis_text, english_ui=english_ui):
            analysis_text = _fallback_chart_analysis_from_figure(display_fig)

        if show_analysis and analysis_text:
            st.markdown("---")
            st.write(analysis_text)


def _clear_visualization_workflow_state(agent) -> None:
    invalidate_from(
        st.session_state,
        "visualization",
        include_source=True,
        reason="visualization analysis cleared",
    )
    agent.clear_memory()
    agent.suggestion = None
    agent.code = None
    agent.user_input = None
    agent.error = None
    agent.fig_desc_list = []
    agent.save_fig([])
    agent.finish_auto_task = False
    clear_suggestion_state(st.session_state, "visualization")
    _clear_visualization_title_inputs()
    for key in (
        "viz_workflow_result", "viz_suggestion", "tu_title", "full",
        "abstract_3", "summary_3", "visual_recommendatio", "final_code",
        "viz_desc_switch", "viz_desc_switch_widget",
        "viz_download_image_cache",
        "report_figure_ledger",
        "_viz_phase1_ctx", "_viz_phase2_inputs", "_viz_phase2_pending",
        "_viz_phase2_requested",
    ):
        st.session_state.pop(key, None)


def _reset_visualization_outputs(agent) -> None:
    invalidate_from(
        st.session_state,
        "visualization",
        include_source=True,
        reason="visualization analysis restarted",
    )
    agent.suggestion = None
    agent.code = None
    agent.error = None
    agent.fig_desc_list = []
    agent.save_fig([])
    agent.finish_auto_task = False
    _clear_visualization_title_inputs()
    for key in (
        "viz_workflow_result", "viz_suggestion", "tu_title", "full",
        "abstract_3", "summary_3", "visual_recommendatio", "final_code",
        "viz_desc_switch", "viz_desc_switch_widget", "viz_current_page", "viz_pagination",
        "viz_download_image_cache",
        "report_figure_ledger",
        "_viz_phase1_ctx", "_viz_phase2_inputs", "_viz_phase2_pending",
        "_viz_phase2_requested",
    ):
        st.session_state.pop(key, None)


def _request_visualization_recommendation(
    agent,
    source_data: Any,
    user_input: str,
    *,
    auto: bool,
) -> None:
    state = get_suggestion_state(st.session_state, "visualization")
    _reset_visualization_outputs(agent)

    inputs = _build_visualization_inputs(
        source_data=source_data,
        agent=agent,
        user_input=user_input,
        vis_auto=True,
    )
    if inputs is None:
        st.error(
            bt(
                "无法从当前可用数据构造可视化工作流输入，请检查预处理结果或原始上传数据是否可解析。",
                "Unable to build visualization workflow input from the available data. Check whether the preprocessing result or original uploaded data can be parsed.",
            )
        )
        return

    # ── Phase 1: 快速获取 suggestion 并展示 ──────────────────────
    with st.spinner(bt("正在生成可视化推荐方案...", "Generating visualization recommendations...")):
        phase1_result = _call_visualization_phase1(inputs)

    if not phase1_result:
        return

    visual_recommendatio = phase1_result.get("visual_recommendatio", "")
    phase1_ctx = phase1_result.get("_ctx", {})

    # 先把 suggestion 写入 session_state，让前端立刻可以显示
    st.session_state.visual_recommendatio = visual_recommendatio
    st.session_state.viz_suggestion = visual_recommendatio
    agent.save_suggestion(visual_recommendatio)

    # 缓存 phase1 上下文和 inputs，供 phase2 使用
    st.session_state._viz_phase1_ctx = phase1_ctx
    st.session_state._viz_phase2_inputs = inputs
    replace_active_suggestion(state, visual_recommendatio)
    if auto:
        confirm_active_suggestion(state)
        st.session_state._viz_phase2_pending = True
    st.rerun()


def _revise_visualization_recommendation(agent, revision_instruction: str) -> None:
    from workflows.visualizing import revise_visualizing_phase1

    state = get_suggestion_state(st.session_state, "visualization")
    phase1_ctx = st.session_state.get("_viz_phase1_ctx")
    phase2_inputs = st.session_state.get("_viz_phase2_inputs")
    if not isinstance(phase1_ctx, dict):
        return
    with st.spinner(bt("正在修改可视化建议...", "Revising visualization recommendations...")):
        revised = revise_visualizing_phase1(
            ctx=phase1_ctx,
            original_requirements=base_requirements_text(state),
            revision_instruction=revision_instruction,
        )
    if state.get("confirmed_version") is not None:
        invalidate_from(
            st.session_state,
            "visualization",
            include_source=True,
            reason="confirmed visualization recommendation is being revised",
        )
    suggestion = str(revised.get("visual_recommendatio") or "")
    st.session_state._viz_phase1_ctx = revised.get("_ctx")
    st.session_state._viz_phase2_inputs = phase2_inputs
    st.session_state.visual_recommendatio = suggestion
    st.session_state.viz_suggestion = suggestion
    agent.save_suggestion(suggestion)
    replace_active_suggestion(state, suggestion, revision_instruction=revision_instruction)
    st.rerun()


def _continue_visualization_phase2(agent) -> None:
    """在 suggestion 已展示的前提下，继续执行 phase2（代码生成 + 图表分析）。"""
    state = get_suggestion_state(st.session_state, "visualization")
    if state.get("status") != "confirmed":
        return
    phase1_ctx = st.session_state.get("_viz_phase1_ctx")
    inputs = st.session_state.get("_viz_phase2_inputs")
    st.session_state.pop("_viz_phase2_pending", None)

    if not phase1_ctx or not inputs:
        return

    with st.spinner(
        bt(
            "可视化建议已生成，正在生成代码与图表分析...",
            "Visualization recommendations are ready. Generating code and chart analysis...",
        )
    ):
        workflow_result = _call_visualization_phase2(inputs, phase1_ctx)

    if not workflow_result:
        return

    if workflow_result.get("_status") != "succeeded":
        error_text = str(
            workflow_result.get("_code_error")
            or workflow_result.get("abstract_3")
            or bt("可视化执行失败。", "Visualization execution failed.")
        )
        final_code = str(workflow_result.get("final_code") or "").strip()
        attempts = int(workflow_result.get("_fix_attempts") or 5)
        if final_code:
            agent.save_code(final_code)
            record_validation_failure(state, final_code, error_text, attempts=attempts)
        st.session_state.visualization_failure = error_text
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
        agent.save_error(error_text)
        if st.session_state.get("auto_mode"):
            st.session_state.auto_mode = False
            st.session_state.auto_mode_paused_stage = "visualization"
        st.rerun()

    invalidate_from(
        st.session_state,
        "visualization",
        reason="visualization result replaced",
    )

    tu_title = _normalize_visualization_titles(workflow_result.get("tu_title", ""))
    st.session_state.viz_workflow_result = workflow_result
    st.session_state.tu_title = tu_title
    st.session_state.full = workflow_result.get("full")
    st.session_state.abstract_3 = workflow_result.get("abstract_3")
    st.session_state.summary_3 = workflow_result.get("summary_3")
    st.session_state.final_code = workflow_result.get("final_code", "")
    figure_artifacts = successful_figure_artifacts(workflow_result.get("figure_artifacts"))
    if figure_artifacts:
        st.session_state[VISUALIZATION_FIGURE_ARTIFACTS_KEY] = figure_artifacts
    if st.session_state.final_code:
        agent.save_code(st.session_state.final_code)
    st.session_state.pop("visualization_failure", None)

    agent.add_memory({"role": "assistant", "content": workflow_result})
    record_stage_status(
        st.session_state,
        "visualization",
        "succeeded",
        input_fingerprint=str(
            st.session_state.get("analysis_dataset_fingerprint")
            or current_dataset_fingerprint(st.session_state)
        ),
        output_fingerprint=stable_fingerprint(
            workflow_result.get("summary_3"),
            workflow_result.get("final_code"),
        ),
    )
    agent.finish_auto()
    st.rerun()


def _has_visualization_execution_result(agent) -> bool:
    return bool(agent.load_fig() and stage_is_current(st.session_state, "visualization"))


def _render_visualization_chat_entry(role: str, content: Any, index: int) -> None:
    if isinstance(content, str):
        with st.chat_message(role):
            st.write(content)
        return

    if isinstance(content, dict):
        return

    if isinstance(content, go.Figure):
        with st.chat_message(role):
            st.plotly_chart(content, use_container_width=True, key=f"chart-{index}")


def vis_chat(agent, source_data: Any, auto: bool = False):
    state = get_suggestion_state(st.session_state, "visualization")
    with st.chat_message("assistant"):
        st.write(
            bt(
                "我是 Autostat 数据分析助手。\n\n"
                "你可以在下方输入具体可视化需求，或者直接点击按钮获取可视化推荐。",
                "I am the Autostat data analysis assistant.\n\n"
                "Enter a visualization request below, or get automatic visualization recommendations.",
            )
        )

        if st.session_state.get("visualization_failure"):
            st.error(bt("上一次可视化代码未通过验证。", "The previous visualization code did not pass validation."))
            # This function is rendered inside the outer "Visualization Suggestions"
            # expander. Streamlit does not allow expanders to be nested.
            st.caption(bt("错误详情", "Error details"))
            st.code(str(st.session_state.visualization_failure), language="text")

        columns = st.columns(2)
        with columns[0]:
            analyze_clicked = st.button(
                bt("🔍 可视化推荐", "🔍 Visualization Recommendation"),
                key="viz_suggest",
                use_container_width=True,
                disabled=bool(state.get("active_suggestion")),
            )
        with columns[1]:
            clear_viz_suggest = st.button(
                bt("♻️ 清除可视化分析", "♻️ Clear Visualization Analysis"),
                key="clear_viz_suggest",
                use_container_width=True,
            )

        if clear_viz_suggest:
            _clear_visualization_workflow_state(agent)
            st.rerun()

    for idx, entry in enumerate(visible_messages(state)):
        role = entry.get("role")
        content = entry.get("content")
        _render_visualization_chat_entry(role, content, idx)

    pending_initial_request = take_pending_initial_request(state)
    if pending_initial_request:
        request_text = base_requirements_text(state, pending_initial_request)
        agent.save_user_input(request_text)
        _request_visualization_recommendation(agent, source_data, request_text, auto=False)
        return

    pending_revision = take_pending_revision(state)
    if pending_revision:
        if isinstance(st.session_state.get("_viz_phase1_ctx"), dict):
            _revise_visualization_recommendation(agent, pending_revision)
        else:
            st.warning(bt(
                "上一轮可视化建议上下文已失效，正在基于当前数据和这条消息重新生成建议。",
                "The previous visualization context expired. Regenerating from the current data and this message.",
            ))
            request_text = revision_fallback_text(
                state,
                pending_revision,
                default=bt("请帮我做可视化分析", "Please create a visualization analysis."),
            )
            agent.save_user_input(request_text)
            _request_visualization_recommendation(agent, source_data, request_text, auto=False)
        return

    pending_code_revision = take_pending_code_revision(state)
    if pending_code_revision:
        _revise_visualization_code_draft(agent, pending_code_revision)
        return

    already_generated = bool(state.get("active_suggestion"))

    # ── Phase 2 自动续接：suggestion 已展示，继续生成代码 ──────────
    if st.session_state.get("_viz_phase2_pending"):
        _continue_visualization_phase2(agent)
        return

    if st.session_state.pop("_viz_code_repair_requested", False):
        _repair_visualization_code_draft(agent)
        return

    if st.session_state.pop("_viz_phase2_requested", False):
        if confirm_active_suggestion(state):
            _generate_visualization_code_draft(agent)
        return

    if auto and _has_visualization_execution_result(agent) and not agent.finish_auto_task:
        agent.finish_auto()
        st.rerun()

    if analyze_clicked or (auto and not already_generated):
        user_prompt = agent.load_user_input() or ""
        prompt_text = bt("请帮我做可视化分析", "Please create a visualization analysis.")
        if user_prompt and not state.get("base_requirements"):
            add_requirement(state, user_prompt)
        if not state.get("base_requirements"):
            add_requirement(state, prompt_text)
        request_text = base_requirements_text(state, prompt_text)
        agent.save_user_input(request_text)
        _request_visualization_recommendation(agent, source_data, request_text, auto=auto)
        return

    user_input = st.chat_input(
        bt(
            "请输入可视化要求；建议生成后可继续提出修改意见",
            "Enter visualization requirements; after generation, request revisions here",
        )
    )
    if user_input:
        if state.get("active_suggestion"):
            queue_revision_request(state, user_input)
            st.rerun()
        else:
            queue_initial_request(state, user_input)
            st.rerun()


if __name__ == "__main__":
    st.title(bt("统计可视化分析", "Statistical Visualization Analysis"))
    st.markdown("---")

    preproc_agent = st.session_state.data_preprocess_agent
    load_agent = st.session_state.data_loading_agent
    planner = st.session_state.planner_agent
    auto = bool(st.session_state.auto_mode and planner.vis_auto)

    source_data, source_kind = _resolve_visualization_source(preproc_agent, load_agent)
    df = _source_to_dataframe(source_data)

    if df is None:
        if source_kind == "preprocessing_failed":
            st.error(bt(
                "预处理尚未成功，不能把原始数据作为处理后数据继续可视化。请先修复或明确跳过预处理。",
                "Preprocessing has not succeeded. Raw data cannot silently continue as processed data. Fix or explicitly skip preprocessing first.",
            ))
        else:
            st.warning(bt("请先在数据导入页面加载数据。", "Please load data on the data import page first."))
        st.stop()

    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    agent = st.session_state.visualization_agent
    agent.add_df(df_shuffled)

    if st.session_state.auto_mode:
        if planner.vis_auto and _has_visualization_execution_result(agent):
            next_page = planner.finish_vis_auto()
            if next_page is not None:
                st.switch_page(next_page)
            st.session_state.auto_mode = False
            st.rerun()

    code = agent.load_code()
    code_expand = code is not None

    fig = agent.load_fig()
    fig_expand = bool(fig)

    if source_kind == "raw":
        st.caption(
            bt(
                "当前未对原始数据进行预处理，后续将基于原始数据进行分析",
                "The original data has not been preprocessed. The following analysis will use the raw data.",
            )
        )

    columns = st.columns(2)
    with columns[0].expander(bt("配色选择", "Color Palette"), True):
        vis_palette(agent)
    with columns[1].expander(bt("可视化建议", "Visualization Suggestions"), True):
        vis_chat(agent, source_data, auto)
        vis_code_gen(agent, auto=auto)
    with columns[0].expander(bt("可视化执行", "Visualization Execution"), code_expand):
        vis_execution(agent, auto=auto)
    with columns[0].expander(bt("可视化结果", "Visualization Result"), fig_expand):
        vis_result(agent)
