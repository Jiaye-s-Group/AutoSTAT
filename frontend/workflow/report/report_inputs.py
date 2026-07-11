"""Build report workflow inputs from Streamlit session state and agents."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from workflow.report.report_constants import FIG_PLACEHOLDER_CAPTURE_PATTERN
from workflow.report.report_content_utils import (
    find_first_nested_field,
    maybe_json_loads,
    normalize_figure_placeholders,
    normalize_toc_list,
    remove_figure_placeholders,
    stringify_string,
)
from workflow.visualization.viz_coding import (
    execute_visualization_code_once,
    generate_visualization_code_once,
)
from workflows.reporting_reference_context import build_stage_reference_contexts


def resolve_loading_field(load_agent, field_name: str, default: Any) -> Any:
    stored_value = st.session_state.get(field_name)
    if stored_value is not None:
        return stored_value

    memory_entries = getattr(load_agent, "load_memory", lambda: [])()
    for entry in reversed(memory_entries):
        content = entry.get("content") if isinstance(entry, dict) else None
        if isinstance(content, dict) and field_name in content:
            return content.get(field_name)

    return default


def extract_toc_text_from_result(workflow_result: dict[str, Any]) -> str:
    return stringify_string(find_first_nested_field(workflow_result, ["toc_text"])).replace("\\r\\n", "\n").replace("\\n", "\n")


def normalize_multiline_text(value: Any) -> str:
    if isinstance(value, str):
        return stringify_string(value).replace("\\r\\n", "\n").replace("\\n", "\n")
    return "\n".join(normalize_toc_list(value))


def normalize_report_format(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            normalized = normalize_report_format(item)
            if normalized:
                return normalized
        return "Word"

    if isinstance(value, dict):
        for key in ("label", "value", "name", "text"):
            if key in value:
                normalized = normalize_report_format(value.get(key))
                if normalized:
                    return normalized
        return "Word"

    text = stringify_string(value).strip().lower()
    if "html" in text:
        return "HTML"
    if "pdf" in text:
        return "PDF"
    if "word" in text or "doc" in text:
        return "Word"
    return "Word"


def normalize_visualization_titles(raw_titles: Any) -> list[str]:
    parsed_titles = maybe_json_loads(raw_titles)

    if parsed_titles is None:
        return []

    if isinstance(parsed_titles, str):
        text = stringify_string(parsed_titles)
        return [line.strip() for line in text.splitlines() if line.strip()]

    if isinstance(parsed_titles, dict):
        for key in ("tu_title", "titles", "data", "items"):
            if key in parsed_titles:
                return normalize_visualization_titles(parsed_titles.get(key))
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


def has_report_prerequisites() -> bool:
    return bool(
        st.session_state.get("summary_1")
        and st.session_state.get("summary_2")
        and st.session_state.get("summary_3")
        and st.session_state.get("summary_4")
    )


def has_usable_visualization_source(source: Any) -> bool:
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


def source_to_visualization_dataframe(source: Any) -> pd.DataFrame | None:
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, np.ndarray):
        return pd.DataFrame(source)

    if isinstance(source, str):
        parsed = maybe_json_loads(source)
        if isinstance(parsed, str):
            return None
        source = parsed

    if isinstance(source, list):
        try:
            return pd.DataFrame(source)
        except Exception:
            return None

    return None


def resolve_visualization_dataframe_for_report(preproc_agent, load_agent) -> pd.DataFrame | None:
    processed_df = preproc_agent.load_processed_df()
    if has_usable_visualization_source(processed_df):
        return source_to_visualization_dataframe(processed_df)

    summary_2 = st.session_state.get("summary_2")
    if isinstance(summary_2, dict):
        summary_processed_df = summary_2.get("processed_df")
        if has_usable_visualization_source(summary_processed_df):
            return source_to_visualization_dataframe(summary_processed_df)

    cached_processed_df = st.session_state.get("prep_result_from_summary_2")
    if has_usable_visualization_source(cached_processed_df):
        return source_to_visualization_dataframe(cached_processed_df)

    raw_df = load_agent.load_df()
    if has_usable_visualization_source(raw_df):
        return source_to_visualization_dataframe(raw_df)

    return None


def has_generated_outline(report_agent) -> bool:
    return bool(normalize_toc_list(report_agent.load_outline()))


def has_generated_word_report(report_agent) -> bool:
    return bool(report_agent.load_report_content() or report_agent.load_html() or report_agent.load_word())


def has_visualization_recommendation(visualization_agent) -> bool:
    if visualization_agent is None:
        return False

    suggestion = (
        st.session_state.get("visual_recommendatio")
        or st.session_state.get("viz_suggestion")
        or visualization_agent.load_suggestion()
    )
    return bool(stringify_string(suggestion))


def ensure_visualization_ready_for_report(visualization_agent) -> bool:
    if visualization_agent is None or not has_visualization_recommendation(visualization_agent):
        st.warning("请先完成可视化推荐部分。")
        return False

    if not visualization_agent.load_code():
        if not generate_visualization_code_once(visualization_agent):
            st.warning("未能自动生成可视化代码，请先前往可视化页面检查推荐结果。")
            return False

    if not visualization_agent.load_fig():
        if not execute_visualization_code_once(visualization_agent):
            st.warning("未能自动生成可视化结果，请先前往可视化页面检查代码或数据。")
            return False

    return True


def loaded_visualization_figure_count() -> int | None:
    visualization_agent = st.session_state.get("visualization_agent")
    if visualization_agent is None:
        return None

    try:
        fig_desc_list = visualization_agent.load_fig() or []
    except Exception:
        return None

    return len(fig_desc_list) if fig_desc_list else None


def has_in_range_figure_refs(content: str, figure_count: int | None) -> bool:
    matches = re.findall(
        FIG_PLACEHOLDER_CAPTURE_PATTERN,
        normalize_figure_placeholders(content or ""),
        flags=re.IGNORECASE,
    )
    if not matches:
        return False

    if figure_count is None:
        return True

    refs = [int(item) for item in matches if str(item).isdigit()]
    if not refs:
        return False

    if 0 in refs:
        return max(refs) < figure_count
    return max(refs) <= figure_count


def visualization_title_items(max_figures: int | None = None) -> list[str]:
    title_items = normalize_visualization_titles(st.session_state.get("tu_title"))

    visualization_agent = st.session_state.get("visualization_agent")
    if visualization_agent is not None:
        try:
            fig_desc_list = visualization_agent.load_fig() or []
        except Exception:
            fig_desc_list = []

        for index, item in enumerate(fig_desc_list):
            if max_figures is not None and index >= max_figures:
                break
            bundled_title = ""
            if isinstance(item, dict):
                bundled_title = stringify_string(item.get("title", "")).strip()
            if not bundled_title:
                continue
            while len(title_items) <= index:
                title_items.append("")
            if not title_items[index]:
                title_items[index] = bundled_title

    if max_figures is not None:
        title_items = title_items[:max_figures]
    return title_items


def selected_full_content_from_fig_analysis(
    visual_summary: Any,
    max_figures: int | None = None,
) -> tuple[str, int]:
    parsed_summary = maybe_json_loads(visual_summary)
    if not isinstance(parsed_summary, dict):
        return "", 0

    fig_analysis = parsed_summary.get("fig_analysis")
    if not isinstance(fig_analysis, list) or not fig_analysis:
        return "", 0

    parts: list[str] = []
    title_items = visualization_title_items(max_figures)
    for index, item in enumerate(fig_analysis):
        if max_figures is not None and index >= max_figures:
            break

        if isinstance(item, dict):
            analysis_text = stringify_string(item.get("analysis") or item.get("desc") or "")
            title_text = stringify_string(item.get("title") or item.get("tu_title") or "")
        else:
            analysis_text = stringify_string(item)
            title_text = ""

        if not title_text and index < len(title_items):
            title_text = title_items[index]

        analysis_text = remove_figure_placeholders(analysis_text)
        analysis_text = re.sub(r"\s+", " ", analysis_text).strip()
        title_text = remove_figure_placeholders(title_text)
        title_text = re.sub(r"\s+", " ", title_text).strip()

        title_line = f"图题：{title_text}" if title_text else ""
        content = "\n".join(part for part in (title_line, analysis_text) if part)
        parts.append(f"[FIG:{index}] {content}".strip())

    return "\n\n".join(parts), len(parts)


def resolve_selected_full_content(
    *,
    visual_summary: Any,
    allow_report_cache: bool,
) -> tuple[str, str]:
    figure_count = loaded_visualization_figure_count()
    fallback_content, fallback_count = selected_full_content_from_fig_analysis(
        visual_summary,
        max_figures=figure_count,
    )
    if fallback_content:
        return fallback_content, f"summary_3.fig_analysis ({fallback_count} items)"

    full_content = stringify_string(st.session_state.get("full"))
    if full_content and has_in_range_figure_refs(full_content, figure_count):
        return normalize_figure_placeholders(full_content), "session_state.full"

    if allow_report_cache:
        cached_content = stringify_string(st.session_state.get("report_selected_full_conten"))
        if cached_content and has_in_range_figure_refs(cached_content, figure_count):
            return normalize_figure_placeholders(cached_content), "report_selected_full_conten"

    if full_content:
        return normalize_figure_placeholders(full_content), "session_state.full"

    if allow_report_cache:
        cached_content = stringify_string(st.session_state.get("report_selected_full_conten"))
        if cached_content:
            return normalize_figure_placeholders(cached_content), "report_selected_full_conten"

    return "", "empty"


def log_selected_full_content(stage: str, content: str, source: str) -> None:
    fig_refs = re.findall(FIG_PLACEHOLDER_CAPTURE_PATTERN, normalize_figure_placeholders(content or ""), flags=re.IGNORECASE)
    print(
        f"[REPORT][INPUT] {stage} selected_full_conten source={source}, "
        f"length={len(content or '')}, fig_refs={fig_refs}"
    )


def build_report_inputs(load_agent, report_agent) -> dict[str, Any]:
    load_summary = maybe_json_loads(resolve_loading_field(load_agent, "summary_1", {}))
    preproc_summary = maybe_json_loads(st.session_state.get("summary_2", {}))
    visual_summary = maybe_json_loads(st.session_state.get("summary_3", {}))
    coding_summary = maybe_json_loads(st.session_state.get("summary_4", {}))

    if not isinstance(load_summary, dict):
        load_summary = {}
    if not isinstance(preproc_summary, dict):
        preproc_summary = {}
    if not isinstance(visual_summary, dict):
        visual_summary = {}
    if not isinstance(coding_summary, dict):
        coding_summary = {}

    selected_full_content, selected_source = resolve_selected_full_content(
        visual_summary=visual_summary,
        allow_report_cache=False,
    )
    log_selected_full_content("toc", selected_full_content, selected_source)

    return {
        "load_summary": load_summary,
        "preproc_summary": preproc_summary,
        "visual_summary": visual_summary,
        "coding_summary": coding_summary,
        "selected_full_conten": selected_full_content,
        "load_abstract": stringify_string(resolve_loading_field(load_agent, "abstract_1", "")),
        "preproc_abstract": stringify_string(st.session_state.get("abstract_2", "")),
        "visual_abstract": stringify_string(st.session_state.get("abstract_3", "")),
        "coding_abstract": stringify_string(st.session_state.get("abstract_4", "")),
        "toc_md": normalize_toc_list(report_agent.load_outline()),
        "outline_length": str(report_agent.load_outline_length() or ""),
        "preference_selected": stringify_string(st.session_state.get("preference_selected")),
        "add_preference": stringify_string(st.session_state.get("add_preference")),
        "report_auto": True,
        "user_input": str(report_agent.load_user_input() or ""),
    }


def build_stage_reference_contexts_for_report(selected_full_content: str = "") -> dict[str, str]:
    load_agent = st.session_state.get("data_loading_agent")
    raw_df = None
    if load_agent is not None:
        try:
            raw_df = load_agent.load_df()
        except Exception:
            raw_df = None

    if isinstance(raw_df, pd.DataFrame):
        plan = {
            "shape_0": raw_df.shape[0],
            "shape_1": raw_df.shape[1],
            "dtype_info_str": raw_df.dtypes.astype(str).to_string(),
            "head_dict_str": json.dumps(raw_df.head(5).to_dict(orient="list"), ensure_ascii=False),
        }
    else:
        plan = {
            "shape_0": "",
            "shape_1": "",
            "dtype_info_str": "",
            "head_dict_str": "",
        }

    loading = {
        "summary_1": maybe_json_loads(st.session_state.get("summary_1", {})),
        "abstract_1": stringify_string(st.session_state.get("abstract_1", "")),
    }
    prep = {
        "summary_2": maybe_json_loads(st.session_state.get("summary_2", {})),
        "abstract_2": stringify_string(st.session_state.get("abstract_2", "")),
    }
    viz = {
        "summary_3": maybe_json_loads(st.session_state.get("summary_3", {})),
        "abstract_3": stringify_string(st.session_state.get("abstract_3", "")),
        "full": selected_full_content,
    }
    model = {
        "summary_4": maybe_json_loads(st.session_state.get("summary_4") or st.session_state.get("modeling_summary_4", {})),
        "abstract_4": stringify_string(st.session_state.get("abstract_4") or st.session_state.get("modeling_abstract_4", "")),
    }

    next_cols: list[str] = []
    next_head = ""
    processed_df = None
    summary_2 = prep.get("summary_2")
    if isinstance(summary_2, dict):
        processed_df = source_to_visualization_dataframe(summary_2.get("processed_df"))
    if processed_df is None and isinstance(raw_df, pd.DataFrame):
        processed_df = raw_df
    if isinstance(processed_df, pd.DataFrame):
        next_cols = list(processed_df.columns.astype(str))
        next_head = json.dumps(processed_df.head(5).to_dict(orient="list"), ensure_ascii=False)

    return build_stage_reference_contexts(
        plan=plan,
        loading=loading,
        prep=prep,
        viz=viz,
        model=model,
        next_cols=next_cols,
        next_head=next_head,
    )


def build_word_report_inputs(report_agent) -> dict[str, Any]:
    current_coding_abstract = stringify_string(
        st.session_state.get("abstract_4") or st.session_state.get("modeling_abstract_4")
    )
    if not current_coding_abstract:
        current_coding_abstract = stringify_string(st.session_state.get("report_coding_abstract"))

    visual_summary = maybe_json_loads(st.session_state.get("summary_3", {}))
    current_selected_full_content, selected_source = resolve_selected_full_content(
        visual_summary=visual_summary if isinstance(visual_summary, dict) else {},
        allow_report_cache=True,
    )
    log_selected_full_content("word", current_selected_full_content, selected_source)

    stage_reference_contexts = build_stage_reference_contexts_for_report(current_selected_full_content)

    return {
        "toc_text": normalize_multiline_text(report_agent.load_outline()),
        "title": "",
        "selected_full_conten": current_selected_full_content,
        "stage_reference_contexts": stage_reference_contexts,
        "preference_selected": stringify_string(st.session_state.get("report_preference_selected")),
        "add_preference": stringify_string(st.session_state.get("report_add_preference")),
        "load_abstract": stringify_string(st.session_state.get("report_load_abstract")),
        "preproc_abstract": stringify_string(st.session_state.get("report_preproc_abstract")),
        "visual_abstract": stringify_string(st.session_state.get("report_visual_abstract")),
        "coding_abstract": current_coding_abstract,
    }


def report_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_report_worker_ref_context(inputs: dict[str, Any]) -> str:
    retriever = st.session_state.get("ref_retriever")
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
        return retriever.retrieve_and_format(
            f"报告撰写 业务背景 {inputs.get('add_preference', '')}",
            top_k=3,
        )
    except Exception as exc:
        print("[REPORT][JOB] reference retrieval failed:", repr(exc))
        return ""


def build_report_worker_payload(report_agent) -> dict[str, Any]:
    inputs = build_word_report_inputs(report_agent)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    inputs.setdefault("preference_select", st.session_state.get("preference_selected") or "")
    inputs["ref_context"] = get_report_worker_ref_context(inputs)

    return {
        "inputs": inputs,
        "llm_config": {
            "api_key": st.session_state.get("llm_api_key") or os.getenv("OPENAI_API_KEY", ""),
            "base_url": st.session_state.get("llm_base_url") or os.getenv("OPENAI_BASE_URL", ""),
            "model": st.session_state.get("llm_model") or os.getenv("OPENAI_MODEL", ""),
        },
    }
