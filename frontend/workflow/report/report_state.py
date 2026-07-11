"""Session-state helpers for report generation and cached outputs."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from workflow.report.report_constants import (
    REPORT_GENERATION_RUNNING_KEY,
    REPORT_GENERATION_TOKEN_KEY,
    REPORT_PDF_EXPORT_KEY,
    REPORT_PENDING_PREVIEW_KEY,
    REPORT_WORD_EXPORT_KEY,
    REPORT_WORKFLOW_OUTPUT_FIELDS,
)
from workflow.report.report_content_utils import (
    extract_report_html,
    find_first_nested_field,
    stringify_string,
)
from workflow.report.report_export import looks_like_html


def extract_report_workflow_outputs(workflow_result: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}

    for field_name in REPORT_WORKFLOW_OUTPUT_FIELDS:
        value = find_first_nested_field(workflow_result, [field_name])
        if value is not None:
            outputs[field_name] = value

    return outputs


def capture_pending_report_preview(report_agent) -> None:
    existing_preview = st.session_state.get(REPORT_PENDING_PREVIEW_KEY)
    html_content = stringify_string(
        report_agent.load_html()
        or st.session_state.get("report_final_html")
        or ""
    )
    markdown_content = stringify_string(
        report_agent.load_markdown()
        or report_agent.load_report_content()
        or ""
    )

    if not html_content and not markdown_content:
        workflow_result = report_agent.load_report_workflow_result()
        raw_content = extract_report_html(workflow_result) if workflow_result else ""
        raw_content = stringify_string(raw_content).strip()
        if raw_content:
            if looks_like_html(raw_content):
                html_content = raw_content
            else:
                markdown_content = raw_content

    if html_content or markdown_content:
        st.session_state[REPORT_PENDING_PREVIEW_KEY] = {
            "html": html_content,
            "markdown": markdown_content,
        }
    elif isinstance(existing_preview, dict) and (
        existing_preview.get("html") or existing_preview.get("markdown")
    ):
        st.session_state[REPORT_PENDING_PREVIEW_KEY] = existing_preview
    else:
        st.session_state.pop(REPORT_PENDING_PREVIEW_KEY, None)


def clear_generated_report_files(report_agent) -> None:
    report_agent.save_word(None)
    report_agent.save_pdf(None)
    report_agent.save_pdf_export_method(None)
    report_agent.save_html(None)
    report_agent.save_markdown(None)
    st.session_state.pop("report_final_html", None)
    st.session_state.pop(REPORT_WORD_EXPORT_KEY, None)
    st.session_state.pop(REPORT_PDF_EXPORT_KEY, None)


def clear_report_workflow_outputs(report_agent) -> None:
    clear_generated_report_files(report_agent)
    st.session_state.pop(REPORT_PENDING_PREVIEW_KEY, None)
    report_agent.save_report_workflow_result(None)
    report_agent.save_report(None)
    report_agent.save_report_content(None)

    for field_name in REPORT_WORKFLOW_OUTPUT_FIELDS:
        st.session_state.pop(f"report_{field_name}", None)

    st.session_state.pop("report_workflow_outputs", None)
    st.session_state.pop("report_preference_selected", None)


def save_report_workflow_outputs(report_agent, workflow_result: dict[str, Any]) -> None:
    extracted_outputs = extract_report_workflow_outputs(workflow_result)

    report_agent.save_report_workflow_result(workflow_result)
    report_agent.save_report(workflow_result)
    report_agent.save_report_content(None)

    st.session_state.report_workflow_outputs = extracted_outputs
    for field_name in REPORT_WORKFLOW_OUTPUT_FIELDS:
        st.session_state[f"report_{field_name}"] = extracted_outputs.get(field_name)
    st.session_state.report_preference_selected = extracted_outputs.get("preference_selected")


def clear_active_report_outputs(report_agent) -> None:
    clear_generated_report_files(report_agent)
    report_agent.save_report_workflow_result(None)
    report_agent.save_report(None)
    report_agent.save_report_content(None)


def begin_report_generation(report_agent) -> str:
    generation_token = str(time.time_ns())
    st.session_state[REPORT_GENERATION_TOKEN_KEY] = generation_token
    st.session_state[REPORT_GENERATION_RUNNING_KEY] = True
    capture_pending_report_preview(report_agent)
    clear_active_report_outputs(report_agent)
    return generation_token


def is_current_report_generation(generation_token: str | None) -> bool:
    return bool(generation_token) and st.session_state.get(REPORT_GENERATION_TOKEN_KEY) == generation_token


def is_report_generation_cancelled(generation_token: str | None) -> bool:
    return not is_current_report_generation(generation_token)


def finish_report_generation(generation_token: str | None) -> None:
    if is_current_report_generation(generation_token):
        st.session_state[REPORT_GENERATION_RUNNING_KEY] = False


def complete_auto_report(report_agent) -> None:
    report_agent.finish_auto()
    st.session_state.auto_mode = False

    planner = st.session_state.get("planner_agent")
    if planner is not None:
        planner.finish_report_auto()
