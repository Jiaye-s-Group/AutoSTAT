import json
import re
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

from core.bounded_code_execution import (
    VISUALIZATION_TIMEOUT_SECONDS,
    run_bounded_safe_exec,
)
from core.figure_artifacts import (
    VISUALIZATION_FAILED_FIGURES_KEY,
    VISUALIZATION_FIGURE_ARTIFACTS_KEY,
    normalize_figure_artifact,
    successful_figure_artifacts,
)
from core.plotly_serialization import figure_to_json, json_safe_figure
from core.visualization_contract import validate_visualization_result
from utils.i18n import bt, get_language
from utils.sanitize_code import sanitize_visualization_code
from utils.suggestion_state import (
    begin_code_execution,
    can_auto_repair,
    code_matches_current_suggestion,
    finish_code_execution,
    get_suggestion_state,
    mark_code_draft,
    record_execution_failure,
    record_validated_code,
    record_validation_failure,
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


def _revise_visualization_code_draft_from_execution(
    agent,
    revision_instruction: str,
    *,
    current_code_override: str = "",
) -> None:
    """Revise the visible visualization code from the execution panel."""
    from utils.local_workflow_bridge import (
        call_visualizing_code_repair_bridge,
        call_visualizing_validated_code_bridge,
    )

    state = get_suggestion_state(st.session_state, "visualization")
    ctx = st.session_state.get("_viz_phase1_ctx")
    inputs = st.session_state.get("_viz_phase2_inputs")
    current_code = str(current_code_override or agent.load_code() or "").strip()
    if not isinstance(ctx, dict) or not isinstance(inputs, dict) or not current_code:
        st.error(bt("当前可视化代码上下文已失效，请重新生成代码。", "The visualization code context has expired. Generate the code again."))
        return

    repair_prompt = (
        "User requested a code revision. Modify the current code to satisfy this instruction "
        "while preserving the confirmed visualization suggestion:\n"
        f"{revision_instruction}"
    )
    with st.spinner(bt("正在按你的意见修改并验证可视化代码...", "Revising and validating visualization code...")):
        agent.save_code(current_code)
        repaired = call_visualizing_code_repair_bridge(ctx, current_code, repair_prompt)
        revised_code = str((repaired or {}).get("code") or "").strip()
        if not revised_code:
            st.session_state.viz_code_revision_flash = {
                "level": "error",
                "message": bt("未能生成修改后的可视化代码。", "No revised visualization code was generated."),
            }
            st.rerun()
        result = call_visualizing_validated_code_bridge(inputs, ctx, revised_code)

    code = str((result or {}).get("code") or revised_code).strip()
    agent.save_code(code)
    attempts = int((result or {}).get("attempts") or 0)
    if (result or {}).get("success"):
        record_validated_code(state, code, attempts=attempts)
        st.session_state.viz_code_revision_flash = {
            "level": "success",
            "message": bt(
                "代码已按你的意见修改并通过验证，请点击执行可视化。",
                "The code was revised and validated. Run visualization to publish the charts.",
            ),
        }
    else:
        record_validation_failure(
            state,
            code,
            str((result or {}).get("error") or "修改后的代码未通过验证。"),
            attempts=attempts or 5,
        )
        st.session_state.viz_code_revision_flash = {
            "level": "error",
            "message": bt(
                "修改后的可视化代码未通过验证。",
                "The revised visualization code did not pass validation.",
            ),
        }
    st.rerun()


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


def _stringify_visual_report_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value).strip()


def _clean_visual_report_title_text(value) -> str:
    text = _stringify_visual_report_value(value)
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


def _normalize_visual_report_titles(raw_titles) -> list[str]:
    if raw_titles is None:
        return []
    if isinstance(raw_titles, str):
        text = raw_titles.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return [
                cleaned
                for line in text.splitlines()
                if (cleaned := _clean_visual_report_title_text(line))
            ]
        return _normalize_visual_report_titles(parsed)
    if isinstance(raw_titles, dict):
        for key in ("tu_title", "titles", "data", "items"):
            if key in raw_titles:
                return _normalize_visual_report_titles(raw_titles.get(key))
        return [
            cleaned
            for value in raw_titles.values()
            if (cleaned := _clean_visual_report_title_text(value))
        ]
    if isinstance(raw_titles, list):
        titles: list[str] = []
        for item in raw_titles:
            if isinstance(item, dict):
                candidate = (
                    item.get("tu_title")
                    or item.get("title")
                    or item.get("name")
                    or item.get("label")
                    or item.get("text")
                )
            else:
                candidate = item
            text = _clean_visual_report_title_text(candidate)
            if text:
                titles.append(text)
        return titles
    text = _clean_visual_report_title_text(raw_titles)
    return [text] if text else []


def _is_generic_visual_report_title(title: str) -> bool:
    text = re.sub(r"\s+", " ", str(title or "")).strip()
    if not text:
        return True
    normalized = text.lower().replace("_", " ").replace("-", " ")
    return bool(
        re.fullmatch(r"(图表|图|chart|figure|fig)\s*\d+", normalized, flags=re.IGNORECASE)
        or re.fullmatch(r"(chart|figure|fig)\s+\d+", normalized, flags=re.IGNORECASE)
        or re.fullmatch(r"(chart|figure|fig)\s*\d+\s*", normalized, flags=re.IGNORECASE)
    )


def _extract_visual_report_title_from_figure(raw_figure) -> str:
    fig = _normalize_figure(raw_figure)
    if fig is None:
        return ""
    try:
        title_obj = fig.layout.title
        title_text = getattr(title_obj, "text", "") if title_obj is not None else ""
        return str(title_text or "").strip()
    except Exception:
        return ""


def _clean_fig_dict_key_title(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"^chart[_\-\s]*\d+[_\-\s:：]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^fig(?:ure)?[_\-\s]*\d+[_\-\s:：]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _chart_id_for_figure(index: int, fig_dict_key=None) -> str:
    key = str(fig_dict_key or "").strip().lower()
    key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", key).strip("_")
    if key and not re.fullmatch(r"(fig|figure|chart)_?\d*", key, flags=re.IGNORECASE):
        return key[:80]
    return f"chart_{index + 1:02d}"


def _figure_report_title(item, index: int, fallback_titles: list[str]) -> str:
    candidates: list[str] = []
    if isinstance(item, dict):
        candidates.append(_stringify_visual_report_value(item.get("title")))
        candidates.append(_extract_visual_report_title_from_figure(item.get("fig")))
        candidates.append(_extract_visual_report_title_from_figure(item.get("base_fig")))
        candidates.append(_clean_fig_dict_key_title(item.get("fig_dict_key")))
        candidates.append(_clean_fig_dict_key_title(item.get("chart_id")))
    else:
        candidates.append(_extract_visual_report_title_from_figure(item))
    if index < len(fallback_titles):
        candidates.append(fallback_titles[index])
    for candidate in candidates:
        candidate = _clean_visual_report_title_text(candidate)
        if candidate and not _is_generic_visual_report_title(candidate):
            return candidate
    return bt(f"图表 {index + 1}", f"Figure {index + 1}")


def _figure_report_description(item, index: int) -> str:
    if isinstance(item, dict):
        for key in ("desc", "analysis", "summary"):
            text = _stringify_visual_report_value(item.get(key))
            if text:
                return text
    return bt(
        f"图表 {index + 1} 已成功生成，可用于报告中的可视化分析。",
        f"Figure {index + 1} was generated successfully and is available for the visualization section.",
    )


def _visual_report_abstract_needs_repair(value) -> bool:
    text = _stringify_visual_report_value(value)
    if not text:
        return True
    lower = text.lower()
    return "失败" in text or "failed" in lower or "error" in lower


def _build_visual_report_payload(fig_desc_list, fallback_titles: list[str]) -> dict[str, object]:
    fig_analysis: list[dict[str, str]] = []
    full_parts: list[str] = []
    titles: list[str] = []
    for index, item in enumerate(fig_desc_list):
        title = _figure_report_title(item, index, fallback_titles)
        desc = _figure_report_description(item, index)
        titles.append(title)
        fig_analysis.append(
            {
                "fig": item.get("chart_id") if isinstance(item, dict) and item.get("chart_id") else f"chart_{index + 1:02d}",
                "title": title,
                "analysis": desc,
                "fig_dict_key": item.get("fig_dict_key", "") if isinstance(item, dict) else "",
                "chart_id": item.get("chart_id", "") if isinstance(item, dict) else "",
                "stage": item.get("stage", "visualization") if isinstance(item, dict) else "visualization",
                "section_scope": item.get("section_scope", "visualization") if isinstance(item, dict) else "visualization",
            }
        )
        full_parts.append(f"[FIG:{index}] {bt('图题', 'Title')}：{title}\n{desc}".strip())

    visible_titles = "、".join(title for title in titles[:5] if title)
    if visible_titles:
        abstract = bt(
            f"已生成 {len(fig_desc_list)} 张可视化图表，主要包括：{visible_titles}。",
            f"{len(fig_desc_list)} visualization figure(s) were generated, including: {visible_titles}.",
        )
    else:
        abstract = bt(
            f"已生成 {len(fig_desc_list)} 张可视化图表，可用于报告中的可视化分析。",
            f"{len(fig_desc_list)} visualization figure(s) were generated for the report.",
        )

    return {
        "titles": titles,
        "summary_3": {
            "title": bt("数据可视化", "Data Visualization"),
            "fig_analysis": fig_analysis,
            "generated_figure_count": len(fig_desc_list),
            "source": "executed_figures",
        },
        "full": "\n\n".join(full_parts),
        "abstract_3": abstract,
    }


def _figure_item_fingerprint(item) -> str:
    if not isinstance(item, dict):
        fig = _normalize_figure(item)
        return stable_fingerprint(figure_to_json(fig) if fig is not None else str(item))

    raw_fig = item.get("fig")
    fig = _normalize_figure(raw_fig)
    if fig is None:
        fig = _normalize_figure(item.get("base_fig"))
    try:
        fig_payload = figure_to_json(fig) if fig is not None else _stringify_visual_report_value(raw_fig)
    except Exception:
        fig_payload = _stringify_visual_report_value(raw_fig)
    return stable_fingerprint(
        fig_payload,
        item.get("chart_id"),
        item.get("fig_dict_key"),
        item.get("title"),
        item.get("generation_order"),
    )


def _clear_visualization_render_state_after_success() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith("viz_title_input_"):
            st.session_state.pop(key, None)
    for key in (
        "viz_download_image_cache",
        "report_figure_data_uri_cache",
        "report_figure_ledger",
        "report_final_html",
        "report_selected_full_conten",
        VISUALIZATION_FIGURE_ARTIFACTS_KEY,
        VISUALIZATION_FAILED_FIGURES_KEY,
    ):
        st.session_state.pop(key, None)


def sync_visualization_report_state_from_figures(agent, *, partial_failures=None, code: str | None = None) -> bool:
    """Make manually executed figures visible to report outline/body generation."""
    fig_desc_list = agent.load_fig() if agent is not None else []
    if not isinstance(fig_desc_list, list) or not fig_desc_list:
        return False

    fallback_titles = _normalize_visual_report_titles(st.session_state.get("tu_title"))
    normalized_fig_items: list[dict[str, object]] = []
    for index, item in enumerate(fig_desc_list):
        normalized_item = dict(item) if isinstance(item, dict) else {"fig": item}
        fig_dict_key = normalized_item.get("fig_dict_key") or normalized_item.get("figure") or ""
        chart_id = normalized_item.get("chart_id") or _chart_id_for_figure(index, fig_dict_key)
        normalized_item["chart_id"] = str(chart_id)
        normalized_item["fig_dict_key"] = str(fig_dict_key or chart_id)
        normalized_item["generation_order"] = int(normalized_item.get("generation_order") or index)
        normalized_item["language"] = normalized_item.get("language") or get_language()
        resolved_title = _figure_report_title(normalized_item, index, fallback_titles)
        if resolved_title:
            normalized_item["title"] = resolved_title
        artifact = normalize_figure_artifact(
            normalized_item,
            index,
            title=resolved_title,
            description=_figure_report_description(normalized_item, index),
            language=str(normalized_item.get("language") or get_language()),
        )
        artifact["figure_fingerprint"] = _figure_item_fingerprint(artifact)
        normalized_fig_items.append(artifact)
    if hasattr(agent, "save_fig"):
        agent.save_fig(normalized_fig_items)
    fig_desc_list = normalized_fig_items
    successful_artifacts = successful_figure_artifacts(fig_desc_list)
    st.session_state[VISUALIZATION_FIGURE_ARTIFACTS_KEY] = successful_artifacts
    payload = _build_visual_report_payload(fig_desc_list, fallback_titles)
    generated_summary = dict(payload["summary_3"])
    partial_failures = [item for item in (partial_failures or []) if isinstance(item, dict)]
    st.session_state[VISUALIZATION_FAILED_FIGURES_KEY] = partial_failures
    if partial_failures:
        generated_summary["validation_status"] = "partial"
        generated_summary["warnings"] = partial_failures
        generated_summary["failed_figure_count"] = len(partial_failures)
    else:
        generated_summary["failed_figure_count"] = 0

    existing_summary = st.session_state.get("summary_3")
    if isinstance(existing_summary, dict):
        summary_3 = dict(existing_summary)
        summary_3["fig_analysis"] = generated_summary["fig_analysis"]
        summary_3["title"] = generated_summary["title"]
        summary_3["generated_figure_count"] = generated_summary["generated_figure_count"]
        summary_3["source"] = generated_summary["source"]
        if partial_failures and summary_3.get("validation_status") in (None, "", "failed"):
            summary_3["validation_status"] = "partial"
        if partial_failures:
            summary_3["warnings"] = partial_failures
        else:
            summary_3.pop("warnings", None)
    else:
        summary_3 = generated_summary

    st.session_state["summary_3"] = summary_3
    st.session_state["tu_title"] = payload["titles"]
    st.session_state["full"] = payload["full"]
    st.session_state["abstract_3"] = payload["abstract_3"]
    st.session_state["viz_figure_result_fingerprint"] = stable_fingerprint(
        [item.get("figure_fingerprint") for item in fig_desc_list if isinstance(item, dict)]
    )

    final_code = code if code is not None else (agent.load_code() if agent is not None else "")
    if final_code:
        st.session_state["final_code"] = final_code

    workflow_result = st.session_state.get("viz_workflow_result")
    if not isinstance(workflow_result, dict):
        workflow_result = {}
    workflow_result.update(
        {
            "summary_3": st.session_state.get("summary_3"),
            "abstract_3": st.session_state.get("abstract_3"),
            "full": st.session_state.get("full"),
            "tu_title": st.session_state.get("tu_title"),
            "visual_recommendatio": st.session_state.get("visual_recommendatio")
            or (agent.load_suggestion() if agent is not None else ""),
            "final_code": st.session_state.get("final_code", ""),
            "figure_result_fingerprint": st.session_state.get("viz_figure_result_fingerprint"),
            "_status": "succeeded",
            "_synchronized_from_figures": True,
        }
    )
    st.session_state["viz_workflow_result"] = workflow_result
    record_stage_status(
        st.session_state,
        "visualization",
        "succeeded",
        input_fingerprint=str(
            st.session_state.get("analysis_dataset_fingerprint")
            or current_dataset_fingerprint(st.session_state)
        ),
        output_fingerprint=stable_fingerprint(final_code, len(fig_desc_list), st.session_state.get("full")),
    )
    return True


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

    agent.save_code(code)
    agent.save_fig([])
    try:
        with st.spinner(bt("正在运行可视化脚本...", "Running the visualization script...")):
            execution_result = run_bounded_safe_exec(
                kind="visualization",
                code=code,
                dataframe=df,
                timeout_seconds=VISUALIZATION_TIMEOUT_SECONDS,
            )
        if not execution_result["is_success"]:
            raise RuntimeError(str(execution_result["error"]))
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

    fig_dict = execution_result.get("value")
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

    _clear_visualization_render_state_after_success()
    summary_3 = st.session_state.get("summary_3")
    fig_analysis_items = _summary_3_fig_analysis(summary_3)
    tu_title_all = st.session_state.get("tu_title")
    if isinstance(tu_title_all, list):
        tu_title_list = list(tu_title_all)
    else:
        tu_title_list = []
    with st.spinner(bt("正在处理可视化结果...", "Processing visualization results...")):
        failed_figures: list[dict[str, str]] = []
        for idx, (col_name, fig) in enumerate(stqdm(fig_dict.items())):
            try:
                normalized_fig = _normalize_figure(fig)
                if normalized_fig is None:
                    failed_figures.append(
                        {"figure": str(col_name), "error": "无法识别为 Plotly 图表。"}
                    )
                    continue
                normalized_fig = json_safe_figure(normalized_fig)
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

                # Bundle a title with the figure so it stays aligned even when
                # other figures are skipped. Prefer the actual figure title/key
                # over positional tu_title, because tu_title may belong to a
                # previous execution after the user edits and reruns code.
                fig_title = (
                    _extract_visual_report_title_from_figure(normalized_fig)
                    or _clean_fig_dict_key_title(col_name)
                    or (tu_title_list[idx] if idx < len(tu_title_list) else "")
                )
                chart_id = _chart_id_for_figure(idx, col_name)

                agent.add_fig(
                    normalized_fig,
                    desc,
                    base_fig=figure_to_json(base_fig),
                    title=fig_title,
                    chart_id=chart_id,
                    fig_dict_key=str(col_name),
                    generation_order=idx,
                    language=get_language(),
                )
            except Exception:
                failed_figures.append(
                    {"figure": str(col_name), "error": traceback.format_exc()[-2000:]}
                )
                continue

    return sync_visualization_report_state_from_figures(agent, partial_failures=failed_figures, code=code)


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
        if code_matches_current_suggestion(state):
            st.caption(bt("代码状态：已同步当前建议。", "Code status: synced with the current suggestion."))
        else:
            st.warning(bt(
                "建议已更新，当前代码仍基于旧建议。请重新生成代码，或让 AI 将当前代码迁移到新建议。",
                "The suggestion was updated; the current code is still based on an older suggestion. Regenerate the code or ask AI to migrate it.",
            ))
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

    if code is not None:
        flash = st.session_state.pop("viz_code_revision_flash", None)
        if isinstance(flash, dict):
            message = str(flash.get("message") or "")
            if flash.get("level") == "success":
                st.success(message)
            elif flash.get("level") == "error":
                st.error(message)
            elif message:
                st.info(message)

        with st.form("viz_code_revision_form", clear_on_submit=True):
            revision_text = st.text_area(
                bt("代码修改要求", "Code Change Request"),
                placeholder=bt(
                    "例如：将全局信号图改为分箱汇总，并将计数轴改为对数尺度。",
                    "For example, aggregate the global signal plot into bins and use a log scale for count axes.",
                ),
                height=90,
            )
            revise_clicked = st.form_submit_button(bt("让 AutoSTAT 修改代码", "Ask AutoSTAT to Revise Code"))
        if revise_clicked:
            if revision_text.strip():
                _revise_visualization_code_draft_from_execution(
                    agent,
                    revision_text,
                    current_code_override=sanitize_visualization_code(edited),
                )
            else:
                st.warning(bt("请输入代码修改要求。", "Enter a code change request."))

    if code is not None and (
        st.button(bt("▶️ 执行可视化", "▶️ Run Visualization"))
        or (auto and not agent.load_fig())
    ):
        current_code = sanitize_visualization_code(edited)
        run_id = begin_code_execution(state, current_code)
        agent.save_code(current_code)
        try:
            with st.spinner(bt("正在运行可视化脚本...", "Running the visualization script...")):
                execution_result = run_bounded_safe_exec(
                    kind="visualization",
                    code=current_code,
                    dataframe=df,
                    timeout_seconds=VISUALIZATION_TIMEOUT_SECONDS,
                )
            if not execution_result["is_success"]:
                raise RuntimeError(str(execution_result["error"]))
            fig_dict = execution_result.get("value")
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
            partial_failures = []
            for warning in execution_result.get("warnings") or []:
                if isinstance(warning, dict):
                    partial_failures.append(
                        {
                            "figure": str(warning.get("figure") or warning.get("scope") or "脚本执行"),
                            "error": str(warning.get("error") or ""),
                        }
                    )
                elif str(warning).strip():
                    partial_failures.append({"figure": "脚本执行", "error": str(warning)})
            with st.spinner(bt("正在处理可视化结果...", "Processing visualization results...")):
                for idx, (col_name, fig) in enumerate(stqdm(fig_dict.items())):
                    try:
                        normalized_fig = _normalize_figure(fig)
                        if normalized_fig is None:
                            partial_failures.append(
                                {"figure": str(col_name), "error": "无法识别为 Plotly 图表。"}
                            )
                            continue
                        normalized_fig = json_safe_figure(normalized_fig)
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
                            "base_fig": figure_to_json(base_fig),
                            "desc": desc,
                            "title": (
                                _extract_visual_report_title_from_figure(normalized_fig)
                                or _clean_fig_dict_key_title(col_name)
                                or (tu_title_list[idx] if idx < len(tu_title_list) else "")
                            ),
                            "chart_id": _chart_id_for_figure(idx, col_name),
                            "fig_dict_key": str(col_name),
                            "generation_order": idx,
                            "language": get_language(),
                        })
                    except Exception:
                        partial_failures.append(
                            {"figure": str(col_name), "error": traceback.format_exc()[-2000:]}
                        )
                        continue
            if not new_figures:
                raise ValueError(bt(
                    "可视化代码运行后没有得到可展示图表。",
                    "The visualization code ran but produced no displayable charts.",
                ))
            phase1_ctx = st.session_state.get("_viz_phase1_ctx")
            if isinstance(phase1_ctx, dict):
                from workflows.visualizing import ensure_visualization_contract

                phase1_ctx = ensure_visualization_contract(phase1_ctx)
                st.session_state._viz_phase1_ctx = phase1_ctx
            contract_result = validate_visualization_result(
                figure_keys=list(fig_dict),
                contract=(phase1_ctx.get("visualization_contract") if isinstance(phase1_ctx, dict) else {}),
            )
            for item in contract_result.get("missing_charts") or []:
                partial_failures.append(
                    {
                        "figure": str(item.get("id") or "required chart"),
                        "error": bt(
                            f"未返回已确认图表：{item.get('spec') or ''}",
                            f"The confirmed chart was not returned: {item.get('spec') or ''}",
                        ),
                    }
                )
            state["visualization_contract_status"] = contract_result.get("status")
        except Exception:
            error_text = traceback.format_exc()
            finish_code_execution(state, run_id, success=False)
            agent.save_error(error_text)
            _record_visualization_failure(agent, state, error_text)
        else:
            invalidate_from(st.session_state, "visualization", reason="visualization result replaced")
            _clear_visualization_render_state_after_success()
            agent.save_fig(new_figures)
            state["partial_figure_failures"] = partial_failures
            sync_visualization_report_state_from_figures(
                agent,
                partial_failures=partial_failures,
                code=current_code,
            )
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

    partial_failures = state.get("partial_figure_failures") or []
    if partial_failures:
        st.warning(
            bt(
                f"已保留 {len(agent.load_fig() or [])} 张成功图表；另有 {len(partial_failures)} 项未能生成或处理。",
                f"Kept {len(agent.load_fig() or [])} successful figure(s); {len(partial_failures)} item(s) could not be generated or processed.",
            )
        )
        st.caption(bt("未成功图表详情", "Unsuccessful figure details"))
        st.code(
            "\n".join(
                f"- {item.get('figure', 'unknown')}: {item.get('error', '')}"
                for item in partial_failures
                if isinstance(item, dict)
            ),
            language="text",
        )

    error_text = str(state.get("last_execution_error") or "")
    if error_text:
        st.error(bt("执行失败", "Execution failed"))
        # This function is rendered inside the outer "Visualization Execution"
        # expander. Streamlit does not allow expanders to be nested.
        st.caption(bt("错误详情", "Error details"))
        st.code(error_text, language="text")
        attempts = int(state.get("auto_repair_attempts") or 0)
        if can_auto_repair(state):
            repair_slot = st.empty()
            with repair_slot.container():
                repair_requested = st.button(
                        bt("AutoSTAT 自动修复代码", "AutoSTAT Auto-fix Code"),
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
