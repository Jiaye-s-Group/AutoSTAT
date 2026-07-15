"""
report_render

"""

import json
import os
import shutil
import time
import re
import html
import base64
import hashlib
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
import streamlit_antd_components as sac
from bs4 import BeautifulSoup, NavigableString, Tag
from core.report_language import (
    REPORT_LANGUAGE_EN,
    REPORT_LANGUAGE_ZH,
    is_english_report,
    normalize_report_language,
    report_language_html_lang,
    report_language_instruction,
    report_language_name,
)

PLATFORM_LLM_BASE_URL = "https://api.deepseek.com/v1"
PLATFORM_LLM_MODEL = "deepseek-v4-flash"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_runtime_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass


def _platform_llm_config() -> tuple[str, str, str]:
    _load_runtime_env()
    return (
        os.getenv("OPENAI_API_KEY", "").strip(),
        os.getenv("OPENAI_BASE_URL", PLATFORM_LLM_BASE_URL).strip(),
        os.getenv("OPENAI_MODEL", PLATFORM_LLM_MODEL).strip(),
    )

from workflow.visualization.viz_coding import (
    execute_visualization_code_once,
    generate_visualization_code_once,
)
from workflow.report.report_content_utils import (
    _split_markdown_heading_lines,
    build_docx_from_html,
    build_docx_from_markdown,
    extract_report_html,
    extract_report_markdown,
    extract_report_text,
    extract_report_word_bytes,
    find_first_nested_field,
    html_to_markdown,
    markdown_to_html,
    maybe_json_loads,
    normalize_figure_placeholders,
    normalize_trailing_punctuation_before_figure_placeholder,
    normalize_toc_list,
    remove_figure_placeholders,
    stringify_string,
)
from workflow.report.report_utils import convert_report_to_pdf_bytes
from utils.i18n import bt, get_language, set_language, sync_report_language
from utils.workflow_state import current_dataset_fingerprint, stage_is_current

REPORT_WORKFLOW_OUTPUT_FIELDS = (
    "title",
    "add_preference",
    "preference_selected",
    "selected_full_conten",
    "toc_text",
    "load_abstract",
    "preproc_abstract",
    "visual_abstract",
    "coding_abstract",
    "report_language",
)
FIG_PLACEHOLDER_PATTERN = r"(?<![A-Za-z0-9_])[\[\uFF3B\u3010]?\s*FIG\s*[:\uFF1A]?\s*(?:\d+)\s*[\]\uFF3D\u3011]?(?![A-Za-z0-9_])"
FIG_PLACEHOLDER_CAPTURE_PATTERN = r"(?<![A-Za-z0-9_])[\[\uFF3B\u3010]?\s*FIG\s*[:\uFF1A]?\s*(\d+)\s*[\]\uFF3D\u3011]?(?![A-Za-z0-9_])"
REPORT_EXPORT_IMAGE_SCALE = 0.6
REPORT_EXPORT_IMAGE_PERCENT = f"{REPORT_EXPORT_IMAGE_SCALE * 100:.0f}%"
REPORT_IMAGE_EXPORT_TIMEOUT_SECONDS = 12
REPORT_FIGURE_DATA_URI_CACHE_KEY = "report_figure_data_uri_cache"
REPORT_GENERATION_TOKEN_KEY = "report_generation_token"
REPORT_GENERATION_RUNNING_KEY = "report_generation_running"
REPORT_GENERATION_PROCESS_KEY = "report_generation_process"
REPORT_GENERATION_JOB_KEY = "report_generation_job"
REPORT_PENDING_PREVIEW_KEY = "report_generation_pending_preview"
REPORT_DISPLAY_OUTLINE_KEY = "report_display_outline"
REPORT_DISPLAY_TO_INTERNAL_TOC_MAP_KEY = "report_display_to_internal_toc_map"
REPORT_OUTLINE_USER_EDITED_KEY = "report_outline_user_edited"
REPORT_WORD_EXPORT_KEY = "report_word_export_key"
REPORT_PDF_EXPORT_KEY = "report_pdf_export_key"
REPORT_LANGUAGE_SELECTOR_KEY = "report_language_selector"
REPORT_LANGUAGE_WIDGET_SYNC_KEY = "report_language_widget_synced"
REPORT_OUTLINE_LENGTH_SELECTOR_KEY = "report_outline_length_selector"
REPORT_FORMAT_SELECTOR_KEY = "report_format_selector"
REPORT_LANGUAGE_NOTICE_SPACER_HTML = '<div style="height: 1.35rem;"></div>'
REPORT_GENERATION_OUTLINE_CACHE_KEY = "report_generation_outline_cache"
REPORT_OUTLINE_CONVERSION_FLASH_KEY = "report_outline_conversion_flash"
REPORT_OUTLINE_CONVERSION_FLASH_SECONDS = 3
REPORT_LANGUAGE_LABELS = {
    REPORT_LANGUAGE_ZH: "中文报告",
    REPORT_LANGUAGE_EN: "English Report",
}
REPORT_LANGUAGE_TARGET_LABELS = {
    REPORT_LANGUAGE_ZH: "中文",
    REPORT_LANGUAGE_EN: "English",
}
REPORT_INLINE_TRANSLATION_CACHE_KEY = "report_inline_translation_cache"
REPORT_MODELING_TABLE_EN_LABELS = {
    "方法/模型": "Method/Model",
    "准确率": "Accuracy",
    "精确率": "Precision",
    "召回率": "Recall",
    "F1值": "F1 Score",
    "解释方差": "Explained Variance",
    "轮廓系数": "Silhouette Score",
    "实验设置": "Experiment Setting",
    "特征组合": "Feature Set",
    "数据集": "Dataset",
    "数据划分": "Data Split",
    "训练时间": "Training Time",
    "推理时间": "Inference Time",
    "参数量": "Parameter Count",
    "分类": "classification",
    "回归": "regression",
    "聚类": "clustering",
}


def _resolve_loading_field(load_agent, field_name: str, default: Any) -> Any:
    stored_value = st.session_state.get(field_name)
    if stored_value is not None:
        return stored_value

    memory_entries = getattr(load_agent, "load_memory", lambda: [])()
    for entry in reversed(memory_entries):
        content = entry.get("content") if isinstance(entry, dict) else None
        if isinstance(content, dict) and field_name in content:
            return content.get(field_name)

    return default


def _merge_report_workflow_results(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    merged_result: dict[str, Any] = {}

    for result in results:
        if isinstance(result, dict):
            merged_result.update(result)

    return merged_result or None


def _extract_report_workflow_outputs(workflow_result: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}

    for field_name in REPORT_WORKFLOW_OUTPUT_FIELDS:
        value = find_first_nested_field(workflow_result, [field_name])
        if value is not None:
            outputs[field_name] = value

    return outputs


def _extract_toc_text_from_result(workflow_result: dict[str, Any]) -> str:
    return stringify_string(find_first_nested_field(workflow_result, ["toc_text"])).replace("\\r\\n", "\n").replace("\\n", "\n")


def _normalize_multiline_text(value: Any) -> str:
    if isinstance(value, str):
        return stringify_string(value).replace("\\r\\n", "\n").replace("\\n", "\n")
    return "\n".join(normalize_toc_list(value))


def _normalize_report_format(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            normalized = _normalize_report_format(item)
            if normalized:
                return normalized
        return "Word"

    if isinstance(value, dict):
        for key in ("label", "value", "name", "text"):
            if key in value:
                normalized = _normalize_report_format(value.get(key))
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


def _get_report_generation_language(report_agent) -> str:
    loader = getattr(report_agent, "load_report_language", None)
    language = loader() if callable(loader) else get_language()
    return normalize_report_language(language)


def _save_report_generation_language(report_agent, language: Any) -> str:
    normalized = normalize_report_language(language)
    saver = getattr(report_agent, "save_report_language", None)
    if callable(saver) and normalized != _get_report_generation_language(report_agent):
        saver(normalized)
    return normalized


def _get_report_current_language(report_agent) -> str:
    loader = getattr(report_agent, "load_report_current_language", None)
    language = loader() if callable(loader) else _get_report_generation_language(report_agent)
    return normalize_report_language(language)


def _save_report_current_language(report_agent, language: Any) -> str:
    normalized = normalize_report_language(language)
    saver = getattr(report_agent, "save_report_current_language", None)
    if callable(saver):
        saver(normalized)
    return normalized


def _language_selector_index(language: Any) -> int:
    return 1 if is_english_report(language) else 0


def _language_from_selector_index(index: Any) -> str:
    try:
        selected_index = int(index)
    except (TypeError, ValueError):
        return normalize_report_language(index)
    return REPORT_LANGUAGE_EN if selected_index == 1 else REPORT_LANGUAGE_ZH


def _target_language_label(language: Any) -> str:
    return REPORT_LANGUAGE_TARGET_LABELS[normalize_report_language(language)]


def _target_language_ui_label(language: Any) -> str:
    normalized = normalize_report_language(language)
    if normalized == REPORT_LANGUAGE_EN:
        return "English"
    return bt("中文", "Chinese")


def _report_language_label(language: Any) -> str:
    normalized = normalize_report_language(language)
    if normalized == REPORT_LANGUAGE_EN:
        return "English Report"
    return bt("中文报告", "Chinese Report")


def _set_report_outline_conversion_flash(
    message: str,
    target_language: Any | None = None,
    tone: str = "success",
) -> None:
    st.session_state[REPORT_OUTLINE_CONVERSION_FLASH_KEY] = {
        "message": stringify_string(message).strip(),
        "target_language": normalize_report_language(target_language) if target_language else "",
        "tone": tone if tone in {"success", "info"} else "success",
        "created_at": time.time(),
    }


def _render_report_outline_conversion_flash(target_language: Any | None = None) -> None:
    flash = st.session_state.get(REPORT_OUTLINE_CONVERSION_FLASH_KEY)
    if not isinstance(flash, dict):
        return

    if target_language is not None:
        flash_target_language = normalize_report_language(flash.get("target_language"))
        if flash_target_language != normalize_report_language(target_language):
            return

    st.session_state.pop(REPORT_OUTLINE_CONVERSION_FLASH_KEY, None)
    message = stringify_string(flash.get("message")).strip()
    if not message:
        return

    duration = max(1, int(REPORT_OUTLINE_CONVERSION_FLASH_SECONDS))
    tone = stringify_string(flash.get("tone")).strip()
    palette = {
        "success": {
            "background": "#e8f7ee",
            "color": "#15803d",
        },
        "info": {
            "background": "#eaf4ff",
            "color": "#0f4c81",
        },
    }.get(tone, {
        "background": "#e8f7ee",
        "color": "#15803d",
    })
    st.markdown(
        f"""
<style>
@keyframes report-outline-flash-hide {{
  to {{
    opacity: 0;
    max-height: 0;
    margin-top: 0;
    padding: 0 1.25rem;
    overflow: hidden;
  }}
}}
.report-outline-conversion-flash {{
  background: {palette["background"]};
  color: {palette["color"]};
  border-radius: 0.5rem;
  font-size: 1rem;
  line-height: 1.5;
  margin-top: 0;
  max-height: 120px;
  overflow: hidden;
  padding: 1rem 1.25rem;
  animation: report-outline-flash-hide 0.25s ease {duration}s forwards;
}}
</style>
<div class="report-outline-conversion-flash">{html.escape(message)}</div>
""",
        unsafe_allow_html=True,
    )


def _guess_report_language_from_text(text: str, fallback: Any = REPORT_LANGUAGE_ZH) -> str:
    text = stringify_string(text)
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(re.findall(r"[A-Za-z]{3,}", text))
    if chinese_chars == 0 and english_words:
        return REPORT_LANGUAGE_EN
    if chinese_chars >= max(4, english_words // 2):
        return REPORT_LANGUAGE_ZH
    return normalize_report_language(fallback)


def _get_generation_outline_cache() -> dict[str, str]:
    cache = st.session_state.get(REPORT_GENERATION_OUTLINE_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[REPORT_GENERATION_OUTLINE_CACHE_KEY] = cache
    return cache


def _format_report_startup_status(action: str, report_language: Any) -> str:
    if is_english_report(report_language):
        return f"Starting the {action} English report. Preparing inputs now."
    return f"正在启动{action}报告生成，正在准备输入内容。"


def _format_outline_language_prepare_status(source_language: Any, target_language: Any) -> str:
    source_label = _target_language_label(source_language)
    target_label = _target_language_label(target_language)
    if is_english_report(target_language):
        return (
            f"Preparing the outline for the English report from the {source_label} display. "
            "The outline shown above will stay unchanged."
        )
    return f"正在将{source_label}展示目录临时对齐为{target_label}报告目录，当前目录框内容不会被改动。"


def _outline_text_for_report_language(
    toc_text: str,
    target_language: Any,
    status_placeholder: Any | None = None,
) -> str:
    toc_text = _normalize_multiline_text(toc_text).strip()
    if not toc_text:
        return ""

    target_language = normalize_report_language(target_language)
    source_language = _guess_report_language_from_text(toc_text, fallback=target_language)
    if source_language == target_language:
        return toc_text

    cache_key = json.dumps(
        {
            "source_language": source_language,
            "target_language": target_language,
            "toc_hash": hashlib.sha256(toc_text.encode("utf-8")).hexdigest(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    cache = _get_generation_outline_cache()
    cached = stringify_string(cache.get(cache_key)).strip()
    if cached:
        return cached

    if status_placeholder is not None:
        status_placeholder.info(
            _format_outline_language_prepare_status(source_language, target_language)
        )

    try:
        from workflows.report_translation import translate_report_toc

        converted_toc = translate_report_toc(
            toc_text,
            source_language=source_language,
            target_language=target_language,
        )
    except Exception as exc:
        print(f"[REPORT][LANG] generation outline translation failed: {exc}")
        return toc_text

    converted_toc = _normalize_multiline_text(converted_toc).strip()
    if not converted_toc:
        return toc_text

    cache[cache_key] = converted_toc
    return converted_toc


def _contains_chinese_text(text: Any) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", stringify_string(text)))


def _get_inline_translation_cache() -> dict[str, str]:
    cache = st.session_state.get(REPORT_INLINE_TRANSLATION_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[REPORT_INLINE_TRANSLATION_CACHE_KEY] = cache
    return cache


def _translate_report_inline_text(text: Any, target_language: Any, context_hint: str) -> str:
    original = stringify_string(text).strip()
    if not original:
        return ""

    target_language = normalize_report_language(target_language)
    if not is_english_report(target_language) or not _contains_chinese_text(original):
        return original

    cache_key = json.dumps(
        {"target_language": target_language, "context_hint": context_hint, "text": original},
        ensure_ascii=False,
        sort_keys=True,
    )
    cache = _get_inline_translation_cache()
    cached = cache.get(cache_key)
    if cached:
        return cached

    try:
        from workflows.report_translation import translate_report_text

        translated = translate_report_text(
            original,
            source_language=REPORT_LANGUAGE_ZH,
            target_language=target_language,
            context_hint=context_hint,
        ).strip()
    except Exception as exc:
        print(f"[REPORT][LANG] inline translation failed: {exc}")
        translated = original

    translated = translated or original
    cache[cache_key] = translated
    return translated


def _localize_modeling_table_title(title: Any, report_language: Any) -> str:
    title_text = stringify_string(title).strip()
    if not title_text:
        return ""
    if not is_english_report(report_language):
        return title_text

    known_titles = {
        "不同模型在分类任务上的性能比较": "Performance Comparison of Different Models on the Classification Task",
        "不同模型在回归任务上的性能比较": "Performance Comparison of Different Models on the Regression Task",
        "不同模型在聚类任务上的性能比较": "Performance Comparison of Different Models on the Clustering Task",
        "不同模型在建模任务上的结果比较": "Result Comparison of Different Models on the Modeling Task",
    }
    if title_text in known_titles:
        return known_titles[title_text]

    match = re.fullmatch(r"不同模型在(.+?)预测任务上的性能比较", title_text)
    if match:
        target_text = _translate_report_inline_text(
            match.group(1),
            report_language,
            "Modeling target name used in an English table title.",
        )
        return f"Performance Comparison of Different Models for {target_text} Prediction"

    return _translate_report_inline_text(
        title_text,
        report_language,
        "Model comparison table title in a data analysis report.",
    )


def _localize_modeling_table_cell_text(text: Any, report_language: Any) -> str:
    cell_text = stringify_string(text).strip()
    if not cell_text or not is_english_report(report_language):
        return cell_text

    if cell_text in REPORT_MODELING_TABLE_EN_LABELS:
        return REPORT_MODELING_TABLE_EN_LABELS[cell_text]

    localized = cell_text
    for zh_label, en_label in REPORT_MODELING_TABLE_EN_LABELS.items():
        localized = localized.replace(zh_label, en_label)
    localized = localized.replace("（最优）", " (Best)").replace("(最优)", " (Best)")
    localized = localized.replace("最优", "Best")
    localized = re.sub(r"\s+\)", ")", localized)
    return localized.strip()


def _localize_modeling_table_tag(table_tag: Tag, report_language: Any) -> None:
    if not is_english_report(report_language):
        return
    for cell in table_tag.find_all(["th", "td"]):
        localized_text = _localize_modeling_table_cell_text(cell.get_text(" ", strip=True), report_language)
        if localized_text:
            cell.clear()
            cell.string = localized_text


def _store_report_language_version(
    report_agent,
    language: Any,
    *,
    html_content: str | None = None,
    markdown_content: str | None = None,
) -> None:
    saver = getattr(report_agent, "save_report_language_version", None)
    if not callable(saver):
        return
    normalized = normalize_report_language(language)
    payload = {
        "html": stringify_string(html_content if html_content is not None else report_agent.load_html()),
        "markdown": stringify_string(markdown_content if markdown_content is not None else report_agent.load_markdown()),
        "report_content": stringify_string(markdown_content if markdown_content is not None else report_agent.load_report_content()),
        "updated_at": time.time(),
    }
    saver(normalized, payload)


def _load_report_language_version(report_agent, language: Any) -> dict[str, str] | None:
    loader = getattr(report_agent, "load_report_language_version", None)
    if not callable(loader):
        return None
    payload = loader(normalize_report_language(language))
    if not isinstance(payload, dict):
        return None
    html_content = stringify_string(payload.get("html")).strip()
    markdown_content = stringify_string(payload.get("markdown") or payload.get("report_content")).strip()
    if not html_content and not markdown_content:
        return None
    return {
        "html": html_content,
        "markdown": markdown_content,
        "report_content": stringify_string(payload.get("report_content") or markdown_content).strip(),
    }


def _restore_report_language_version(report_agent, language: Any, version: dict[str, str]) -> None:
    language = normalize_report_language(language)
    html_content = stringify_string(version.get("html")).strip()
    markdown_content = stringify_string(version.get("markdown") or version.get("report_content")).strip()
    if not html_content and markdown_content:
        html_content = markdown_to_html(markdown_content, title="")
    html_content = _set_report_html_language(html_content, language)
    if not markdown_content:
        markdown_content = html_to_markdown(html_content)
    report_agent.save_html(html_content)
    report_agent.save_markdown(markdown_content)
    report_agent.save_report_content(stringify_string(version.get("report_content") or markdown_content))
    _save_report_current_language(report_agent, language)
    _save_report_generation_language(report_agent, language)
    _clear_report_binary_exports(report_agent)
    st.session_state.report_final_html = html_content
    _clear_pending_report_preview()


def _normalize_visualization_titles(raw_titles: Any) -> list[str]:
    parsed_titles = maybe_json_loads(raw_titles)

    if parsed_titles is None:
        return []

    if isinstance(parsed_titles, str):
        text = stringify_string(parsed_titles)
        return [line.strip() for line in text.splitlines() if line.strip()]

    if isinstance(parsed_titles, dict):
        for key in ("tu_title", "titles", "data", "items"):
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


def _clean_report_title_text(raw_title: Any) -> str:
    if raw_title is None:
        return ""

    if isinstance(raw_title, dict):
        for key in ("title", "标题", "题目", "text", "name", "label", "content"):
            cleaned = _clean_report_title_text(raw_title.get(key))
            if cleaned:
                return cleaned
        return ""

    if isinstance(raw_title, list):
        for item in raw_title:
            cleaned = _clean_report_title_text(item)
            if cleaned:
                return cleaned
        return ""

    text = stringify_string(raw_title)
    if not text:
        return ""

    code_block_match = re.match(
        r"^```(?:json|text|markdown)?\s*(.*?)\s*```$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if code_block_match:
        text = code_block_match.group(1).strip()

    parsed_text = maybe_json_loads(text)
    if parsed_text is not text:
        cleaned = _clean_report_title_text(parsed_text)
        if cleaned:
            return cleaned

    json_title_match = re.search(r'"(?:title|标题|题目)"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if json_title_match:
        return json_title_match.group(1).strip()

    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r'^[\s`"\']+', "", text)
    text = re.sub(r'[\s`"\']+$', "", text)
    text = re.sub(r"^[《【「『]+", "", text)
    text = re.sub(r"[》】」』]+$", "", text)
    return text.strip()


def _extract_report_title(workflow_result: Any) -> str:
    title_value = find_first_nested_field(workflow_result, ["title", "标题", "题目"])
    title_text = _clean_report_title_text(title_value)
    if title_text:
        return title_text
    return _clean_report_title_text(st.session_state.get("report_title"))


def _build_figure_caption(
    display_number: int,
    fig_index: int,
    title_items: list[str],
    report_language: Any | None = None,
) -> str:
    title_text = ""
    if 0 <= fig_index < len(title_items):
        title_text = _normalize_figure_title_text(title_items[fig_index])

    if report_language is None:
        report_agent = st.session_state.get("report_agent")
        language = _get_report_current_language(report_agent) if report_agent is not None else REPORT_LANGUAGE_ZH
    else:
        language = normalize_report_language(report_language)

    title_text = _translate_report_inline_text(
        title_text,
        language,
        "Figure caption title in a data analysis report.",
    )
    if is_english_report(language):
        return f"Figure {display_number} {title_text}".strip()
    return f"图{display_number} {title_text}".strip()


def _normalize_figure_title_text(text: str) -> str:
    normalized = html.unescape(str(text or ""))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"^(?:(?:图|Figure|[^\w\s])\s*)?\d+\s*[:：、.．\-]?\s*", "", normalized, flags=re.IGNORECASE)
    return normalized.strip()


def _extract_adjacent_text_sibling(tag: Tag, direction: str) -> Tag | None:
    siblings = tag.previous_siblings if direction == "previous" else tag.next_siblings
    for sibling in siblings:
        if isinstance(sibling, NavigableString):
            if str(sibling).strip():
                return None
            continue
        if not isinstance(sibling, Tag):
            continue
        if sibling.name == "br":
            continue
        if sibling.get_text(" ", strip=True):
            return sibling
    return None


def _looks_like_standalone_figure_title(text: str) -> bool:
    normalized = html.unescape(str(text or ""))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False
    return bool(
        re.match(r"^(?:图|Figure|[^\w\s])\s*\d+\s+", normalized, flags=re.IGNORECASE)
        or re.match(r"^\d+\s+", normalized)
    )


def _remove_duplicate_figure_titles(final_html: str) -> str:
    if not final_html:
        return final_html

    soup = BeautifulSoup(final_html, "html.parser")
    changed = False

    for figure_block in soup.find_all("div", class_="report-figure-block"):
        caption_tag = figure_block.find("div", class_="report-figure-caption")
        if caption_tag is None:
            continue

        caption_text = caption_tag.get_text(" ", strip=True)
        caption_key = _normalize_figure_title_text(caption_text)
        if not caption_key:
            continue

        for direction in ("previous", "next"):
            sibling = _extract_adjacent_text_sibling(figure_block, direction)
            while sibling is not None:
                sibling_text = sibling.get_text(" ", strip=True)
                if _looks_like_standalone_figure_title(sibling_text):
                    next_sibling = _extract_adjacent_text_sibling(sibling, direction)
                    sibling.decompose()
                    changed = True
                    sibling = next_sibling
                    continue
                sibling_key = _normalize_figure_title_text(sibling_text)
                if not sibling_key or sibling_key != caption_key:
                    break
                next_sibling = _extract_adjacent_text_sibling(sibling, direction)
                sibling.decompose()
                changed = True
                sibling = next_sibling

    return str(soup) if changed else final_html

def _looks_like_html(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(
        tag in lowered
        for tag in ("<html", "<body", "<main", "<section", "<div", "<p", "<h1", "<h2", "<img")
    )


def _normalize_visual_figure(raw_figure: Any) -> go.Figure | None:
    if isinstance(raw_figure, go.Figure):
        return go.Figure(raw_figure)

    if isinstance(raw_figure, str):
        try:
            return pio.from_json(raw_figure)
        except Exception:
            return None

    return None


def _localize_plotly_figure_for_report(fig: go.Figure, report_language: Any) -> go.Figure:
    if not is_english_report(report_language):
        return fig

    localized = go.Figure(fig)

    def translate_visible_text(text: Any, context_hint: str) -> str:
        return _translate_report_inline_text(text, report_language, context_hint)

    try:
        title_text = stringify_string(localized.layout.title.text).strip()
        if title_text:
            localized.update_layout(
                title_text=translate_visible_text(title_text, "Plotly chart title in an English report.")
            )
    except Exception:
        pass

    try:
        layout_dict = localized.to_dict().get("layout", {})
        for axis_name, axis_value in layout_dict.items():
            if not re.fullmatch(r"[xy]axis\d*", str(axis_name)):
                continue
            if not isinstance(axis_value, dict):
                continue
            raw_title = axis_value.get("title")
            axis_title = raw_title.get("text") if isinstance(raw_title, dict) else raw_title
            axis_title = stringify_string(axis_title).strip()
            if not axis_title:
                continue
            localized.update_layout(
                **{
                    axis_name: {
                        "title": {
                            "text": translate_visible_text(
                                axis_title,
                                "Plotly axis title in an English report.",
                            )
                        }
                    }
                }
            )
    except Exception:
        pass

    try:
        legend_title = stringify_string(localized.layout.legend.title.text).strip()
        if legend_title:
            localized.update_layout(
                legend_title_text=translate_visible_text(
                    legend_title,
                    "Plotly legend title in an English report.",
                )
            )
    except Exception:
        pass

    try:
        for annotation in localized.layout.annotations or []:
            annotation_text = stringify_string(getattr(annotation, "text", "")).strip()
            if annotation_text:
                annotation.text = translate_visible_text(
                    annotation_text,
                    "Plotly annotation text in an English report.",
                )
    except Exception:
        pass

    try:
        for trace in localized.data:
            trace_name = stringify_string(getattr(trace, "name", "")).strip()
            if trace_name:
                trace.name = translate_visible_text(
                    trace_name,
                    "Plotly legend item name in an English report.",
                )
    except Exception:
        pass

    return localized


def _figure_to_data_uri(fig: go.Figure) -> str | None:
    fig_json = fig.to_json()
    export_script = """
import sys
import plotly.io as pio

json_path, png_path = sys.argv[1], sys.argv[2]
with open(json_path, "r", encoding="utf-8") as f:
    fig_json = f.read()
fig = pio.from_json(fig_json)
image_bytes = pio.to_image(fig, format="png", width=1400, height=900, scale=2)
with open(png_path, "wb") as f:
    f.write(image_bytes)
"""
    try:
        with tempfile.TemporaryDirectory(prefix="report_fig_export_") as temp_dir:
            json_path = f"{temp_dir}/figure.json"
            png_path = f"{temp_dir}/figure.png"
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(fig_json)
            subprocess.run(
                [sys.executable, "-c", export_script, json_path, png_path],
                check=True,
                capture_output=True,
                timeout=REPORT_IMAGE_EXPORT_TIMEOUT_SECONDS,
            )
            with open(png_path, "rb") as f:
                image_bytes = f.read()
        return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
    except subprocess.TimeoutExpired:
        print(
            f"[REPORT][FIG] pio.to_image timeout after {REPORT_IMAGE_EXPORT_TIMEOUT_SECONDS}s; skip this figure"
        )
        return None
    except Exception as exc:
        print("[REPORT][FIG] pio.to_image failed:", repr(exc))
        return None


def _get_report_figure_data_uri_cache() -> dict[str, str]:
    cache = st.session_state.get(REPORT_FIGURE_DATA_URI_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[REPORT_FIGURE_DATA_URI_CACHE_KEY] = cache
    return cache


def _build_figure_cache_key(fig_index: int, fig: go.Figure) -> str:
    try:
        fig_json = fig.to_json()
    except Exception:
        fig_json = str(fig.to_plotly_json())
    digest = hashlib.sha256(fig_json.encode("utf-8", errors="ignore")).hexdigest()
    return f"{fig_index}:{digest}"


def _get_cached_figure_data_uri(
    fig_index: int,
    fig: go.Figure,
    image_uri_cache: dict[str, str],
) -> str | None:
    cache_key = _build_figure_cache_key(fig_index, fig)
    image_uri = image_uri_cache.get(cache_key)
    if image_uri:
        return image_uri

    image_uri = _figure_to_data_uri(fig)
    if not image_uri:
        return None

    image_uri_cache[cache_key] = image_uri
    return image_uri


def _prune_stale_figure_cache_entries(
    fig_desc_list: list[Any],
    fig_indices: list[int],
    image_uri_cache: dict[str, str],
) -> None:
    if not image_uri_cache or not fig_indices:
        return

    for fig_index in fig_indices:
        prefix = f"{fig_index}:"
        stale_keys = [key for key in image_uri_cache if key.startswith(prefix)]

        current_key = ""
        if 0 <= fig_index < len(fig_desc_list):
            fig_item = fig_desc_list[fig_index]
            fig = _normalize_visual_figure(fig_item.get("fig") if isinstance(fig_item, dict) else fig_item)
            if fig is not None:
                current_key = _build_figure_cache_key(fig_index, fig)

        for cache_key in stale_keys:
            if cache_key != current_key:
                image_uri_cache.pop(cache_key, None)


def _normalize_report_figure_layout(final_html: str) -> str:
    if not final_html:
        return final_html

    soup = BeautifulSoup(final_html, "html.parser")
    changed = False

    def is_figure_block(tag: Tag) -> bool:
        return tag.name == "div" and "report-figure-block" in (tag.get("class") or [])

    for paragraph in list(soup.find_all("p")):
        figure_blocks = paragraph.find_all(is_figure_block)
        if not figure_blocks:
            continue

        before_blocks: list[Tag] = []
        after_blocks: list[Tag] = []
        total_text = _paragraph_text_without_figure_blocks(paragraph)

        for figure_block in figure_blocks:
            text_before = _paragraph_text_before_node(paragraph, figure_block)
            target_blocks = (
                before_blocks
                if _should_place_figure_before_paragraph(total_text, len(text_before))
                else after_blocks
            )
            target_blocks.append(figure_block.extract())

        paragraph_text = _clean_paragraph_text(paragraph.get_text(" ", strip=True))
        if paragraph_text:
            paragraph.clear()
            paragraph.string = paragraph_text
            for block in before_blocks:
                paragraph.insert_before(block)
            current_node: Tag = paragraph
            for block in after_blocks:
                current_node.insert_after(block)
                current_node = block
        else:
            ordered_blocks = before_blocks + after_blocks
            if ordered_blocks:
                first_node = ordered_blocks[0]
                paragraph.replace_with(first_node)
                current_node = first_node
                for block in ordered_blocks[1:]:
                    current_node.insert_after(block)
                    current_node = block
            else:
                paragraph.decompose()
        changed = True

    return str(soup) if changed else final_html


def _clean_paragraph_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"\s+([,.;:!?，。；：！？、])", r"\1", text)
    text = re.sub(r"([（(【\[])\s+", r"\1", text)
    text = re.sub(r"\s+([）)】\]])", r"\1", text)
    return text.strip()


def _text_length_for_figure_position(text: str) -> int:
    return len(re.sub(r"\s+", "", str(text or "")))


def _should_place_figure_before_paragraph(
    paragraph_text: str,
    raw_offset: int,
    raw_end: int | None = None,
) -> bool:
    before_len = _text_length_for_figure_position(paragraph_text[:raw_offset])
    after_start = raw_offset if raw_end is None else raw_end
    after_len = _text_length_for_figure_position(paragraph_text[after_start:])
    return before_len <= after_len


def _paragraph_text_without_figure_blocks(paragraph: Tag) -> str:
    text_parts: list[str] = []
    for text_node in paragraph.find_all(string=True):
        if text_node.find_parent("div", class_="report-figure-block") is not None:
            continue
        text_parts.append(str(text_node))
    return "".join(text_parts)


def _paragraph_text_before_node(paragraph: Tag, target: Tag) -> str:
    text_parts: list[str] = []
    for descendant in paragraph.descendants:
        if descendant is target:
            break
        if isinstance(descendant, NavigableString):
            if descendant.find_parent("div", class_="report-figure-block") is not None:
                continue
            text_parts.append(str(descendant))
    return "".join(text_parts)


def _resolve_placeholder_figure_index(raw_index: int, fig_count: int, prefer_one_based: bool) -> int | None:
    if fig_count <= 0:
        return None

    candidate = raw_index - 1 if prefer_one_based else raw_index
    if 0 <= candidate < fig_count:
        return candidate
    return None


def _build_figure_block_tag(
    soup: BeautifulSoup,
    fig_desc_list: list[Any],
    title_items: list[str],
    fig_index: int,
    display_number: int,
    image_uri_cache: dict[str, str],
    report_language: Any,
) -> Tag | None:
    if fig_index < 0 or fig_index >= len(fig_desc_list):
        print(f"[REPORT][FIG] fig index out of range: {fig_index}")
        return None

    fig_item = fig_desc_list[fig_index]
    fig = _normalize_visual_figure(fig_item.get("fig") if isinstance(fig_item, dict) else fig_item)
    if fig is None:
        print(f"[REPORT][FIG] fig at index {fig_index} cannot be normalized")
        return None
    fig = _localize_plotly_figure_for_report(fig, report_language)

    image_uri = _get_cached_figure_data_uri(fig_index, fig, image_uri_cache)
    if not image_uri:
        print(f"[REPORT][FIG] fig at index {fig_index} cannot convert to image")
        return None

    caption_text = _build_figure_caption(display_number, fig_index, title_items, report_language)

    block = soup.new_tag("div")
    block["class"] = ["report-figure-block"]
    block["data-fig-index"] = str(fig_index)
    block["data-report-figure-number"] = str(display_number)

    image = soup.new_tag("img")
    image["src"] = image_uri
    image["alt"] = f"Figure {display_number}"
    image["style"] = (
        f"max-width: {REPORT_EXPORT_IMAGE_PERCENT}; "
        f"width: {REPORT_EXPORT_IMAGE_PERCENT}; "
        "height: auto; border-radius: 8px;"
    )
    block.append(image)

    caption = soup.new_tag("div")
    caption["class"] = ["report-figure-caption"]
    caption.string = caption_text
    block.append(caption)

    return block


def _replace_paragraph_placeholders_at_boundaries(
    soup: BeautifulSoup,
    fig_desc_list: list[Any],
    title_items: list[str],
    prefer_one_based: bool,
    image_uri_cache: dict[str, str],
    inserted_fig_indices: set[int],
    report_language: Any,
) -> int:
    inserted_count = 0

    for paragraph in list(soup.find_all("p")):
        if paragraph.find_parent("div", class_="report-figure-block") is not None:
            continue

        text_nodes = [
            text_node
            for text_node in paragraph.find_all(string=True)
            if text_node.parent is not None
            and text_node.parent.name not in {"script", "style", "noscript"}
            and text_node.find_parent("div", class_="report-figure-block") is None
        ]
        if not text_nodes:
            continue

        paragraph_text = "".join(str(text_node) for text_node in text_nodes)
        if not re.search(FIG_PLACEHOLDER_CAPTURE_PATTERN, paragraph_text, flags=re.IGNORECASE):
            continue

        before_blocks: list[Tag] = []
        after_blocks: list[Tag] = []
        valid_ranges_by_node: list[tuple[NavigableString, list[tuple[int, int]]]] = []
        offset = 0

        for text_node in text_nodes:
            text = str(text_node)
            valid_ranges: list[tuple[int, int]] = []

            for match in re.finditer(FIG_PLACEHOLDER_CAPTURE_PATTERN, text, flags=re.IGNORECASE):
                try:
                    raw_index = int(match.group(1))
                except Exception:
                    continue

                fig_index = _resolve_placeholder_figure_index(raw_index, len(fig_desc_list), prefer_one_based)
                if fig_index is None:
                    continue

                valid_ranges.append((match.start(), match.end()))
                if fig_index in inserted_fig_indices:
                    continue

                figure_block = _build_figure_block_tag(
                    soup=soup,
                    fig_desc_list=fig_desc_list,
                    title_items=title_items,
                    fig_index=fig_index,
                    display_number=0,
                    image_uri_cache=image_uri_cache,
                    report_language=report_language,
                )
                if figure_block is None:
                    continue

                inserted_fig_indices.add(fig_index)

                target_blocks = (
                    before_blocks
                    if _should_place_figure_before_paragraph(
                        paragraph_text,
                        offset + match.start(),
                        offset + match.end(),
                    )
                    else after_blocks
                )
                target_blocks.append(figure_block)
                inserted_count += 1

            if valid_ranges:
                valid_ranges_by_node.append((text_node, valid_ranges))
            offset += len(text)

        if not valid_ranges_by_node:
            continue

        for text_node, ranges in valid_ranges_by_node:
            text = str(text_node)
            pieces: list[str] = []
            last_end = 0
            for start, end in ranges:
                pieces.append(text[last_end:start])
                last_end = end
            pieces.append(text[last_end:])

            replacement_text = "".join(pieces)
            if replacement_text:
                text_node.replace_with(NavigableString(replacement_text))
            else:
                text_node.extract()

        paragraph_text_after_removal = _clean_paragraph_text(paragraph.get_text(" ", strip=True))
        if paragraph_text_after_removal:
            paragraph.clear()
            paragraph.string = paragraph_text_after_removal
            for block in before_blocks:
                paragraph.insert_before(block)
            current_node: Tag = paragraph
            for block in after_blocks:
                current_node.insert_after(block)
                current_node = block
        else:
            ordered_blocks = before_blocks + after_blocks
            if ordered_blocks:
                first_node = ordered_blocks[0]
                paragraph.replace_with(first_node)
                current_node = first_node
                for block in ordered_blocks[1:]:
                    current_node.insert_after(block)
                    current_node = block
            else:
                paragraph.decompose()

    return inserted_count


def _replace_remaining_placeholders_in_soup(
    soup: BeautifulSoup,
    fig_desc_list: list[Any],
    title_items: list[str],
    prefer_one_based: bool,
    image_uri_cache: dict[str, str],
    inserted_fig_indices: set[int],
    report_language: Any,
) -> int:
    inserted_count = _replace_paragraph_placeholders_at_boundaries(
        soup=soup,
        fig_desc_list=fig_desc_list,
        title_items=title_items,
        prefer_one_based=prefer_one_based,
        image_uri_cache=image_uri_cache,
        inserted_fig_indices=inserted_fig_indices,
        report_language=report_language,
    )
    text_nodes = list(soup.find_all(string=True))

    for text_node in text_nodes:
        parent = text_node.parent
        if not isinstance(parent, Tag):
            continue
        if parent.name in {"script", "style", "noscript"}:
            continue
        if parent.find_parent("p") is not None or parent.name == "p":
            continue
        if parent.find_parent("div", class_="report-figure-block") is not None:
            continue

        text = str(text_node)
        matches = list(re.finditer(FIG_PLACEHOLDER_CAPTURE_PATTERN, text, flags=re.IGNORECASE))
        if not matches:
            continue

        replacement_nodes: list[Tag | NavigableString] = []
        last_end = 0

        for match in matches:
            leading_text = text[last_end:match.start()]
            if leading_text:
                replacement_nodes.append(NavigableString(leading_text))

            try:
                raw_index = int(match.group(1))
            except Exception:
                replacement_nodes.append(NavigableString(match.group(0)))
                last_end = match.end()
                continue

            fig_index = _resolve_placeholder_figure_index(raw_index, len(fig_desc_list), prefer_one_based)
            if fig_index is None:
                replacement_nodes.append(NavigableString(match.group(0)))
                last_end = match.end()
                continue

            if fig_index in inserted_fig_indices:
                last_end = match.end()
                continue

            figure_block = _build_figure_block_tag(
                soup=soup,
                fig_desc_list=fig_desc_list,
                title_items=title_items,
                fig_index=fig_index,
                display_number=0,
                image_uri_cache=image_uri_cache,
                report_language=report_language,
            )
            if figure_block is None:
                replacement_nodes.append(NavigableString(match.group(0)))
            else:
                inserted_fig_indices.add(fig_index)
                replacement_nodes.append(figure_block)
                inserted_count += 1

            last_end = match.end()

        trailing_text = text[last_end:]
        if trailing_text:
            replacement_nodes.append(NavigableString(trailing_text))

        if not replacement_nodes:
            continue

        if parent.name == "p" and all(isinstance(child, NavigableString) for child in parent.contents):
            paragraph_nodes: list[Tag] = []
            for node in replacement_nodes:
                if isinstance(node, Tag):
                    paragraph_nodes.append(node)
                    continue

                segment_text = re.sub(r"\s+", " ", str(node)).strip()
                if not segment_text:
                    continue
                paragraph_tag = soup.new_tag("p")
                for attr_name, attr_value in parent.attrs.items():
                    paragraph_tag[attr_name] = attr_value
                paragraph_tag.string = segment_text
                paragraph_nodes.append(paragraph_tag)

            if not paragraph_nodes:
                parent.decompose()
                continue

            first_node = paragraph_nodes[0]
            parent.replace_with(first_node)
            current_node = first_node
            for node in paragraph_nodes[1:]:
                current_node.insert_after(node)
                current_node = node
            continue

        first_node = replacement_nodes[0]
        text_node.replace_with(first_node)
        current_node = first_node
        for node in replacement_nodes[1:]:
            current_node.insert_after(node)
            current_node = node

    return inserted_count


def _renumber_report_figure_blocks(
    final_html: str,
    title_items: list[str],
    report_language: Any,
) -> str:
    if not final_html:
        return final_html

    soup = BeautifulSoup(final_html, "html.parser")

    display_number = 0
    for figure_block in list(soup.find_all("div", class_="report-figure-block")):
        image_tag = figure_block.find("img")
        if image_tag is None or not str(image_tag.get("src") or "").strip():
            figure_block.decompose()
            continue

        display_number += 1
        raw_fig_index = figure_block.get("data-fig-index", "")
        try:
            fig_index = int(raw_fig_index)
        except Exception:
            fig_index = -1

        figure_block["data-report-figure-number"] = str(display_number)
        image_tag["alt"] = f"Figure {display_number}"

        caption_tag = figure_block.find("div", class_="report-figure-caption")
        if caption_tag is None:
            caption_tag = soup.new_tag("div")
            caption_tag["class"] = ["report-figure-caption"]
            figure_block.append(caption_tag)
        caption_tag.string = _build_figure_caption(display_number, fig_index, title_items, report_language)

    return str(soup)


def _inject_visualizations_into_html(final_html: str, report_language: Any | None = None) -> str:
    report_language = normalize_report_language(report_language)
    visualization_agent = st.session_state.get("visualization_agent")
    if visualization_agent is None or not final_html:
        print("[REPORT][FIG] visualization_agent missing or final_html empty")
        return final_html

    fig_desc_list = visualization_agent.load_fig() or []
    title_items = _normalize_visualization_titles(st.session_state.get("tu_title"))

    print("[REPORT][FIG] total figures loaded =", len(fig_desc_list))

    if not fig_desc_list:
        print("[REPORT][FIG] no figures loaded, remove placeholders only")
        return remove_figure_placeholders(final_html)

    final_html = normalize_figure_placeholders(final_html)
    final_html = normalize_trailing_punctuation_before_figure_placeholder(final_html)

    matches = re.findall(FIG_PLACEHOLDER_CAPTURE_PATTERN, final_html, flags=re.IGNORECASE)
    print("[REPORT][FIG] placeholders found in html =", matches)

    match_numbers = [int(item) for item in matches if str(item).isdigit()]
    # [FIG:i] uses zero-based visualization indices.
    prefer_one_based = False
    print("[REPORT][FIG] placeholder numbering mode =", "1-based" if prefer_one_based else "0-based")

    valid_unique_fig_indices: list[int] = []
    seen_fig_indices: set[int] = set()
    for raw_number in match_numbers:
        fig_index = _resolve_placeholder_figure_index(raw_number, len(fig_desc_list), prefer_one_based)
        if fig_index is None or fig_index in seen_fig_indices:
            continue
        seen_fig_indices.add(fig_index)
        valid_unique_fig_indices.append(fig_index)
    print("[REPORT][FIG] valid unique figure refs =", valid_unique_fig_indices)

    image_uri_cache = _get_report_figure_data_uri_cache()
    _prune_stale_figure_cache_entries(fig_desc_list, valid_unique_fig_indices, image_uri_cache)
    inserted_fig_indices: set[int] = set()
    soup = BeautifulSoup(final_html, "html.parser")
    inserted_figure_count = _replace_remaining_placeholders_in_soup(
        soup=soup,
        fig_desc_list=fig_desc_list,
        title_items=title_items,
        prefer_one_based=prefer_one_based,
        image_uri_cache=image_uri_cache,
        inserted_fig_indices=inserted_fig_indices,
        report_language=report_language,
    )
    injected_html = str(soup)
    injected_html = _normalize_report_figure_layout(injected_html)
    injected_html = _remove_duplicate_figure_titles(injected_html)
    injected_html = _renumber_report_figure_blocks(injected_html, title_items, report_language)
    injected_html = re.sub(FIG_PLACEHOLDER_PATTERN, "", injected_html, flags=re.IGNORECASE)
    print("[REPORT][FIG] inserted figure count =", inserted_figure_count)
    return injected_html



def _looks_like_modeling_heading(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or "")).lower()
    if not normalized:
        return False
    keywords = (
        "建模",
        "模型构建",
        "模型建立",
        "模型分析",
        "模型评估",
        "模型训练",
        "建模分析",
        "modelinganalysis",
        "modelanalysis",
        "modelbuilding",
        "modelevaluation",
        "modeltraining",
    )
    return any(keyword in normalized for keyword in keywords)


def _looks_like_chapter4_heading(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or "")).lower()
    if not normalized:
        return False
    chapter4_prefixes = (
        "4",
        "4.",
        "4、",
        "4．",
        "4章",
        "第4章",
        "第四章",
    )
    return any(normalized.startswith(prefix) for prefix in chapter4_prefixes)


def _is_heading_tag(tag: Any) -> bool:
    return isinstance(tag, Tag) and bool(re.match(r"^h[1-6]$", tag.name or ""))


def _normalize_heading_text(text: str) -> str:
    normalized = html.unescape(str(text or "")).lower()
    normalized = normalized.replace("\u3000", " ")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _extract_chapter_number_from_text(text: str) -> int | None:
    normalized = html.unescape(str(text or ""))
    match = re.search(r"\b(\d+)(?:[\.．、]\d+)*\b", normalized)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    chinese_match = re.search(r"\u7b2c\s*([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+)\s*\u7ae0", normalized)
    if not chinese_match:
        return None

    numeral = chinese_match.group(1)
    chinese_numerals = {
        "\u4e00": 1,
        "\u4e8c": 2,
        "\u4e09": 3,
        "\u56db": 4,
        "\u4e94": 5,
        "\u516d": 6,
        "\u4e03": 7,
        "\u516b": 8,
        "\u4e5d": 9,
        "\u5341": 10,
    }
    return chinese_numerals.get(numeral)


def _extract_section_path(text: str) -> tuple[int, ...]:
    normalized = html.unescape(str(text or "")).strip()
    match = re.search(r"\b(\d+(?:[\.．、]\d+)*)\b", normalized)
    if match:
        parts = [part for part in re.split(r"[\.．、]", match.group(1)) if part]
        try:
            return tuple(int(part) for part in parts)
        except ValueError:
            return ()

    chapter_number = _extract_chapter_number_from_text(normalized)
    if chapter_number is not None:
        return (chapter_number,)

    return ()


def _is_top_level_chapter4_heading(tag: Tag) -> bool:
    text = tag.get_text(" ", strip=True)
    path = _extract_section_path(text)
    if path == (4,):
        return True
    normalized = _normalize_heading_text(text)
    return "\u7b2c\u56db\u7ae0" in normalized


def _find_modeling_chapter_heading(soup: BeautifulSoup) -> Tag | None:
    heading_tags = soup.find_all(re.compile(r"^h[1-6]$"))
    for heading in heading_tags:
        if _is_top_level_chapter4_heading(heading):
            return heading

    for heading in heading_tags:
        text = heading.get_text(" ", strip=True)
        if _looks_like_chapter4_heading(text) and _looks_like_modeling_heading(text):
            return heading

    candidate_tags = soup.find_all(["p", "div", "section", "article"])
    for tag in candidate_tags:
        if tag.find(["img", "table", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6", "p"]):
            continue
        text = tag.get_text(" ", strip=True)
        path = _extract_section_path(text)
        if path == (4,) or (_looks_like_chapter4_heading(text) and _looks_like_modeling_heading(text)):
            return tag

    return None


def _iter_named_next_siblings(tag: Tag):
    for sibling in tag.next_siblings:
        if isinstance(sibling, Tag):
            yield sibling


def _score_modeling_table_heading(text: str) -> int:
    normalized = _normalize_heading_text(text)
    if not normalized:
        return 0

    score = 0
    weighted_keywords = (
        ("\u6bd4\u8f83", 6),
        ("\u5bf9\u6bd4", 6),
        ("\u6027\u80fd", 5),
        ("\u8bc4\u4f30", 5),
        ("\u8bc4\u4ef7", 5),
        ("\u7ed3\u679c", 4),
        ("\u5b9e\u9a8c", 4),
        ("\u6700\u4f18", 4),
        ("\u6700\u4f73", 4),
        ("\u9009\u62e9", 3),
        ("comparison", 6),
        ("compare", 5),
        ("performance", 5),
        ("evaluation", 5),
        ("metric", 4),
        ("metrics", 4),
        ("result", 4),
        ("results", 4),
        ("best", 3),
        ("accuracy", 3),
        ("precision", 3),
        ("recall", 3),
        ("auc", 3),
        ("f1", 3),
        ("rmse", 3),
        ("mae", 3),
        ("mse", 3),
        ("r2", 3),
    )
    for keyword, weight in weighted_keywords:
        if keyword in normalized:
            score += weight

    if "\u6a21\u578b" in normalized or "model" in normalized:
        score += 1
    if "\u5206\u6790" in normalized or "analysis" in normalized:
        score += 1

    path = _extract_section_path(text)
    if path and path[0] == 4 and len(path) >= 2:
        score += 2

    return score


def _find_best_modeling_table_section_heading(modeling_heading: Tag) -> Tag | None:
    best_heading: Tag | None = None
    best_score = 0

    for sibling in _iter_named_next_siblings(modeling_heading):
        if _is_heading_tag(sibling):
            sibling_text = sibling.get_text(" ", strip=True)
            sibling_path = _extract_section_path(sibling_text)
            if sibling_path and len(sibling_path) == 1 and sibling_path[0] != 4:
                break

            score = _score_modeling_table_heading(sibling_text)
            if score > best_score:
                best_score = score
                best_heading = sibling

    if best_score < 5:
        return None
    return best_heading


def _find_modeling_table_insert_anchor(target_heading: Tag) -> Tag:
    insert_after: Tag = target_heading
    for sibling in _iter_named_next_siblings(target_heading):
        if _is_heading_tag(sibling):
            break
        if sibling.get_text(" ", strip=True):
            insert_after = sibling
            break
    return insert_after


def _load_modeling_table_payload(report_language: Any = REPORT_LANGUAGE_ZH) -> dict[str, str]:
    summary_4 = maybe_json_loads(
        st.session_state.get("summary_4") or st.session_state.get("modeling_summary_4", {})
    )
    if not isinstance(summary_4, dict):
        return {"title": "", "caption": "", "table_html": "", "table_markdown": ""}

    report_language = normalize_report_language(report_language)
    title = _localize_modeling_table_title(summary_4.get("table_title"), report_language)
    table_html = stringify_string(summary_4.get("table_html"))
    table_markdown = stringify_string(summary_4.get("table_markdown"))
    if is_english_report(report_language):
        caption = f"Table 1 {title}" if title else ""
    else:
        caption = f"表1 {title}" if title else ""
    return {
        "title": title,
        "caption": caption,
        "table_html": table_html,
        "table_markdown": table_markdown,
    }


def _inject_modeling_table_into_html(final_html: str, report_language: Any = REPORT_LANGUAGE_ZH) -> str:
    if not final_html:
        return final_html

    report_language = normalize_report_language(report_language)
    payload = _load_modeling_table_payload(report_language)
    caption_text = payload.get("caption", "")
    table_html = payload.get("table_html", "")
    table_markdown = payload.get("table_markdown", "")
    if not caption_text or (not table_html and not table_markdown):
        return final_html

    soup = BeautifulSoup(final_html, "html.parser")
    if soup.find("div", class_="report-modeling-table-block") is not None:
        return final_html

    modeling_heading = _find_modeling_chapter_heading(soup)
    if modeling_heading is None:
        return final_html

    target_heading = _find_best_modeling_table_section_heading(modeling_heading) or modeling_heading
    insert_after = _find_modeling_table_insert_anchor(target_heading)

    block = soup.new_tag("div")
    block["class"] = ["report-modeling-table-block"]

    title_tag = soup.new_tag("p")
    title_tag["class"] = ["report-modeling-table-title"]
    title_tag.string = caption_text
    block.append(title_tag)

    if not table_html and table_markdown:
        table_fragment = BeautifulSoup(markdown_to_html(table_markdown, title=""), "html.parser")
    else:
        table_fragment = BeautifulSoup(table_html, "html.parser")
    table_tag = table_fragment.find("table")
    if table_tag is None:
        return final_html
    _localize_modeling_table_tag(table_tag, report_language)
    block.append(table_tag)

    insert_after.insert_after(block)
    return str(soup)


def _set_report_html_language(final_html: str, report_language: Any) -> str:
    if not final_html:
        return final_html
    try:
        soup = BeautifulSoup(final_html, "html.parser")
        html_tag = soup.find("html")
        if html_tag is None:
            return final_html
        html_tag["lang"] = report_language_html_lang(report_language)
        return str(soup)
    except Exception:
        return final_html


def _finalize_report_html(
    final_html: str,
    report_title: str,
    report_language: Any | None = None,
) -> str:
    if not final_html:
        return final_html

    if report_language is None:
        report_agent = st.session_state.get("report_agent")
        report_language = _get_report_current_language(report_agent) if report_agent is not None else REPORT_LANGUAGE_ZH

    final_html = _normalize_markdown_headings_in_html(final_html)
    final_html = _inject_report_title_into_html(final_html, report_title)
    final_html = _inject_visualizations_into_html(final_html, report_language)
    final_html = _inject_modeling_table_into_html(final_html, report_language)
    final_html = _inject_report_base_style(final_html)
    final_html = _set_report_html_language(final_html, report_language)
    return final_html


def _inject_report_base_style(final_html: str) -> str:
    if not final_html or "data-report-font-patch" in final_html:
        return final_html

    style_block = """
<style data-report-font-patch>
@page {
  size: A4;
  margin: 0.9in 0.95in;
}
html {
  font-size: 11pt;
}
body, main, article, section, aside, nav, div, p, span, li, table, td, th, figcaption,
h1, h2, h3, h4, h5, h6, a, strong, em, b, i {
  font-family: "Times New Roman", "Microsoft YaHei", serif !important;
}
body, main {
  color: #111827;
  background: #ffffff;
  margin: 0;
  padding: 0;
}
main {
  max-width: 980px;
  margin: 0 auto;
  padding: 12px 20px 28px 20px;
}
main h1 {
  margin: 1.2em 0 0.6em 0 !important;
  font-size: 1.55rem !important;
  line-height: 1.3 !important;
  font-weight: 800 !important;
  color: #111827 !important;
}
main h2 {
  margin: 1.1em 0 0.5em 0 !important;
  font-size: 1.3rem !important;
  line-height: 1.32 !important;
  font-weight: 700 !important;
  color: #111827 !important;
}
main h3,
main h4,
main h5,
main h6 {
  color: #111827 !important;
  line-height: 1.35 !important;
  font-weight: 700 !important;
}
main p,
main li,
main td,
main th {
  font-size: 1rem !important;
  line-height: 1.85 !important;
  color: #111827 !important;
}
main ul, main ol {
  padding-left: 1.25rem !important;
}
.report-figure-block {
  margin: 24px 0 !important;
  text-align: center !important;
  break-inside: avoid !important;
  page-break-inside: avoid !important;
}
.report-figure-block img {
  display: block !important;
  max-width: """ + REPORT_EXPORT_IMAGE_PERCENT + """ !important;
  width: """ + REPORT_EXPORT_IMAGE_PERCENT + """ !important;
  height: auto !important;
  margin: 0 auto !important;
}
.report-figure-caption {
  margin-top: 10px !important;
  text-align: center !important;
  font-size: 0.96rem !important;
  color: #4b5563 !important;
}
.report-modeling-table-block {
  margin: 20px 0 24px 0 !important;
  text-align: center !important;
  break-inside: avoid !important;
  page-break-inside: avoid !important;
}
.report-modeling-table-title {
  margin: 0 0 10px 0 !important;
  text-align: center !important;
  font-weight: 700 !important;
  color: #111827 !important;
}
.report-model-comparison-table {
  width: auto !important;
  max-width: 100% !important;
  border-collapse: collapse !important;
  margin: 0 auto !important;
  table-layout: auto !important;
}
.report-model-comparison-table th,
.report-model-comparison-table td {
  padding: 8px 10px !important;
  text-align: center !important;
  vertical-align: middle !important;
  border-left: none !important;
  border-right: none !important;
  border-top: none !important;
  border-bottom: none !important;
}
.report-model-comparison-table th {
  font-weight: 700 !important;
  background: transparent !important;
  border-top: 2px solid #111827 !important;
  border-bottom: 1px solid #111827 !important;
}
.report-model-comparison-table tbody tr:last-child td {
  border-bottom: 2px solid #111827 !important;
}
@media print {
  body {
    margin: 0 !important;
    padding: 0 !important;
  }
  main {
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
  }
  main p,
  main li,
  main td,
  main th {
    font-size: 11pt !important;
    line-height: 1.5 !important;
  }
  main h1 {
    font-size: 22pt !important;
  }
  main h2 {
    font-size: 18pt !important;
  }
  .report-figure-caption {
    font-size: 9.5pt !important;
  }
}
</style>
"""
    head_match = re.search(r"</head\s*>", final_html, flags=re.IGNORECASE)
    if head_match:
        insert_at = head_match.start()
        return final_html[:insert_at] + style_block + final_html[insert_at:]
    return style_block + final_html

def _normalize_markdown_headings_in_html(final_html: str) -> str:
    if not final_html:
        return final_html

    soup = BeautifulSoup(final_html, "html.parser")
    candidate_tags = soup.find_all(["p", "div", "span", "section", "article", "main"])

    for tag in candidate_tags:
        if not isinstance(tag, Tag):
            continue

        if tag.find(
            [
                "img",
                "table",
                "ul",
                "ol",
                "li",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "p",
                "section",
                "article",
                "main",
            ]
        ):
            continue

        text_content = tag.get_text("\n", strip=True)
        if not text_content or "#" not in text_content:
            continue

        lines = [line.strip() for line in text_content.replace("\r\n", "\n").split("\n") if line.strip()]
        if not lines:
            continue

        replacement_nodes: list[Tag] = []
        for line in lines:
            heading_match = re.match(r"^(#{1,6})[ \t\u3000]*(.*)$", line)
            if heading_match:
                heading_level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                if heading_text:
                    heading_tag = soup.new_tag(f"h{heading_level}")
                    heading_tag.string = heading_text
                    replacement_nodes.append(heading_tag)
                    continue

            parsed_segments = _split_markdown_heading_lines(line)
            if parsed_segments:
                for line_kind, line_text in parsed_segments:
                    if line_kind == "heading":
                        heading_tag = soup.new_tag("h2")
                        heading_tag.string = line_text
                        replacement_nodes.append(heading_tag)
                    else:
                        paragraph_tag = soup.new_tag("p")
                        paragraph_tag.string = line_text
                        replacement_nodes.append(paragraph_tag)
            else:
                paragraph_tag = soup.new_tag("p")
                paragraph_tag.string = line
                replacement_nodes.append(paragraph_tag)

        if not replacement_nodes or not any(re.match(r"^h[1-6]$", node.name or "") for node in replacement_nodes):
            continue

        first_node = replacement_nodes[0]
        tag.replace_with(first_node)
        current_node = first_node
        for node in replacement_nodes[1:]:
            current_node.insert_after(node)
            current_node = node

    return str(soup)


def _inject_report_title_into_html(final_html: str, report_title: str) -> str:
    normalized_title = stringify_string(report_title)
    if not final_html or not normalized_title:
        return final_html

    visible_text = html.unescape(re.sub(r"<[^>]+>", " ", final_html))
    visible_text = re.sub(r"\s+", " ", visible_text).strip()
    if normalized_title in visible_text:
        return final_html

    title_html = (
        "<section class='report-title-block' style='text-align: center; margin: 0 0 28px 0;'>"
        f"<h1 style='margin: 0; font-size: 2.1rem; line-height: 1.3; color: #111827;'>{html.escape(normalized_title)}</h1>"
        "</section>"
    )

    main_match = re.search(r"<main[^>]*>", final_html, flags=re.IGNORECASE)
    if main_match:
        insert_at = main_match.end()
        return final_html[:insert_at] + title_html + final_html[insert_at:]

    return title_html + final_html


def _build_display_export_content(
    html_content: str | None,
    markdown_content: str | None,
) -> tuple[str, str]:
    export_html = _renumber_report_html_for_display(stringify_string(html_content).strip())
    export_markdown = _renumber_report_markdown_for_display(stringify_string(markdown_content).strip())

    if export_html and not export_markdown:
        export_markdown = html_to_markdown(export_html)
    elif export_markdown and not export_html:
        export_html = markdown_to_html(export_markdown, title="")

    return export_html, export_markdown


def _report_export_cache_key(html_content: str, markdown_content: str) -> str:
    display_mapping = _get_display_to_internal_toc_map()
    report_agent = st.session_state.get("report_agent")
    report_language = _get_report_current_language(report_agent) if report_agent is not None else REPORT_LANGUAGE_ZH
    payload = json.dumps(
        {
            "html": html_content or "",
            "markdown": markdown_content or "",
            "display_mapping": display_mapping,
            "report_language": report_language,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prepare_downloadable_reports(report_agent, generation_token: str | None = None) -> dict[str, Any]:
    workflow_result = report_agent.load_report_workflow_result()
    html_content = report_agent.load_html()
    markdown_content = report_agent.load_markdown()
    word_bytes = report_agent.load_word()
    pdf_bytes = report_agent.load_pdf()
    pdf_export_method = report_agent.load_pdf_export_method()
    report_language = _get_report_current_language(report_agent)

    def can_save_prepared_output() -> bool:
        return generation_token is None or _is_current_report_generation(generation_token)

    if workflow_result and not html_content:
        raw_content = extract_report_html(workflow_result)
        if raw_content:
            raw_content = raw_content.strip()
            report_title = _extract_report_title(workflow_result)
            if report_title:
                st.session_state.report_title = report_title

            if _looks_like_html(raw_content):
                html_content = raw_content
                markdown_content = html_to_markdown(html_content)
            else:
                markdown_content = raw_content
                html_content = markdown_to_html(markdown_content, title="")

            # 不再注入额外 report title
            html_content = _finalize_report_html(html_content, report_title, report_language=report_language)
            markdown_content = html_to_markdown(html_content) if html_content else markdown_content

            if can_save_prepared_output():
                report_agent.save_html(html_content)
            if markdown_content and can_save_prepared_output():
                report_agent.save_markdown(markdown_content)
            if can_save_prepared_output():
                st.session_state.report_final_html = html_content

    if not markdown_content:
        if html_content:
            markdown_content = html_to_markdown(html_content)
        elif workflow_result:
            markdown_content = extract_report_markdown(workflow_result) or extract_report_text(workflow_result)

        if markdown_content and can_save_prepared_output():
            report_agent.save_markdown(markdown_content)

    export_html_content, export_markdown_content = _build_display_export_content(
        html_content,
        markdown_content,
    )
    export_cache_key = _report_export_cache_key(export_html_content, export_markdown_content)

    if st.session_state.get(REPORT_WORD_EXPORT_KEY) != export_cache_key:
        word_bytes = None
    if st.session_state.get(REPORT_PDF_EXPORT_KEY) != export_cache_key:
        pdf_bytes = None
        pdf_export_method = None

    def _count_docx_media_files(docx_content: bytes | None) -> int:
        if not docx_content:
            return 0
        try:
            with zipfile.ZipFile(io.BytesIO(docx_content)) as archive:
                return sum(1 for name in archive.namelist() if name.startswith("word/media/"))
        except Exception:
            return 0

    if export_html_content and word_bytes is None:
        try:
            word_bytes = build_docx_from_html(export_html_content)
        except Exception as exc:
            print("[REPORT][WORD] build_docx_from_html failed:", repr(exc))
            word_bytes = None

    html_image_count = len(re.findall(r"<img\b", export_html_content or "", flags=re.IGNORECASE))
    docx_media_count = _count_docx_media_files(word_bytes)
    if word_bytes is not None and html_image_count > 0 and docx_media_count < html_image_count:
        print(
            f"[REPORT][WORD] docx embedded media count mismatch: html_images={html_image_count}, docx_media={docx_media_count}; retry with markdown fallback"
        )
        word_bytes = None

    if word_bytes is None:
        markdown_source = html_to_markdown(export_html_content) if export_html_content else export_markdown_content
        if not markdown_source:
            markdown_source = export_markdown_content
        if markdown_source:
            word_bytes = build_docx_from_markdown(markdown_source)

    if word_bytes is None and workflow_result and not _get_display_to_internal_toc_map():
        word_bytes = extract_report_word_bytes(workflow_result)

    if word_bytes is not None and can_save_prepared_output():
        print(
            f"[REPORT][WORD] final media count = {_count_docx_media_files(word_bytes)}, html_image_count = {html_image_count}"
        )
        report_agent.save_word(word_bytes)
        st.session_state[REPORT_WORD_EXPORT_KEY] = export_cache_key

    if pdf_bytes is not None and can_save_prepared_output():
        report_agent.save_pdf(pdf_bytes)
        report_agent.save_pdf_export_method(pdf_export_method)
        st.session_state[REPORT_PDF_EXPORT_KEY] = export_cache_key

    return {
        "word": word_bytes,
        "html": export_html_content,
        "markdown": export_markdown_content,
        "pdf": pdf_bytes,
        "pdf_export_method": pdf_export_method,
        "export_cache_key": export_cache_key,
    }


def _ensure_pdf_download_ready(report_agent, downloadable_reports: dict[str, Any]) -> dict[str, Any]:
    if downloadable_reports.get("pdf") is not None:
        return downloadable_reports

    word_bytes = downloadable_reports.get("word")
    html_content = downloadable_reports.get("html")
    if not word_bytes and not html_content:
        return downloadable_reports

    try:
        with st.spinner(bt("正在生成 PDF 报告...", "Generating the PDF report...")):
            pdf_bytes, pdf_export_method = convert_report_to_pdf_bytes(
                word_bytes=word_bytes,
                html_content=html_content,
            )
    except Exception as exc:
        print("[REPORT][PDF] convert_report_to_pdf_bytes failed:", repr(exc))
        return downloadable_reports

    downloadable_reports["pdf"] = pdf_bytes
    downloadable_reports["pdf_export_method"] = pdf_export_method
    report_agent.save_pdf(pdf_bytes)
    report_agent.save_pdf_export_method(pdf_export_method)
    export_cache_key = downloadable_reports.get("export_cache_key")
    if export_cache_key:
        st.session_state[REPORT_PDF_EXPORT_KEY] = export_cache_key
    return downloadable_reports


def _build_markdown_preview(markdown_text: str) -> str:
    preview = re.sub(
        r"^\s*!\[[^\]]*\]\((?:data:image/[^)]+|embedded-image)\)\s*$",
        "",
        markdown_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    preview = re.sub(
        r"!\[([^\]]*)\]\(data:image/[^)]+\)",
        lambda match: f"![{match.group(1) or bt('图表', 'Figure')}](embedded-image)",
        preview,
        flags=re.IGNORECASE,
    )
    preview = re.sub(
        r"(?:!\[[^\]]*\]\(embedded-image\)\s*){2,}",
        bt("[图表已嵌入，预览中省略重复图片占位]\n\n", "[Figures are embedded; repeated image placeholders are omitted in the preview.]\n\n"),
        preview,
        flags=re.IGNORECASE,
    )
    preview = re.sub(r"\n{3,}", "\n\n", preview).strip()
    if not preview:
        preview = bt(
            "[正文预览为空，下载的 Markdown 文件中仍包含完整图表内容]",
            "[The body preview is empty; the downloaded Markdown file still includes the full chart content.]",
        )
    if len(preview) > 60000:
        preview = preview[:60000].rstrip() + bt(
            "\n\n...[预览已截断，下载文件中仍保留完整内容]",
            "\n\n...[Preview truncated; the downloaded file keeps the full content.]",
        )
    return preview


def _render_report_preview(html_content: str, markdown_content: str) -> None:
    if (html_content or "").strip():
        components.html(_renumber_report_html_for_display(html_content), height=720, scrolling=True)
        return

    if (markdown_content or "").strip():
        display_markdown = _renumber_report_markdown_for_display(markdown_content)
        st.markdown(_build_markdown_preview(display_markdown))


def _insert_dim_preview_style(html_content: str) -> str:
    style_block = """
<style>
html, body {
  opacity: 0.42 !important;
  filter: grayscale(0.15) !important;
  pointer-events: none !important;
}
</style>
"""
    head_match = re.search(r"</head\s*>", html_content or "", flags=re.IGNORECASE)
    if head_match:
        insert_at = head_match.start()
        return html_content[:insert_at] + style_block + html_content[insert_at:]
    return style_block + (html_content or "")


def _style_pending_report_preview_html(html_content: str, *, live: bool) -> str:
    styled_html = _inject_report_base_style(html_content or "")
    if live:
        return styled_html
    return _insert_dim_preview_style(styled_html)


def _capture_pending_report_preview(report_agent) -> None:
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
            if _looks_like_html(raw_content):
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


def _clear_pending_report_preview() -> None:
    st.session_state.pop(REPORT_PENDING_PREVIEW_KEY, None)


def _render_pending_report_preview() -> None:
    preview = st.session_state.get(REPORT_PENDING_PREVIEW_KEY)
    if not isinstance(preview, dict):
        return

    is_live = bool(preview.get("live"))
    html_content = stringify_string(preview.get("html")).strip()
    markdown_content = stringify_string(preview.get("markdown")).strip()
    if html_content:
        display_html = _renumber_report_html_for_display(html_content)
        components.html(
            _style_pending_report_preview_html(display_html, live=is_live),
            height=720,
            scrolling=True,
        )
        return

    if markdown_content:
        display_markdown = _renumber_report_markdown_for_display(markdown_content)
        preview_html = markdown_to_html(_build_markdown_preview(display_markdown), title="")
        components.html(
            _style_pending_report_preview_html(preview_html, live=is_live),
            height=720,
            scrolling=True,
        )


def _clear_generated_report_files(report_agent) -> None:
    report_agent.save_word(None)
    report_agent.save_pdf(None)
    report_agent.save_pdf_export_method(None)
    report_agent.save_html(None)
    report_agent.save_markdown(None)
    st.session_state.pop("report_final_html", None)
    st.session_state.pop(REPORT_WORD_EXPORT_KEY, None)
    st.session_state.pop(REPORT_PDF_EXPORT_KEY, None)


def _clear_report_binary_exports(report_agent) -> None:
    report_agent.save_word(None)
    report_agent.save_pdf(None)
    report_agent.save_pdf_export_method(None)
    st.session_state.pop(REPORT_WORD_EXPORT_KEY, None)
    st.session_state.pop(REPORT_PDF_EXPORT_KEY, None)


def _clear_report_workflow_outputs(report_agent) -> None:
    _clear_generated_report_files(report_agent)
    _clear_pending_report_preview()
    report_agent.save_report_workflow_result(None)
    report_agent.save_report(None)
    report_agent.save_report_content(None)

    for field_name in REPORT_WORKFLOW_OUTPUT_FIELDS:
        st.session_state.pop(f"report_{field_name}", None)

    st.session_state.pop("report_workflow_outputs", None)
    st.session_state.pop("report_preference_selected", None)
    st.session_state.pop(REPORT_DISPLAY_OUTLINE_KEY, None)
    st.session_state.pop(REPORT_DISPLAY_TO_INTERNAL_TOC_MAP_KEY, None)
    st.session_state.pop(REPORT_OUTLINE_USER_EDITED_KEY, None)


def _save_report_workflow_outputs(report_agent, workflow_result: dict[str, Any]) -> None:
    extracted_outputs = _extract_report_workflow_outputs(workflow_result)

    report_agent.save_report_workflow_result(workflow_result)
    report_agent.save_report(workflow_result)
    report_agent.save_report_content(None)

    st.session_state.report_workflow_outputs = extracted_outputs
    for field_name in REPORT_WORKFLOW_OUTPUT_FIELDS:
        st.session_state[f"report_{field_name}"] = extracted_outputs.get(field_name)
    st.session_state.report_preference_selected = extracted_outputs.get("preference_selected")


def _clear_active_report_outputs(report_agent) -> None:
    _clear_generated_report_files(report_agent)
    report_agent.save_report_workflow_result(None)
    report_agent.save_report(None)
    report_agent.save_report_content(None)


def _begin_report_generation(report_agent) -> str:
    generation_token = str(time.time_ns())
    st.session_state[REPORT_GENERATION_TOKEN_KEY] = generation_token
    st.session_state[REPORT_GENERATION_RUNNING_KEY] = True
    _capture_pending_report_preview(report_agent)
    _clear_active_report_outputs(report_agent)
    _save_report_current_language(report_agent, _get_report_generation_language(report_agent))
    return generation_token


def _is_current_report_generation(generation_token: str | None) -> bool:
    return bool(generation_token) and st.session_state.get(REPORT_GENERATION_TOKEN_KEY) == generation_token


def _is_report_generation_cancelled(generation_token: str | None) -> bool:
    return not _is_current_report_generation(generation_token)


def _finish_report_generation(generation_token: str | None) -> None:
    if _is_current_report_generation(generation_token):
        st.session_state[REPORT_GENERATION_RUNNING_KEY] = False


def _complete_auto_report(report_agent) -> None:
    report_agent.finish_auto()
    st.session_state.auto_mode = False

    planner = st.session_state.get("planner_agent")
    if planner is not None:
        planner.finish_report_auto()


def _auto_stage_was_planned(stage: str) -> bool:
    planner = st.session_state.get("planner_agent")
    if planner is None:
        return True

    stage_was_planned = getattr(planner, "stage_was_planned", None)
    if callable(stage_was_planned):
        return bool(stage_was_planned(stage))

    return bool(getattr(planner, stage, True))


REPORT_STAGE_BY_TOP_LEVEL = {
    "1": "loading_auto",
    "2": "prep_auto",
    "3": "vis_auto",
    "4": "modeling_auto",
}

REPORT_STAGE_CONTENT_KEYS = {
    "loading_auto": (
        "loading_workflow_result",
        "summary_1",
        "abstract_1",
        "report_load_abstract",
    ),
    "prep_auto": (
        "summary_2",
        "abstract_2",
        "report_preproc_abstract",
    ),
    "vis_auto": (
        "summary_3",
        "abstract_3",
        "full",
        "viz_workflow_result",
        "report_visual_abstract",
        "report_selected_full_conten",
    ),
    "modeling_auto": (
        "summary_4",
        "abstract_4",
        "modeling_summary_4",
        "modeling_abstract_4",
        "modeling_workflow_result",
        "report_coding_abstract",
    ),
}


def _has_report_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


def _stage_has_report_content(stage: str) -> bool:
    if st.session_state.get("auto_mode") and not _auto_stage_was_planned(stage):
        return False
    state_stage = {
        "loading_auto": "loading",
        "prep_auto": "preprocessing",
        "vis_auto": "visualization",
        "modeling_auto": "modeling",
    }.get(stage)
    if state_stage and not stage_is_current(st.session_state, state_stage):
        return False
    return any(
        _has_report_value(st.session_state.get(key))
        for key in REPORT_STAGE_CONTENT_KEYS.get(stage, ())
    )


def _report_stage_value(stage: str, value: Any, empty_value: Any) -> Any:
    if not _stage_has_report_content(stage):
        return empty_value
    return value


def _has_any_report_stage_content() -> bool:
    return any(
        _stage_has_report_content(stage)
        for stage in REPORT_STAGE_CONTENT_KEYS
    )


def _parse_toc_line_for_renumber(line: str) -> tuple[str, str] | None:
    match = re.match(r"^(\d+(?:[\.．]\d+)*)(?:[\.．、]|\s+)?\s*(.*)$", str(line or "").strip())
    if not match:
        return None

    old_num = match.group(1).replace("．", ".").strip()
    remainder = match.group(2).strip()
    return old_num, remainder


def _format_renumbered_toc_line(new_num: str, remainder: str) -> str:
    remainder = str(remainder or "").strip()
    if not remainder:
        return new_num
    if "." not in new_num:
        return f"{new_num}.{remainder}"
    return f"{new_num}{remainder}"


def _renumber_toc_text_with_mapping(toc_text: str) -> tuple[str, dict[str, str]]:
    lines = [
        line.strip()
        for line in str(toc_text or "").replace("\\r\\n", "\n").replace("\\n", "\n").splitlines()
        if line.strip()
    ]
    top_level_map: dict[str, str] = {}
    sibling_counters: dict[tuple[str, ...], int] = {}
    next_top_level = 1
    renumbered_lines: list[str] = []
    number_mapping: dict[str, str] = {}

    for line in lines:
        parsed = _parse_toc_line_for_renumber(line)
        if parsed is None:
            renumbered_lines.append(line)
            continue

        old_num, remainder = parsed
        old_parts = [part for part in old_num.split(".") if part]
        if not old_parts:
            renumbered_lines.append(line)
            continue

        old_top = old_parts[0]
        if old_top not in top_level_map:
            top_level_map[old_top] = str(next_top_level)
            next_top_level += 1

        new_parts = [top_level_map[old_top]]
        number_mapping[new_parts[0]] = old_top
        new_prefix_parts = [new_parts[0]]

        for old_part in old_parts[1:]:
            parent_key = tuple(new_prefix_parts)
            sibling_key = parent_key + (old_part,)
            if sibling_key not in sibling_counters:
                sibling_counters[sibling_key] = 1 + sum(
                    1
                    for key in sibling_counters
                    if len(key) == len(sibling_key) and key[:-1] == parent_key
                )
            new_part = str(sibling_counters[sibling_key])
            new_parts.append(new_part)
            new_prefix_parts.append(new_part)

        new_num = ".".join(new_parts)
        old_normalized_num = ".".join(old_parts)
        number_mapping[new_num] = old_normalized_num
        renumbered_lines.append(_format_renumbered_toc_line(new_num, remainder))

    return "\n".join(renumbered_lines), number_mapping


def _renumber_toc_text(toc_text: str) -> str:
    renumbered_text, _ = _renumber_toc_text_with_mapping(toc_text)
    return renumbered_text


def _filter_toc_text_for_available_content(toc_text: str) -> str:
    kept_lines: list[str] = []

    for raw_line in str(toc_text or "").replace("\\r\\n", "\n").replace("\\n", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = re.match(r"^(\d+)(?:[\.．]\d+)*", line)
        if match:
            stage = REPORT_STAGE_BY_TOP_LEVEL.get(match.group(1))
            if stage is not None and not _stage_has_report_content(stage):
                continue

        if not _stage_has_report_content("modeling_auto"):
            line = (
                line.replace("及模型表现", "")
                .replace("模型表现", "分析结果")
                .replace("建模结果", "分析结果")
                .replace("模型结果", "分析结果")
            )

        kept_lines.append(line)

    return "\n".join(kept_lines)


def _save_display_outline_from_internal(internal_toc_text: str) -> str:
    display_toc, number_mapping = _renumber_toc_text_with_mapping(internal_toc_text)
    st.session_state[REPORT_DISPLAY_OUTLINE_KEY] = display_toc
    st.session_state[REPORT_DISPLAY_TO_INTERNAL_TOC_MAP_KEY] = number_mapping
    return display_toc


def _load_display_outline_from_internal(report_agent) -> str:
    internal_toc_text = _normalize_multiline_text(report_agent.load_outline())
    display_toc = stringify_string(st.session_state.get(REPORT_DISPLAY_OUTLINE_KEY)).strip()
    if display_toc:
        return display_toc
    return _save_display_outline_from_internal(internal_toc_text)


def _get_display_toc_title_internal_map(display_toc_text: str | None = None) -> dict[str, str]:
    display_toc = (
        _normalize_multiline_text(display_toc_text)
        if display_toc_text is not None
        else _display_toc_text_for_numbering()
    )
    if not display_toc:
        return {}

    number_mapping = _get_display_to_internal_toc_map()
    title_to_internal: dict[str, str] = {}
    duplicate_titles: set[str] = set()
    for raw_line in display_toc.replace("\\r\\n", "\n").replace("\\n", "\n").splitlines():
        parsed = _parse_toc_line_for_renumber(raw_line)
        if parsed is None:
            continue

        display_num, title_text = parsed
        title_key = _normalize_toc_title_key(title_text)
        internal_num = stringify_string(number_mapping.get(display_num)).strip()
        if not title_key or not internal_num:
            continue

        existing_num = title_to_internal.get(title_key)
        if existing_num and existing_num != internal_num:
            duplicate_titles.add(title_key)
            continue
        title_to_internal[title_key] = internal_num

    for title_key in duplicate_titles:
        title_to_internal.pop(title_key, None)
    return title_to_internal


def _build_display_to_internal_map_from_texts(display_toc_text: str, internal_toc_text: str) -> dict[str, str]:
    display_lines = [
        line.strip()
        for line in str(display_toc_text or "").replace("\\r\\n", "\n").replace("\\n", "\n").splitlines()
        if line.strip()
    ]
    internal_lines = [
        line.strip()
        for line in str(internal_toc_text or "").replace("\\r\\n", "\n").replace("\\n", "\n").splitlines()
        if line.strip()
    ]

    number_mapping: dict[str, str] = {}
    for display_line, internal_line in zip(display_lines, internal_lines):
        display_parsed = _parse_toc_line_for_renumber(display_line)
        internal_parsed = _parse_toc_line_for_renumber(internal_line)
        if display_parsed is None or internal_parsed is None:
            continue
        number_mapping[display_parsed[0]] = internal_parsed[0]
    return number_mapping


def _display_toc_text_to_internal(display_toc_text: str, previous_display_toc: str | None = None) -> str:
    number_mapping = st.session_state.get(REPORT_DISPLAY_TO_INTERNAL_TOC_MAP_KEY)
    if not isinstance(number_mapping, dict):
        number_mapping = {}
    title_internal_mapping = _get_display_toc_title_internal_map(previous_display_toc)

    internal_lines: list[str] = []
    for raw_line in str(display_toc_text or "").replace("\\r\\n", "\n").replace("\\n", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parsed = _parse_toc_line_for_renumber(line)
        if parsed is None:
            internal_lines.append(line)
            continue

        display_num, remainder = parsed
        title_key = _normalize_toc_title_key(remainder)
        internal_num = stringify_string(title_internal_mapping.get(title_key)).strip()
        if not internal_num:
            internal_num = stringify_string(number_mapping.get(display_num)).strip()
        if not internal_num:
            display_parts = [part for part in display_num.split(".") if part]
            mapped_top = stringify_string(number_mapping.get(display_parts[0])).strip() if display_parts else ""
            if mapped_top:
                internal_parts = [mapped_top.split(".")[0], *display_parts[1:]]
                internal_num = ".".join(internal_parts)

        internal_lines.append(
            _format_renumbered_toc_line(internal_num or display_num, remainder)
        )

    return "\n".join(internal_lines)


def _get_display_to_internal_toc_map() -> dict[str, str]:
    number_mapping = st.session_state.get(REPORT_DISPLAY_TO_INTERNAL_TOC_MAP_KEY)
    if isinstance(number_mapping, dict) and number_mapping:
        return {
            stringify_string(display_num).strip(): stringify_string(internal_num).strip()
            for display_num, internal_num in number_mapping.items()
            if stringify_string(display_num).strip() and stringify_string(internal_num).strip()
        }

    report_agent = st.session_state.get("report_agent")
    if report_agent is None:
        return {}

    _, generated_mapping = _renumber_toc_text_with_mapping(
        _normalize_multiline_text(report_agent.load_outline())
    )
    return generated_mapping


def _display_toc_text_for_numbering() -> str:
    display_toc = stringify_string(st.session_state.get(REPORT_DISPLAY_OUTLINE_KEY)).strip()
    if display_toc:
        return _normalize_multiline_text(display_toc)

    report_agent = st.session_state.get("report_agent")
    if report_agent is None:
        return ""

    display_toc, _ = _renumber_toc_text_with_mapping(
        _normalize_multiline_text(report_agent.load_outline())
    )
    return display_toc


def _strip_leading_toc_number(text: str) -> str:
    match = re.match(
        r"^\s*\d+(?:[\.．]\d+)*(?:[\.．、]|\s+)?\s*(.*)$",
        stringify_string(text),
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else stringify_string(text).strip()


def _normalize_toc_title_key(text: str) -> str:
    normalized = _strip_leading_toc_number(text)
    normalized = re.sub(r"[（(][^()（）]*[）)]", "", normalized)
    normalized = re.sub(r"[【\[][^][】\[]*[】\]]", "", normalized)
    normalized = html.unescape(normalized)
    normalized = normalized.lower()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized, flags=re.UNICODE)


def _get_display_toc_title_number_map() -> dict[str, str]:
    display_toc = _display_toc_text_for_numbering()
    if not display_toc:
        return {}

    title_to_num: dict[str, str] = {}
    duplicate_titles: set[str] = set()
    for raw_line in display_toc.replace("\\r\\n", "\n").replace("\\n", "\n").splitlines():
        parsed = _parse_toc_line_for_renumber(raw_line)
        if parsed is None:
            continue

        display_num, title_text = parsed
        title_key = _normalize_toc_title_key(title_text)
        if not title_key:
            continue

        existing_num = title_to_num.get(title_key)
        if existing_num and existing_num != display_num:
            duplicate_titles.add(title_key)
            continue
        title_to_num[title_key] = display_num

    for title_key in duplicate_titles:
        title_to_num.pop(title_key, None)
    return title_to_num


def _map_internal_toc_number_to_display(internal_num: str) -> str:
    internal_num = stringify_string(internal_num).replace("．", ".").strip()
    if not internal_num:
        return ""

    display_to_internal = _get_display_to_internal_toc_map()
    internal_to_display = {
        internal: display
        for display, internal in display_to_internal.items()
        if display and internal
    }
    if internal_num in internal_to_display:
        return internal_to_display[internal_num]

    internal_parts = [part for part in internal_num.split(".") if part]
    if not internal_parts:
        return ""

    display_top = internal_to_display.get(internal_parts[0])
    if not display_top:
        return ""

    display_top_part = display_top.split(".")[0]
    return ".".join([display_top_part, *internal_parts[1:]])


def _replace_leading_internal_toc_number_for_display(text: str) -> str:
    original_text = stringify_string(text)
    match = re.match(
        r"^(\s*)(\d+(?:[\.．]\d+)*)([\.．、]|\s+)?(.*)$",
        original_text,
        flags=re.DOTALL,
    )
    title_to_display_num = _get_display_toc_title_number_map()
    if not match:
        title_key = _normalize_toc_title_key(original_text)
        display_num = title_to_display_num.get(title_key)
        return f"{display_num} {original_text.strip()}" if display_num else original_text

    display_num = _map_internal_toc_number_to_display(match.group(2))
    title_key = _normalize_toc_title_key(match.group(4))
    expected_display_num = title_to_display_num.get(title_key)
    if expected_display_num:
        display_num = expected_display_num

    if not display_num or display_num == match.group(2).replace("．", "."):
        return original_text

    separator = match.group(3)
    if separator is None:
        separator = "." if "." not in display_num else ""

    internal_num = match.group(2).replace("．", ".").strip()
    remainder = match.group(4)
    duplicate_internal_num = re.match(
        rf"^\s*{re.escape(internal_num)}(?:[\.．、]|\s+)?\s*(.*)$",
        remainder,
        flags=re.DOTALL,
    )
    if duplicate_internal_num:
        remainder = duplicate_internal_num.group(1)

    return f"{match.group(1)}{display_num}{separator}{remainder}"


def _renumber_report_html_for_display(html_content: str) -> str:
    if not html_content or not _get_display_to_internal_toc_map():
        return html_content

    soup = BeautifulSoup(html_content, "html.parser")
    changed = False

    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if not isinstance(tag, Tag):
            continue

        original_text = tag.get_text("", strip=False)
        display_text = _replace_leading_internal_toc_number_for_display(original_text)
        if display_text == original_text:
            continue

        tag.clear()
        tag.string = display_text
        changed = True

    return str(soup) if changed else html_content


def _renumber_report_markdown_for_display(markdown_content: str) -> str:
    if not markdown_content or not _get_display_to_internal_toc_map():
        return markdown_content

    display_lines: list[str] = []
    changed = False
    for line in str(markdown_content).replace("\\r\\n", "\n").replace("\\n", "\n").splitlines():
        heading_match = re.match(r"^(\s{0,3}#{1,6}\s+)(.*)$", line)
        if not heading_match:
            display_lines.append(line)
            continue

        heading_text = heading_match.group(2)
        display_heading_text = _replace_leading_internal_toc_number_for_display(heading_text)
        if display_heading_text != heading_text:
            changed = True
        display_lines.append(f"{heading_match.group(1)}{display_heading_text}")

    return "\n".join(display_lines) if changed else markdown_content


def _has_report_prerequisites() -> bool:
    if st.session_state.get("auto_mode"):
        checks: list[bool] = []
        if _auto_stage_was_planned("loading_auto"):
            checks.append(bool(st.session_state.get("summary_1")) and stage_is_current(st.session_state, "loading"))
        if _auto_stage_was_planned("prep_auto"):
            checks.append(
                bool(st.session_state.get("summary_2"))
                and stage_is_current(
                    st.session_state,
                    "preprocessing",
                    input_fingerprint=current_dataset_fingerprint(st.session_state),
                )
            )
        if _auto_stage_was_planned("vis_auto"):
            checks.append(bool(st.session_state.get("summary_3")) and stage_is_current(st.session_state, "visualization"))
        if _auto_stage_was_planned("modeling_auto"):
            checks.append(bool(st.session_state.get("summary_4")) and stage_is_current(st.session_state, "modeling"))
        return all(checks)

    return bool(
        st.session_state.get("summary_1")
        and stage_is_current(st.session_state, "loading")
        and st.session_state.get("summary_2")
        and stage_is_current(st.session_state, "preprocessing")
        and st.session_state.get("summary_3")
        and stage_is_current(st.session_state, "visualization")
        and st.session_state.get("summary_4")
        and stage_is_current(st.session_state, "modeling")
    )


def _has_usable_visualization_source(source: Any) -> bool:
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


def _source_to_visualization_dataframe(source: Any) -> pd.DataFrame | None:
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


def _resolve_visualization_dataframe_for_report(preproc_agent, load_agent) -> pd.DataFrame | None:
    processed_df = preproc_agent.load_processed_df()
    if _has_usable_visualization_source(processed_df):
        return _source_to_visualization_dataframe(processed_df)

    summary_2 = st.session_state.get("summary_2")
    if isinstance(summary_2, dict):
        summary_processed_df = summary_2.get("processed_df")
        if _has_usable_visualization_source(summary_processed_df):
            return _source_to_visualization_dataframe(summary_processed_df)

    cached_processed_df = st.session_state.get("prep_result_from_summary_2")
    if _has_usable_visualization_source(cached_processed_df):
        return _source_to_visualization_dataframe(cached_processed_df)

    raw_df = load_agent.load_df()
    if _has_usable_visualization_source(raw_df):
        return _source_to_visualization_dataframe(raw_df)

    return None


def _has_generated_outline(report_agent) -> bool:
    return bool(normalize_toc_list(report_agent.load_outline()))


def _has_generated_word_report(report_agent) -> bool:
    return bool(report_agent.load_report_content() or report_agent.load_html() or report_agent.load_word())


def _has_visualization_recommendation(visualization_agent) -> bool:
    if visualization_agent is None:
        return False

    suggestion = (
        st.session_state.get("visual_recommendatio")
        or st.session_state.get("viz_suggestion")
        or visualization_agent.load_suggestion()
    )
    return bool(stringify_string(suggestion))


def _ensure_visualization_ready_for_report(visualization_agent) -> bool:
    if visualization_agent is None or not _has_visualization_recommendation(visualization_agent):
        st.warning(bt("如需生成图文报告，请先完成可视化推荐部分。", "Complete the visualization recommendations before generating a report with charts."))
        return False

    if not visualization_agent.load_code():
        if not generate_visualization_code_once(visualization_agent):
            st.warning(bt("未能自动生成可视化代码，请先前往可视化页面检查推荐结果。", "Visualization code could not be generated automatically. Check the visualization recommendations first."))
            return False

    if not visualization_agent.load_fig():
        if not execute_visualization_code_once(visualization_agent):
            st.warning(bt("未能自动生成可视化结果，请先前往可视化页面检查代码或数据。", "Visualization results could not be generated automatically. Check the visualization code or data first."))
            return False

    return True


def _loaded_visualization_figure_count() -> int | None:
    visualization_agent = st.session_state.get("visualization_agent")
    if visualization_agent is None:
        return None

    try:
        fig_desc_list = visualization_agent.load_fig() or []
    except Exception:
        return None

    return len(fig_desc_list) if fig_desc_list else None


def _has_in_range_figure_refs(content: str, figure_count: int | None) -> bool:
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


def _visualization_title_items(max_figures: int | None = None) -> list[str]:
    title_items = _normalize_visualization_titles(st.session_state.get("tu_title"))

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


def _selected_full_content_from_fig_analysis(
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
    title_items = _visualization_title_items(max_figures)
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


def _resolve_selected_full_content(
    *,
    visual_summary: Any,
    allow_report_cache: bool,
) -> tuple[str, str]:
    figure_count = _loaded_visualization_figure_count()
    fallback_content, fallback_count = _selected_full_content_from_fig_analysis(
        visual_summary,
        max_figures=figure_count,
    )
    if fallback_content:
        return fallback_content, f"summary_3.fig_analysis ({fallback_count} items)"

    full_content = stringify_string(st.session_state.get("full"))
    if full_content and _has_in_range_figure_refs(full_content, figure_count):
        return normalize_figure_placeholders(full_content), "session_state.full"

    if allow_report_cache:
        cached_content = stringify_string(st.session_state.get("report_selected_full_conten"))
        if cached_content and _has_in_range_figure_refs(cached_content, figure_count):
            return normalize_figure_placeholders(cached_content), "report_selected_full_conten"

    if full_content:
        return normalize_figure_placeholders(full_content), "session_state.full"

    if allow_report_cache:
        cached_content = stringify_string(st.session_state.get("report_selected_full_conten"))
        if cached_content:
            return normalize_figure_placeholders(cached_content), "report_selected_full_conten"

    return "", "empty"


def _log_selected_full_content(stage: str, content: str, source: str) -> None:
    fig_refs = re.findall(FIG_PLACEHOLDER_CAPTURE_PATTERN, normalize_figure_placeholders(content or ""), flags=re.IGNORECASE)
    print(
        f"[REPORT][INPUT] {stage} selected_full_conten source={source}, "
        f"length={len(content or '')}, fig_refs={fig_refs}"
    )


def _build_report_inputs(load_agent, report_agent) -> dict[str, Any]:
    report_language = _get_report_generation_language(report_agent)
    load_summary = maybe_json_loads(_resolve_loading_field(load_agent, "summary_1", {}))
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

    load_summary = _report_stage_value("loading_auto", load_summary, {})
    preproc_summary = _report_stage_value("prep_auto", preproc_summary, {})
    visual_summary = _report_stage_value("vis_auto", visual_summary, {})
    coding_summary = _report_stage_value("modeling_auto", coding_summary, {})

    selected_full_content, selected_source = _resolve_selected_full_content(
        visual_summary=visual_summary,
        allow_report_cache=False,
    )
    selected_full_content = _report_stage_value("vis_auto", selected_full_content, "")
    _log_selected_full_content("toc", selected_full_content, selected_source)

    return {
        "load_summary": load_summary,
        "preproc_summary": preproc_summary,
        "visual_summary": visual_summary,
        "coding_summary": coding_summary,
        "selected_full_conten": selected_full_content,
        "load_abstract": _report_stage_value(
            "loading_auto",
            stringify_string(_resolve_loading_field(load_agent, "abstract_1", "")),
            "",
        ),
        "preproc_abstract": _report_stage_value(
            "prep_auto",
            stringify_string(st.session_state.get("abstract_2", "")),
            "",
        ),
        "visual_abstract": _report_stage_value(
            "vis_auto",
            stringify_string(st.session_state.get("abstract_3", "")),
            "",
        ),
        "coding_abstract": _report_stage_value(
            "modeling_auto",
            stringify_string(st.session_state.get("abstract_4", "")),
            "",
        ),
        "toc_md": normalize_toc_list(report_agent.load_outline()),
        "outline_length": str(report_agent.load_outline_length() or ""),
        "preference_selected": stringify_string(st.session_state.get("preference_selected")),
        "add_preference": stringify_string(st.session_state.get("add_preference")),
        "report_auto": True,
        "user_input": str(report_agent.load_user_input() or ""),
        "report_language": report_language,
        "language_instruction": report_language_instruction(report_language),
        "language_name": report_language_name(report_language),
    }


def _build_word_report_inputs(report_agent, status_placeholder: Any | None = None) -> dict[str, Any]:
    report_language = _get_report_generation_language(report_agent)
    source_toc_text = _filter_toc_text_for_available_content(
        _normalize_multiline_text(report_agent.load_outline())
    )
    generation_toc_text = _outline_text_for_report_language(
        source_toc_text,
        report_language,
        status_placeholder=status_placeholder,
    )
    current_coding_abstract = stringify_string(
        st.session_state.get("abstract_4") or st.session_state.get("modeling_abstract_4")
    )
    if not current_coding_abstract:
        current_coding_abstract = stringify_string(st.session_state.get("report_coding_abstract"))
    current_coding_abstract = _report_stage_value("modeling_auto", current_coding_abstract, "")

    visual_summary = maybe_json_loads(st.session_state.get("summary_3", {}))
    current_selected_full_content, selected_source = _resolve_selected_full_content(
        visual_summary=visual_summary if isinstance(visual_summary, dict) else {},
        allow_report_cache=True,
    )
    current_selected_full_content = _report_stage_value("vis_auto", current_selected_full_content, "")
    _log_selected_full_content("word", current_selected_full_content, selected_source)
    return {
        "toc_text": generation_toc_text,
        "respect_user_toc": bool(st.session_state.get(REPORT_OUTLINE_USER_EDITED_KEY)),
        "title": "",
        "report_language": report_language,
        "language_instruction": report_language_instruction(report_language),
        "language_name": report_language_name(report_language),
        "selected_full_conten": current_selected_full_content,
        "preference_selected": stringify_string(st.session_state.get("report_preference_selected")),
        "add_preference": stringify_string(st.session_state.get("report_add_preference")),
        "user_input": str(report_agent.load_user_input() or ""),
        "load_abstract": _report_stage_value(
            "loading_auto",
            stringify_string(st.session_state.get("report_load_abstract")),
            "",
        ),
        "preproc_abstract": _report_stage_value(
            "prep_auto",
            stringify_string(st.session_state.get("report_preproc_abstract")),
            "",
        ),
        "visual_abstract": _report_stage_value(
            "vis_auto",
            stringify_string(st.session_state.get("report_visual_abstract")),
            "",
        ),
        "coding_abstract": current_coding_abstract,
    }


def _report_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _get_report_worker_ref_context(inputs: dict[str, Any]) -> str:
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
        query = (
            f"report writing business context {inputs.get('add_preference', '')}"
            if is_english_report(inputs.get("report_language"))
            else f"报告撰写 业务背景 {inputs.get('add_preference', '')}"
        )
        return retriever.retrieve_and_format(
            query,
            top_k=3,
        )
    except Exception as exc:
        print("[REPORT][JOB] reference retrieval failed:", repr(exc))
        return ""


def _build_report_worker_payload(report_agent, status_placeholder: Any | None = None) -> dict[str, Any]:
    inputs = _build_word_report_inputs(report_agent, status_placeholder=status_placeholder)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    inputs.setdefault("preference_select", st.session_state.get("preference_selected") or "")
    inputs["ref_context"] = _get_report_worker_ref_context(inputs)

    llm_config = {
        "api_key": st.session_state.get("llm_api_key") or "",
        "base_url": st.session_state.get("llm_base_url") or "",
        "model": st.session_state.get("llm_model") or "",
    }

    return {
        "inputs": inputs,
        "llm_config": llm_config,
    }


def _cleanup_report_job_files(job: dict[str, Any] | None) -> None:
    if not isinstance(job, dict):
        return

    work_dir = job.get("work_dir")
    if not work_dir:
        return

    try:
        work_path = Path(str(work_dir)).resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if temp_root not in (work_path, *work_path.parents):
            print("[REPORT][JOB] skip cleanup outside temp dir:", work_path)
            return
        if not work_path.name.startswith("autostat_report_"):
            print("[REPORT][JOB] skip cleanup for unexpected temp dir:", work_path)
            return
        shutil.rmtree(work_path, ignore_errors=True)
    except Exception as exc:
        print("[REPORT][JOB] cleanup failed:", repr(exc))


def _terminate_report_generation_process() -> None:
    job = st.session_state.get(REPORT_GENERATION_JOB_KEY)
    process = st.session_state.get(REPORT_GENERATION_PROCESS_KEY)
    if process is None and isinstance(job, dict):
        process = job.get("process")

    poll = getattr(process, "poll", None)
    if callable(poll):
        try:
            if process.poll() is None:
                print(f"[REPORT][JOB] terminate previous report process pid={getattr(process, 'pid', None)}")
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        except Exception as exc:
            print("[REPORT][JOB] terminate failed:", repr(exc))

    if isinstance(job, dict):
        _cleanup_report_job_files(job)
        token = job.get("token")
        if token and _is_current_report_generation(token):
            st.session_state[REPORT_GENERATION_RUNNING_KEY] = False

    st.session_state.pop(REPORT_GENERATION_PROCESS_KEY, None)
    st.session_state.pop(REPORT_GENERATION_JOB_KEY, None)


def _start_report_generation_process(
    report_agent,
    action: str,
    status_placeholder: Any | None = None,
) -> bool:
    _terminate_report_generation_process()
    generation_token = _begin_report_generation(report_agent)

    work_dir = tempfile.mkdtemp(prefix="autostat_report_")
    input_path = os.path.join(work_dir, "input.json")
    output_path = os.path.join(work_dir, "output.json")
    progress_path = os.path.join(work_dir, "progress.json")
    payload = _build_report_worker_payload(report_agent, status_placeholder=status_placeholder)
    llm_config = payload.get("llm_config") if isinstance(payload.get("llm_config"), dict) else {}

    try:
        with open(input_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

        env = os.environ.copy()
        if llm_config.get("api_key"):
            env["OPENAI_API_KEY"] = str(llm_config["api_key"])
        if llm_config.get("base_url"):
            env["OPENAI_BASE_URL"] = str(llm_config["base_url"])
        if llm_config.get("model"):
            env["OPENAI_MODEL"] = str(llm_config["model"])

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "workflows.reporting_partly_worker",
                input_path,
                output_path,
                progress_path,
            ],
            cwd=str(_report_repo_root()),
            env=env,
        )
    except Exception as exc:
        _cleanup_report_job_files({"work_dir": work_dir})
        _finish_report_generation(generation_token)
        st.error(bt("报告生成进程启动失败：{error}", "Failed to start the report generation process: {error}", error=exc))
        return False

    job = {
        "token": generation_token,
        "action": action,
        "work_dir": work_dir,
        "input_path": input_path,
        "output_path": output_path,
        "progress_path": progress_path,
        "pid": process.pid,
        "started_at": time.time(),
        "process": process,
    }
    st.session_state[REPORT_GENERATION_PROCESS_KEY] = process
    st.session_state[REPORT_GENERATION_JOB_KEY] = job
    st.session_state[REPORT_GENERATION_RUNNING_KEY] = True
    print(f"[REPORT][JOB] started report process pid={process.pid}")
    return True


def _read_report_worker_output(job: dict[str, Any]) -> dict[str, Any] | None:
    output_path = job.get("output_path")
    if not output_path or not os.path.exists(str(output_path)):
        return None
    try:
        with open(str(output_path), "r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        return {
            "ok": False,
            "error": bt(
                "报告生成结果读取失败：{error}",
                "Failed to read report generation output: {error}",
                error=exc,
            ),
        }


def _read_report_worker_progress(job: dict[str, Any]) -> dict[str, Any] | None:
    progress_path = job.get("progress_path")
    if not progress_path or not os.path.exists(str(progress_path)):
        return None

    try:
        with open(str(progress_path), "r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None
    except Exception as exc:
        print("[REPORT][PROGRESS] read failed:", repr(exc))
        return None


def _format_progress_section_title(progress: dict[str, Any]) -> str:
    section_title = stringify_string(progress.get("section_title")).strip()
    section_num = stringify_string(progress.get("section_num")).strip()
    if section_num:
        section_title = re.sub(
            rf"^\s*{re.escape(section_num)}(?:[\.．、]|\s+)?\s*",
            "",
            section_title,
        ).strip()
    return section_title


def _safe_progress_int(progress: dict[str, Any], key: str) -> int:
    try:
        return int(progress.get(key) or 0)
    except Exception:
        return 0


def _format_report_progress_status(
    action: str,
    progress: dict[str, Any] | None,
    report_language: Any = REPORT_LANGUAGE_ZH,
) -> str:
    if is_english_report(report_language):
        action_label = action or "selected"
        if not isinstance(progress, dict):
            return f"Generating the {action_label} English report. Please wait."

        phase = stringify_string(progress.get("phase")).strip()
        section_title = _format_progress_section_title(progress)
        total_sections = _safe_progress_int(progress, "total_sections")
        section_index = _safe_progress_int(progress, "section_index")
        completed_sections = _safe_progress_int(progress, "completed_sections")

        if progress.get("status") == "finalizing" or phase == "body_completed":
            return f"The report body is complete. Finalizing the title and preparing the {action_label} file."

        if phase == "section_draft" and section_title:
            return f"The draft for \"{section_title}\" is ready. Polishing it into final prose. ({section_index}/{total_sections})"

        if phase == "section_completed" and section_title:
            return f"\"{section_title}\" is complete. Continuing with the next section. ({completed_sections}/{total_sections} done)"

        if phase == "section_started" and section_title:
            return f"Writing \"{section_title}\". ({section_index}/{total_sections})"

        if total_sections:
            return f"Generating the {action_label} English report body. ({completed_sections}/{total_sections} sections done)"

        return f"Generating the {action_label} English report. Please wait."

    if not isinstance(progress, dict):
        return f"正在生成{action}报告，请耐心等待"

    phase = stringify_string(progress.get("phase")).strip()
    section_title = _format_progress_section_title(progress)
    total_sections = _safe_progress_int(progress, "total_sections")
    section_index = _safe_progress_int(progress, "section_index")
    completed_sections = _safe_progress_int(progress, "completed_sections")

    if progress.get("status") == "finalizing" or phase == "body_completed":
        return f"报告正文已完成，正在整理标题并准备生成{action}文件。"

    if phase == "section_draft" and section_title:
        return f"“{section_title}”草稿已生成，正在整理为正式段落。（{section_index}/{total_sections}）"

    if phase == "section_completed" and section_title:
        return f"“{section_title}”已完成，继续生成后续内容。（已完成 {completed_sections}/{total_sections} 节）"

    if phase == "section_started" and section_title:
        return f"正在撰写“{section_title}”。（{section_index}/{total_sections}）"

    if total_sections:
        return f"正在生成{action}报告正文。（已完成 {completed_sections}/{total_sections} 节）"

    return f"正在生成{action}报告，请耐心等待"


def _remember_live_report_progress(progress: dict[str, Any] | None) -> None:
    if not isinstance(progress, dict):
        return

    html_content = stringify_string(progress.get("html")).strip()
    markdown_content = stringify_string(progress.get("markdown")).strip()
    existing_preview = st.session_state.get(REPORT_PENDING_PREVIEW_KEY)
    preview = dict(existing_preview) if isinstance(existing_preview, dict) else {}

    if html_content:
        preview = {
            "html": html_content,
            "markdown": markdown_content,
            "progress": progress,
            "live": True,
        }
    elif markdown_content:
        preview = {
            "html": "",
            "markdown": markdown_content,
            "progress": progress,
            "live": True,
        }
    else:
        preview["progress"] = progress

    st.session_state[REPORT_PENDING_PREVIEW_KEY] = preview


def _save_formatted_report_result(
    report_agent,
    action: str,
    workflow_result: dict[str, Any],
    generation_token: str | None,
) -> bool:
    status_placeholder = st.empty()
    report_language = _get_report_generation_language(report_agent)
    raw_content = extract_report_html(workflow_result)
    if not raw_content:
        status_placeholder.empty()
        st.error(bt("Word 报告工作流未返回 `final_html`。", "The Word report workflow did not return `final_html`."))
        return False

    raw_content = raw_content.strip()
    report_title = _extract_report_title(workflow_result)
    if report_title and _is_current_report_generation(generation_token):
        st.session_state.report_title = report_title

    if _looks_like_html(raw_content):
        html_content = raw_content
        markdown_content = html_to_markdown(html_content)
    else:
        markdown_content = raw_content
        markdown_content = _deduplicate_report_html_blocks(markdown_content)
        html_content = markdown_to_html(markdown_content, title="")

    if _is_report_generation_cancelled(generation_token):
        return False

    html_content = _finalize_report_html(html_content, report_title, report_language=report_language)
    markdown_content = html_to_markdown(html_content) if html_content else markdown_content

    if _is_report_generation_cancelled(generation_token):
        return False

    _clear_generated_report_files(report_agent)

    if _is_report_generation_cancelled(generation_token):
        return False

    report_agent.save_report_workflow_result(workflow_result)
    report_agent.save_report(workflow_result)
    report_agent.save_report_content(markdown_content)
    report_agent.save_markdown(markdown_content)
    report_agent.save_html(html_content)
    _save_report_current_language(report_agent, report_language)
    _store_report_language_version(
        report_agent,
        report_language,
        html_content=html_content,
        markdown_content=markdown_content,
    )

    if is_english_report(report_language):
        status_placeholder.info("Preparing the report file. Please wait.")
    else:
        status_placeholder.info(bt("正在生成报告文件，请稍后。", "Preparing the report file. Please wait."))
    downloadable_reports = _prepare_downloadable_reports(report_agent, generation_token=generation_token)

    if _is_report_generation_cancelled(generation_token):
        return False

    if action == "Word" and downloadable_reports.get("word") is None:
        status_placeholder.empty()
        st.error(bt(
            "Word 报告内容已生成，但 Word 文件转换失败，请重试或切换为 HTML 报告。",
            "The report content was generated, but Word conversion failed. Try again or switch to HTML.",
        ))
        return False

    st.session_state.report_final_html = html_content
    _clear_pending_report_preview()
    status_placeholder.empty()
    return True


def _poll_report_generation_job(report_agent, action: str) -> str:
    job = st.session_state.get(REPORT_GENERATION_JOB_KEY)
    if not isinstance(job, dict):
        return "idle"

    process = st.session_state.get(REPORT_GENERATION_PROCESS_KEY) or job.get("process")
    poll = getattr(process, "poll", None)
    token = job.get("token")
    if not token or not _is_current_report_generation(token):
        _terminate_report_generation_process()
        return "idle"

    if not callable(poll):
        _cleanup_report_job_files(job)
        st.session_state.pop(REPORT_GENERATION_PROCESS_KEY, None)
        st.session_state.pop(REPORT_GENERATION_JOB_KEY, None)
        _finish_report_generation(token)
        st.error(bt("报告生成进程状态丢失，请重新生成。", "The report generation process state was lost. Please generate it again."))
        return "failed"

    return_code = process.poll()
    if return_code is None:
        progress = _read_report_worker_progress(job)
        _remember_live_report_progress(progress)
        st.info(_format_report_progress_status(action, progress, _get_report_current_language(report_agent)))
        st.session_state[REPORT_GENERATION_RUNNING_KEY] = True
        return "running"

    worker_payload = _read_report_worker_output(job)
    _cleanup_report_job_files(job)
    st.session_state.pop(REPORT_GENERATION_PROCESS_KEY, None)
    st.session_state.pop(REPORT_GENERATION_JOB_KEY, None)

    if _is_report_generation_cancelled(token):
        return "idle"

    if not worker_payload:
        _finish_report_generation(token)
        st.error(bt(
            "报告生成进程已退出（code={code}），但没有返回可用结果。",
            "The report generation process exited (code={code}) without returning usable output.",
            code=return_code,
        ))
        return "failed"

    if not worker_payload.get("ok"):
        _finish_report_generation(token)
        error_message = worker_payload.get("error") or bt("未知错误", "Unknown error")
        st.error(bt("报告生成失败：{error}", "Report generation failed: {error}", error=error_message))
        traceback_text = stringify_string(worker_payload.get("traceback"))
        if traceback_text:
            print("[REPORT][JOB] worker traceback:\n", traceback_text)
        return "failed"

    workflow_result = _merge_report_workflow_results([worker_payload.get("result")])
    if workflow_result is None:
        _finish_report_generation(token)
        st.error(bt("Word 报告生成失败，未解析到有效输出，请重新生成。", "Word report generation failed because no valid output was parsed. Please generate it again."))
        return "failed"

    success = _save_formatted_report_result(report_agent, action, workflow_result, token)
    _finish_report_generation(token)
    if success:
        if is_english_report(_get_report_current_language(report_agent)):
            st.success(f"The {action} English report is ready and shown below.")
        else:
            st.success(bt("{action} 报告已生成，已在下方展示。", "The {action} report is ready and shown below.", action=action))
        return "complete"
    return "failed"


def _is_report_generation_job_running() -> bool:
    job = st.session_state.get(REPORT_GENERATION_JOB_KEY)
    process = st.session_state.get(REPORT_GENERATION_PROCESS_KEY)
    poll = getattr(process, "poll", None)
    return isinstance(job, dict) and callable(poll) and process.poll() is None


def call_report_workflow(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Run the local report-outline workflow."""
    from utils.local_workflow_bridge import call_reporting_toc_bridge

    status_placeholder = st.empty()
    report_language = normalize_report_language(inputs.get("report_language"))
    if is_english_report(report_language):
        status_placeholder.info("Generating the English outline. Please wait.")
    else:
        status_placeholder.info("正在生成目录，请稍后。")

    inputs = dict(inputs)
    inputs.setdefault("report_language", report_language)
    inputs.setdefault("language_instruction", report_language_instruction(report_language))
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    inputs.setdefault("preference_selected", st.session_state.get("preference_selected") or "")

    result = call_reporting_toc_bridge(inputs)
    if result is None:
        status_placeholder.empty()
        return None

    merged_result = _merge_report_workflow_results([result])
    if merged_result is not None:
        status_placeholder.empty()
        return merged_result

    status_placeholder.empty()
    st.error(bt("目录工作流已完成，但未解析到有效输出。", "The outline workflow completed, but no valid output was parsed."))
    return None


def call_report_document_workflow(
    inputs: dict[str, Any],
    status_placeholder: Any | None = None,
    clear_on_success: bool = True,
    generation_token: str | None = None,
) -> dict[str, Any] | None:
    """Run the local report-writing workflow."""
    from utils.local_workflow_bridge import call_reporting_partly_bridge

    if status_placeholder is None:
        status_placeholder = st.empty()

    inputs = dict(inputs)
    report_language = normalize_report_language(inputs.get("report_language"))
    if is_english_report(report_language):
        status_placeholder.info("Generating the English report. Please wait.")
    else:
        status_placeholder.info("正在生成报告，请稍后。")
    inputs.setdefault("report_language", report_language)
    inputs.setdefault("language_instruction", report_language_instruction(report_language))
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    # report workflow 字段名是 preference_select
    inputs.setdefault("preference_select", st.session_state.get("preference_selected") or "")

    if generation_token is not None:
        inputs["_report_cancel_check"] = lambda: _is_report_generation_cancelled(generation_token)

    result = call_reporting_partly_bridge(inputs)
    if result is None:
        status_placeholder.empty()
        return None

    merged_result = _merge_report_workflow_results([result])
    if merged_result is not None:
        if clear_on_success:
            status_placeholder.empty()
        return merged_result

    status_placeholder.empty()
    st.error(bt(
        "Word 报告生成失败，未解析到有效输出，请点击生成报告按钮重新生成。",
        "Word report generation failed because no valid output was parsed. Click Generate Report to try again.",
    ))
    return None


def _format_report_translation_progress_status(
    progress: dict[str, Any] | None,
    target_language: Any,
) -> str:
    target_label = _target_language_label(target_language)

    def progress_snippet() -> str:
        raw = stringify_string(progress.get("section_title") if isinstance(progress, dict) else "").strip()
        if not raw:
            return ""
        clean = re.sub(r"\s+", " ", raw).strip()
        clean = re.sub(r"\.{3,}$", "", clean).strip()
        return f"{clean[:15]}..."

    if is_english_report(target_language):
        if not isinstance(progress, dict):
            return "Translating the report into English. Please wait."
        phase = stringify_string(progress.get("phase")).strip()
        total_blocks = _safe_progress_int(progress, "total_blocks")
        block_index = _safe_progress_int(progress, "block_index")
        completed_blocks = _safe_progress_int(progress, "completed_blocks")
        section_title = progress_snippet()
        if phase == "completed":
            return "English conversion is complete. Preparing the export files."
        if phase == "block_completed" and total_blocks:
            return f"One block has been converted while preserving the layout. ({completed_blocks}/{total_blocks})"
        if phase == "block_started" and section_title:
            return f"Translating: {section_title} ({block_index + 1}/{total_blocks})"
        if total_blocks:
            return f"Translating into English. ({completed_blocks}/{total_blocks} blocks complete)"
        return "Translating the report into English. Please wait."

    if not isinstance(progress, dict):
        return f"正在转换为{target_label}，请稍后。"

    phase = stringify_string(progress.get("phase")).strip()
    total_blocks = _safe_progress_int(progress, "total_blocks")
    block_index = _safe_progress_int(progress, "block_index")
    completed_blocks = _safe_progress_int(progress, "completed_blocks")
    section_title = progress_snippet()

    if phase == "completed":
        return f"已完成{target_label}转换，正在整理导出文件。"
    if phase == "block_completed" and total_blocks:
        return f"已转换一段内容，继续保持原排版处理后续段落。（{completed_blocks}/{total_blocks}）"
    if phase == "block_started" and section_title:
        return f"正在转换：{section_title}（{block_index + 1}/{total_blocks}）"
    if total_blocks:
        return f"正在转换为{target_label}，已完成 {completed_blocks}/{total_blocks} 段。"
    return f"正在转换为{target_label}，请稍后。"


def _current_report_html_for_conversion(report_agent) -> str:
    html_content = stringify_string(report_agent.load_html()).strip()
    if html_content:
        return html_content

    markdown_content = stringify_string(report_agent.load_markdown() or report_agent.load_report_content()).strip()
    if markdown_content:
        return markdown_to_html(markdown_content, title="")

    workflow_result = report_agent.load_report_workflow_result()
    raw_content = extract_report_html(workflow_result) if workflow_result else ""
    if raw_content:
        if _looks_like_html(raw_content):
            return raw_content
        return markdown_to_html(raw_content, title="")
    return ""


def _convert_report_outline_language(report_agent, target_language: Any, display_toc_text: str) -> None:
    from workflows.report_translation import translate_report_toc

    target_language = normalize_report_language(target_language)
    target_label = _target_language_ui_label(target_language)
    source_display_toc = _normalize_multiline_text(display_toc_text).strip()
    if not source_display_toc:
        st.warning(bt("当前没有可转换的目录。", "There is no outline to convert."))
        return
    source_language = _guess_report_language_from_text(
        source_display_toc,
        fallback=_get_report_generation_language(report_agent),
    )
    if source_language == target_language:
        _set_report_outline_conversion_flash(
            bt(
                "当前目录看起来已经是{language}。",
                "The current outline already appears to be in {language}.",
                language=target_label,
            ),
            target_language=target_language,
            tone="info",
        )
        return

    with st.spinner(bt("正在转换目录为{language}...", "Converting the outline to {language}...", language=target_label)):
        converted_display_toc = translate_report_toc(
            source_display_toc,
            source_language=source_language,
            target_language=target_language,
        )

    if not converted_display_toc:
        st.error(bt("目录转换失败，请稍后重试。", "Outline conversion failed. Please try again later."))
        return

    internal_toc_text = _display_toc_text_to_internal(
        converted_display_toc,
        previous_display_toc=source_display_toc,
    )
    report_agent.save_outline(internal_toc_text)
    st.session_state[REPORT_DISPLAY_OUTLINE_KEY] = converted_display_toc
    st.session_state[REPORT_DISPLAY_TO_INTERNAL_TOC_MAP_KEY] = _build_display_to_internal_map_from_texts(
        converted_display_toc,
        internal_toc_text,
    )
    st.session_state[REPORT_OUTLINE_USER_EDITED_KEY] = True
    _set_report_outline_conversion_flash(
        bt("目录已转换为{language}。", "The outline has been converted to {language}.", language=target_label),
        target_language=target_language,
        tone="success",
    )
    st.rerun()


def _convert_report_content_language(report_agent, target_language: Any) -> None:
    from workflows.report_translation import translate_report_html

    target_language = normalize_report_language(target_language)
    target_label = _target_language_ui_label(target_language)
    source_language = _get_report_current_language(report_agent)
    if source_language == target_language:
        st.info(bt("当前报告已经是{language}。", "The current report is already in {language}.", language=target_label))
        return

    target_version = _load_report_language_version(report_agent, target_language)
    if target_version:
        _store_report_language_version(report_agent, source_language)
        _restore_report_language_version(report_agent, target_language, target_version)
        st.success(bt(
            "已切换为{language}版本，内容和排版结构已保留。",
            "Switched to the {language} version. Content and layout structure were preserved.",
            language=target_label,
        ))
        st.rerun()
        return

    source_html = _current_report_html_for_conversion(report_agent)
    if not source_html:
        st.warning(bt("当前没有可转换的报告正文，请先生成报告。", "There is no report body to convert. Generate a report first."))
        return

    _store_report_language_version(report_agent, source_language)
    status_placeholder = st.empty()
    preview_placeholder = st.empty()

    def on_progress(progress: dict[str, Any]) -> None:
        _remember_live_report_progress(progress)
        status_placeholder.info(_format_report_translation_progress_status(progress, target_language))
        with preview_placeholder.container():
            _render_pending_report_preview()

    status_placeholder.info(_format_report_translation_progress_status(None, target_language))
    result = translate_report_html(
        source_html,
        source_language=source_language,
        target_language=target_language,
        progress_callback=on_progress,
    )
    translated_html = stringify_string(result.get("html")).strip()
    if not translated_html:
        status_placeholder.empty()
        st.error(bt("报告正文转换失败，请稍后重试。", "Report body conversion failed. Please try again later."))
        return

    translated_html = _set_report_html_language(translated_html, target_language)
    translated_markdown = html_to_markdown(translated_html)
    report_agent.save_html(translated_html)
    report_agent.save_markdown(translated_markdown)
    report_agent.save_report_content(translated_markdown)
    _save_report_current_language(report_agent, target_language)
    _save_report_generation_language(report_agent, target_language)
    _store_report_language_version(
        report_agent,
        target_language,
        html_content=translated_html,
        markdown_content=translated_markdown,
    )
    _clear_report_binary_exports(report_agent)
    st.session_state.report_final_html = translated_html
    _clear_pending_report_preview()
    status_placeholder.empty()
    preview_placeholder.empty()
    st.success(bt(
        "报告正文已转换为{language}，排版结构已保留。",
        "The report body has been converted to {language}; the layout structure was preserved.",
        language=target_label,
    ))
    st.rerun()


def _render_report_language_conversion_controls(report_agent) -> None:
    if not _has_generated_word_report(report_agent):
        return

    current_language = _get_report_current_language(report_agent)
    conversion_running = _is_report_generation_job_running()
    requested_target_language: str | None = None
    col_zh, col_en = st.columns(2)
    with col_zh:
        if st.button(
            bt("一键转换为中文", "Convert to Chinese"),
            disabled=conversion_running or current_language == REPORT_LANGUAGE_ZH,
            use_container_width=True,
            key="convert_report_to_zh",
        ):
            requested_target_language = REPORT_LANGUAGE_ZH
    with col_en:
        if st.button(
            bt("一键转换为英文", "Convert to English"),
            disabled=conversion_running or current_language == REPORT_LANGUAGE_EN,
            use_container_width=True,
            key="convert_report_to_en",
        ):
            requested_target_language = REPORT_LANGUAGE_EN

    if requested_target_language:
        _convert_report_content_language(report_agent, requested_target_language)


def _generate_formatted_report(report_agent, action: str) -> None:
    report_language = _get_report_generation_language(report_agent)
    startup_placeholder = st.empty()
    startup_placeholder.info(_format_report_startup_status(action, report_language))
    if _start_report_generation_process(
        report_agent,
        action,
        status_placeholder=startup_placeholder,
    ):
        startup_placeholder.info(_format_report_progress_status(action, None, report_language))
    return

    generation_token = _begin_report_generation(report_agent)
    status_placeholder = st.empty()
    workflow_result = call_report_document_workflow(
        _build_word_report_inputs(report_agent, status_placeholder=status_placeholder),
        status_placeholder=status_placeholder,
        clear_on_success=False,
        generation_token=generation_token,
    )
    if not workflow_result:
        _finish_report_generation(generation_token)
        return

    if _is_report_generation_cancelled(generation_token):
        return

    raw_content = extract_report_html(workflow_result)
    if not raw_content:
        status_placeholder.empty()
        st.error(bt("Word 报告工作流未返回 `final_html`。", "The Word report workflow did not return `final_html`."))
        _finish_report_generation(generation_token)
        return

    raw_content = raw_content.strip()
    report_title = _extract_report_title(workflow_result)
    if report_title and _is_current_report_generation(generation_token):
        st.session_state.report_title = report_title

    if _looks_like_html(raw_content):
        html_content = raw_content
        markdown_content = html_to_markdown(html_content)
    else:
        markdown_content = raw_content
        markdown_content = _deduplicate_report_html_blocks(markdown_content)
        html_content = markdown_to_html(markdown_content, title="")

    if _is_report_generation_cancelled(generation_token):
        return

    html_content = _finalize_report_html(html_content, report_title, report_language=report_language)
    markdown_content = html_to_markdown(html_content) if html_content else markdown_content

    if _is_report_generation_cancelled(generation_token):
        return

    _clear_generated_report_files(report_agent)

    if _is_report_generation_cancelled(generation_token):
        return

    report_agent.save_report_workflow_result(workflow_result)
    report_agent.save_report(workflow_result)
    report_agent.save_report_content(markdown_content)
    report_agent.save_markdown(markdown_content)
    report_agent.save_html(html_content)
    _save_report_current_language(report_agent, report_language)
    _store_report_language_version(
        report_agent,
        report_language,
        html_content=html_content,
        markdown_content=markdown_content,
    )

    if is_english_report(report_language):
        status_placeholder.info("Preparing the report file. Please wait.")
    else:
        status_placeholder.info(bt("正在生成报告，请稍后。", "Generating the report. Please wait."))
    downloadable_reports = _prepare_downloadable_reports(report_agent, generation_token=generation_token)

    if _is_report_generation_cancelled(generation_token):
        return

    if action == "Word" and downloadable_reports.get("word") is None:
        status_placeholder.empty()
        st.error(bt(
            "Word 报告内容已生成，但 Word 文件转换失败，请重试或切换为 HTML 报告。",
            "The report content was generated, but Word conversion failed. Try again or switch to HTML.",
        ))
        _finish_report_generation(generation_token)
        return

    st.session_state.report_final_html = html_content
    _clear_pending_report_preview()
    status_placeholder.empty()
    if is_english_report(report_language):
        st.success(f"The {action} English report is ready and shown below.")
    else:
        st.success(bt("{action} 报告已生成，已在下侧展示。", "The {action} report is ready and shown below.", action=action))
    _finish_report_generation(generation_token)
    

def _normalize_report_block_for_dedup(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", "", text)
    return text


def _deduplicate_report_html_blocks(final_html: str) -> str:
    if not final_html:
        return final_html

    blocks = re.split(r"(?=^#{2,4}\s+)", final_html, flags=re.MULTILINE)
    if len(blocks) <= 1:
        return final_html.strip()

    deduped: list[str] = []
    seen: set[str] = set()

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        key = _normalize_report_block_for_dedup(block)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(block)

    return "\n\n".join(deduped).strip()


def report_basic_info(load_agent, report_agent, auto: bool) -> None:
    saved_language = sync_report_language(report_agent)
    if st.session_state.get(REPORT_LANGUAGE_WIDGET_SYNC_KEY) != saved_language:
        st.session_state[REPORT_LANGUAGE_SELECTOR_KEY] = _language_selector_index(saved_language)
        st.session_state[REPORT_LANGUAGE_WIDGET_SYNC_KEY] = saved_language
    selected_language_index = sac.segmented(
        items=[
            sac.SegmentedItem(label=_report_language_label(REPORT_LANGUAGE_ZH)),
            sac.SegmentedItem(label=_report_language_label(REPORT_LANGUAGE_EN)),
        ],
        label=bt("报告语言", "Report Language"),
        index=_language_selector_index(saved_language),
        align="center",
        size="sm",
        radius="sm",
        use_container_width=True,
        return_index=True,
        key=REPORT_LANGUAGE_SELECTOR_KEY,
    )
    selected_language = _language_from_selector_index(selected_language_index)
    if selected_language != saved_language:
        selected_language = _save_report_generation_language(report_agent, selected_language)
        set_language(selected_language)
        sync_report_language(report_agent, selected_language)
        st.session_state[REPORT_LANGUAGE_WIDGET_SYNC_KEY] = selected_language

    generated_language = _get_report_current_language(report_agent)
    if _has_generated_word_report(report_agent) and selected_language != generated_language:
        st.caption(bt(
            "语言选择将作用于下一次生成；已生成报告可在下方一键转换。",
            "The language choice applies to the next generation. Existing reports can be converted below.",
        ))
    else:
        st.markdown(REPORT_LANGUAGE_NOTICE_SPACER_HTML, unsafe_allow_html=True)

    outline_length = sac.segmented(
        items=[
            sac.SegmentedItem(label=bt("简要", "Brief")),
            sac.SegmentedItem(label=bt("标准", "Standard")),
            sac.SegmentedItem(label=bt("详细", "Detailed")),
        ],
        label=bt("详细程度", "Detail Level"),
        index=1,
        align="center",
        size="sm",
        radius="sm",
        use_container_width=True,
        key=REPORT_OUTLINE_LENGTH_SELECTOR_KEY,
    )
    report_agent.save_outline_length(outline_length)

    report_format = sac.chip(
        items=[
            sac.ChipItem(label="Word", icon=sac.BsIcon(name="file-earmark-word", size=16)),
            sac.ChipItem(label="HTML", icon=sac.BsIcon(name="filetype-html", size=16)),
            sac.ChipItem(label="PDF", icon=sac.BsIcon(name="file-earmark-pdf", size=16)),
        ],
        label=bt("选择报告生成格式", "Report Format"),
        index=[0],
        align="start",
        radius="md",
        multiple=False,
        key=REPORT_FORMAT_SELECTOR_KEY,
    )
    if auto:
        report_format = "Word"
    report_agent.save_report_format(_normalize_report_format(report_format))

    user_input = st.text_input(bt("报告生成要求", "Report Requirements"), bt("默认", "Default"))
    report_agent.save_user_input(user_input)
    visualization_agent = st.session_state.get("visualization_agent")

    if not auto and not _stage_has_report_content("vis_auto"):
        st.warning(bt("如需生成图文报告，请先完成可视化推荐部分。", "Complete the visualization recommendations before generating a report with charts."))

    not_generated = not _has_generated_outline(report_agent)
    if st.button(bt("生成目录", "Generate Outline")) or (auto and not_generated):
        if not auto and not _has_any_report_stage_content():
            st.warning(bt("请先完成前述任意流程，为报告撰写提供信息。", "Complete at least one previous workflow to provide material for the report."))
            return

        _clear_report_workflow_outputs(report_agent)
        report_agent.save_outline([])

        inputs = _build_report_inputs(load_agent, report_agent)
        workflow_result = call_report_workflow(inputs)

        if not workflow_result:
            return

        _save_report_workflow_outputs(report_agent, workflow_result)

        internal_toc_text = _filter_toc_text_for_available_content(
            _extract_toc_text_from_result(workflow_result)
        )
        if not internal_toc_text:
            st.error(bt("报告工作流未返回 `toc_text`。", "The report workflow did not return `toc_text`."))
            return

        report_agent.save_outline(internal_toc_text)
        _save_display_outline_from_internal(internal_toc_text)
        if auto:
            st.rerun()
        st.success(bt("目录已生成，已在右侧显示文本。", "The outline has been generated and is shown on the right."))


def report_outline(report_agent) -> None:
    st.subheader(bt("目录结构预览与编辑", "Outline Preview and Editing"))

    default_toc = _load_display_outline_from_internal(report_agent)
    generated_display_toc, _ = _renumber_toc_text_with_mapping(
        _normalize_multiline_text(report_agent.load_outline())
    )
    toc_text = st.text_area(
        bt("您可以在此处编辑目录结构，每行一个目录项", "Edit the outline here, with one item per line."),
        value=default_toc,
        height=260,
        placeholder=bt("# 数据分析报告\n## 1. 数据导入", "# Data Analysis Report\n## 1. Data Import"),
    )
    internal_toc_text = _display_toc_text_to_internal(toc_text, previous_display_toc=default_toc)
    st.session_state[REPORT_OUTLINE_USER_EDITED_KEY] = (
        _normalize_multiline_text(toc_text).strip() != _normalize_multiline_text(generated_display_toc).strip()
    )
    st.session_state[REPORT_DISPLAY_OUTLINE_KEY] = toc_text
    st.session_state[REPORT_DISPLAY_TO_INTERNAL_TOC_MAP_KEY] = _build_display_to_internal_map_from_texts(
        toc_text,
        internal_toc_text,
    )
    report_agent.save_outline(internal_toc_text)

    if _normalize_multiline_text(toc_text).strip():
        col_zh, col_en = st.columns(2)
        with col_zh:
            if st.button(
                bt("目录转中文", "Outline to Chinese"),
                use_container_width=True,
                key="convert_outline_to_zh",
            ):
                _convert_report_outline_language(report_agent, REPORT_LANGUAGE_ZH, toc_text)
            _render_report_outline_conversion_flash(REPORT_LANGUAGE_ZH)
        with col_en:
            if st.button(
                bt("目录转英文", "Outline to English"),
                use_container_width=True,
                key="convert_outline_to_en",
            ):
                _convert_report_outline_language(report_agent, REPORT_LANGUAGE_EN, toc_text)
            _render_report_outline_conversion_flash(REPORT_LANGUAGE_EN)


def report_save(report_agent, auto: bool) -> None:
    action = _normalize_report_format(report_agent.load_report_format())
    report_agent.save_report_format(action)
    visualization_agent = st.session_state.get("visualization_agent")
    job_state = _poll_report_generation_job(report_agent, action)

    outline_generated = _has_generated_outline(report_agent)
    report_generated = _has_generated_word_report(report_agent)
    not_generate = outline_generated and not report_generated and job_state != "running"

    if auto and report_generated and not report_agent.finish_auto_task:
        _complete_auto_report(report_agent)
        st.rerun()

    generate_clicked = st.button(bt("生成 {action} 报告", "Generate {action} Report", action=action))
    if generate_clicked or (auto and not_generate):
        if not auto and not _has_any_report_stage_content():
            st.warning(bt("如需生成图文报告，请先完成可视化推荐部分。", "Complete the visualization recommendations before generating a report with charts."))
            return

        if (
            _stage_has_report_content("vis_auto")
            and st.session_state.get("report_selected_full_conten") is None
            and st.session_state.get("full") is None
        ):
            st.warning(bt("请先点击“生成目录”获取新 workflow 输出。", "Click Generate Outline first to get fresh workflow output."))
            return

        if (
            not auto
            and _stage_has_report_content("vis_auto")
            and not _ensure_visualization_ready_for_report(visualization_agent)
        ):
            return

        _generate_formatted_report(report_agent, action)

        if auto:
            if isinstance(st.session_state.get(REPORT_GENERATION_JOB_KEY), dict):
                _complete_auto_report(report_agent)
                st.rerun()


def report_execution(report_agent) -> None:
    action = _normalize_report_format(report_agent.load_report_format())
    report_agent.save_report_format(action)
    if _is_report_generation_job_running():
        _render_pending_report_preview()
        time.sleep(1)
        st.rerun()
        return

    output_placeholder = st.empty()
    with output_placeholder.container():
        _render_report_language_conversion_controls(report_agent)
        downloadable_reports = _prepare_downloadable_reports(report_agent)
        if action == "PDF":
            downloadable_reports = _ensure_pdf_download_ready(report_agent, downloadable_reports)

        html_content = (downloadable_reports.get("html") or "").strip()
        markdown_content = (downloadable_reports.get("markdown") or "").strip()

        if action == "Word":
            st.download_button(
                label=bt("下载 Word 报告", "Download Word Report"),
                data=downloadable_reports["word"] or b"",
                file_name="report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                disabled=downloadable_reports["word"] is None,
            )
            _render_report_preview(html_content, markdown_content)
            return

        if action == "HTML":
            st.download_button(
                label=bt("下载 HTML 报告", "Download HTML Report"),
                data=html_content.encode("utf-8") if html_content else b"",
                file_name="report.html",
                mime="text/html",
                disabled=not bool(html_content),
            )
            _render_report_preview(html_content, markdown_content)
            return

        if action == "PDF":
            st.download_button(
                label=bt("下载 PDF 报告", "Download PDF Report"),
                data=downloadable_reports["pdf"] or b"",
                file_name="report.pdf",
                mime="application/pdf",
                disabled=downloadable_reports["pdf"] is None,
            )

            _render_report_preview(html_content, markdown_content)
            return

        _render_report_preview(html_content, markdown_content)
            

if __name__ == "__main__":
    st.title(bt("报告生成", "Report Generation"))
    st.markdown("---")

    load_agent = st.session_state.data_loading_agent
    preproc_agent = st.session_state.data_preprocess_agent
    planner = st.session_state.planner_agent
    auto = bool(st.session_state.auto_mode and planner.report_auto)

    if st.session_state.auto_mode and not _has_report_prerequisites():
        st.warning(bt("自动模式需要在前序步骤都生成结果后，才会进入报告生成。", "Auto mode enters report generation only after all previous stages have produced results."))
        st.stop()

    processed_df = preproc_agent.load_processed_df()
    df = processed_df if processed_df is not None else load_agent.load_df()

    if df is None:
        st.warning(bt("请先在数据导入页面加载数据。", "Please load data on the Data Import page first."))
        st.stop()

    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    sampled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    visualization_df = _resolve_visualization_dataframe_for_report(preproc_agent, load_agent)
    if isinstance(visualization_df, pd.DataFrame):
        visualization_df = visualization_df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        visualization_df = sampled_df
    report_agent = st.session_state.report_agent
    report_agent.add_df(sampled_df)
    visualization_agent = st.session_state.get("visualization_agent")
    if visualization_agent is not None:
        visualization_agent.add_df(visualization_df)
    outline_generated = _has_generated_outline(report_agent)
    report_generated = _has_generated_word_report(report_agent)

    columns = st.columns(2)
    with columns[0].expander(bt("报告设置", "Report Settings"), expanded=True):
        report_basic_info(load_agent, report_agent, auto)

    with columns[1].expander(bt("报告大纲", "Report Outline"), expanded=True):
        report_outline(report_agent)
        report_save(report_agent, auto)
        report_execution(report_agent)
