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


def _build_figure_caption(display_number: int, fig_index: int, title_items: list[str]) -> str:
    title_text = ""
    if 0 <= fig_index < len(title_items):
        title_text = _normalize_figure_title_text(title_items[fig_index])

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
) -> Tag | None:
    if fig_index < 0 or fig_index >= len(fig_desc_list):
        print(f"[REPORT][FIG] fig index out of range: {fig_index}")
        return None

    fig_item = fig_desc_list[fig_index]
    fig = _normalize_visual_figure(fig_item.get("fig") if isinstance(fig_item, dict) else fig_item)
    if fig is None:
        print(f"[REPORT][FIG] fig at index {fig_index} cannot be normalized")
        return None

    image_uri = _get_cached_figure_data_uri(fig_index, fig, image_uri_cache)
    if not image_uri:
        print(f"[REPORT][FIG] fig at index {fig_index} cannot convert to image")
        return None

    caption_text = _build_figure_caption(display_number, fig_index, title_items)

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
) -> int:
    inserted_count = _replace_paragraph_placeholders_at_boundaries(
        soup=soup,
        fig_desc_list=fig_desc_list,
        title_items=title_items,
        prefer_one_based=prefer_one_based,
        image_uri_cache=image_uri_cache,
        inserted_fig_indices=inserted_fig_indices,
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


def _renumber_report_figure_blocks(final_html: str, title_items: list[str]) -> str:
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
        caption_tag.string = _build_figure_caption(display_number, fig_index, title_items)

    return str(soup)


def _inject_visualizations_into_html(final_html: str) -> str:
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
    prefer_one_based = bool(match_numbers) and 0 not in match_numbers and max(match_numbers) <= len(fig_desc_list)
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
    )
    injected_html = str(soup)
    injected_html = _normalize_report_figure_layout(injected_html)
    injected_html = _remove_duplicate_figure_titles(injected_html)
    injected_html = _renumber_report_figure_blocks(injected_html, title_items)
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


def _load_modeling_table_payload() -> dict[str, str]:
    summary_4 = maybe_json_loads(
        st.session_state.get("summary_4") or st.session_state.get("modeling_summary_4", {})
    )
    if not isinstance(summary_4, dict):
        return {"title": "", "caption": "", "table_html": "", "table_markdown": ""}

    title = stringify_string(summary_4.get("table_title"))
    table_html = stringify_string(summary_4.get("table_html"))
    table_markdown = stringify_string(summary_4.get("table_markdown"))
    caption = f"表1 {title}" if title else ""
    return {
        "title": title,
        "caption": caption,
        "table_html": table_html,
        "table_markdown": table_markdown,
    }


def _inject_modeling_table_into_html(final_html: str) -> str:
    if not final_html:
        return final_html

    payload = _load_modeling_table_payload()
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
    block.append(table_tag)

    insert_after.insert_after(block)
    return str(soup)


def _finalize_report_html(final_html: str, report_title: str) -> str:
    if not final_html:
        return final_html

    final_html = _normalize_markdown_headings_in_html(final_html)
    final_html = _inject_report_title_into_html(final_html, report_title)
    final_html = _inject_visualizations_into_html(final_html)
    final_html = _inject_modeling_table_into_html(final_html)
    final_html = _inject_report_base_style(final_html)
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
            parsed_segments = _split_markdown_heading_lines(line)
            if parsed_segments:
                for line_kind, line_text in parsed_segments:
                    if line_kind == "heading":
                        heading_tag = soup.new_tag("h1")
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

        if not replacement_nodes or not any(node.name == "h1" for node in replacement_nodes):
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


def _prepare_downloadable_reports(report_agent, generation_token: str | None = None) -> dict[str, Any]:
    workflow_result = report_agent.load_report_workflow_result()
    html_content = report_agent.load_html()
    markdown_content = report_agent.load_markdown()
    word_bytes = report_agent.load_word()
    pdf_bytes = report_agent.load_pdf()
    pdf_export_method = report_agent.load_pdf_export_method()

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
            html_content = _finalize_report_html(html_content, report_title)
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

    def _count_docx_media_files(docx_content: bytes | None) -> int:
        if not docx_content:
            return 0
        try:
            with zipfile.ZipFile(io.BytesIO(docx_content)) as archive:
                return sum(1 for name in archive.namelist() if name.startswith("word/media/"))
        except Exception:
            return 0

    if html_content and word_bytes is None:
        try:
            word_bytes = build_docx_from_html(html_content)
        except Exception as exc:
            print("[REPORT][WORD] build_docx_from_html failed:", repr(exc))
            word_bytes = None

    html_image_count = len(re.findall(r"<img\b", html_content or "", flags=re.IGNORECASE))
    docx_media_count = _count_docx_media_files(word_bytes)
    if report_agent.load_word() is None and word_bytes is not None and html_image_count > 0 and docx_media_count < html_image_count:
        print(
            f"[REPORT][WORD] docx embedded media count mismatch: html_images={html_image_count}, docx_media={docx_media_count}; retry with markdown fallback"
        )
        word_bytes = None

    if word_bytes is None:
        markdown_source = html_to_markdown(html_content) if html_content else markdown_content
        if not markdown_source:
            markdown_source = markdown_content
        if markdown_source:
            word_bytes = build_docx_from_markdown(markdown_source)

    if word_bytes is None and workflow_result:
        word_bytes = extract_report_word_bytes(workflow_result)

    if word_bytes is not None and can_save_prepared_output():
        print(
            f"[REPORT][WORD] final media count = {_count_docx_media_files(word_bytes)}, html_image_count = {html_image_count}"
        )
        report_agent.save_word(word_bytes)

    if pdf_bytes is not None and can_save_prepared_output():
        report_agent.save_pdf(pdf_bytes)
        report_agent.save_pdf_export_method(pdf_export_method)

    return {
        "word": word_bytes,
        "html": html_content,
        "markdown": markdown_content,
        "pdf": pdf_bytes,
        "pdf_export_method": pdf_export_method,
    }


def _refresh_markdown_from_html(report_agent, html_content: str) -> str:
    markdown_content = html_to_markdown(html_content) if html_content else ""
    if markdown_content:
        report_agent.save_markdown(markdown_content)
    return markdown_content


def _ensure_pdf_download_ready(report_agent, downloadable_reports: dict[str, Any]) -> dict[str, Any]:
    if downloadable_reports.get("pdf") is not None:
        return downloadable_reports

    word_bytes = downloadable_reports.get("word")
    html_content = downloadable_reports.get("html")
    if not word_bytes and not html_content:
        return downloadable_reports

    try:
        with st.spinner("正在生成 PDF 报告..."):
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
        lambda match: f"![{match.group(1) or '图表'}](embedded-image)",
        preview,
        flags=re.IGNORECASE,
    )
    preview = re.sub(
        r"(?:!\[[^\]]*\]\(embedded-image\)\s*){2,}",
        "[图表已嵌入，预览中省略重复图片占位]\n\n",
        preview,
        flags=re.IGNORECASE,
    )
    preview = re.sub(r"\n{3,}", "\n\n", preview).strip()
    if not preview:
        preview = "[正文预览为空，下载的 Markdown 文件中仍包含完整图表内容]"
    if len(preview) > 60000:
        preview = preview[:60000].rstrip() + "\n\n...[预览已截断，下载文件中仍保留完整内容]"
    return preview


def _render_report_preview(html_content: str, markdown_content: str) -> None:
    if (html_content or "").strip():
        components.html(html_content, height=720, scrolling=True)
        return

    if (markdown_content or "").strip():
        st.markdown(_build_markdown_preview(markdown_content))


def _ensure_pdf_download_ready(report_agent, downloadable_reports: dict[str, Any]) -> dict[str, Any]:
    if downloadable_reports.get("pdf") is not None:
        return downloadable_reports

    word_bytes = downloadable_reports.get("word")
    html_content = downloadable_reports.get("html")
    if not word_bytes and not html_content:
        return downloadable_reports

    try:
        with st.spinner("正在生成 PDF 报告..."):
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
        lambda match: f"![{match.group(1) or '图表'}](embedded-image)",
        preview,
        flags=re.IGNORECASE,
    )
    preview = re.sub(
        r"(?:!\[[^\]]*\]\(embedded-image\)\s*){2,}",
        "[图表已嵌入，预览中省略重复图片占位]\n\n",
        preview,
        flags=re.IGNORECASE,
    )
    preview = re.sub(r"\n{3,}", "\n\n", preview).strip()
    if not preview:
        preview = "[正文预览为空，下载的 Markdown 文件中仍包含完整图表内容]"
    if len(preview) > 60000:
        preview = preview[:60000].rstrip() + "\n\n...[预览已截断，下载文件中仍保留完整内容]"
    return preview


def _render_report_preview(html_content: str, markdown_content: str) -> None:
    if (html_content or "").strip():
        components.html(html_content, height=720, scrolling=True)
        return

    if (markdown_content or "").strip():
        st.markdown(_build_markdown_preview(markdown_content))


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

    html_content = stringify_string(preview.get("html")).strip()
    markdown_content = stringify_string(preview.get("markdown")).strip()
    if html_content:
        components.html(_insert_dim_preview_style(html_content), height=720, scrolling=True)
        return

    if markdown_content:
        preview_html = markdown_to_html(_build_markdown_preview(markdown_content), title="")
        components.html(_insert_dim_preview_style(preview_html), height=720, scrolling=True)


def _clear_generated_report_files(report_agent) -> None:
    report_agent.save_word(None)
    report_agent.save_pdf(None)
    report_agent.save_pdf_export_method(None)
    report_agent.save_html(None)
    report_agent.save_markdown(None)
    st.session_state.pop("report_final_html", None)


def _clear_report_workflow_outputs(report_agent) -> None:
    _clear_generated_report_files(report_agent)
    _clear_pending_report_preview()
    report_agent.save_report_workflow_result(None)
    report_agent.save_report(None)
    report_agent.save_report_content(None)


def _begin_report_generation(report_agent) -> str:
    generation_token = str(time.time_ns())
    st.session_state[REPORT_GENERATION_TOKEN_KEY] = generation_token
    st.session_state[REPORT_GENERATION_RUNNING_KEY] = True
    _capture_pending_report_preview(report_agent)
    _clear_active_report_outputs(report_agent)
    return generation_token


def _is_current_report_generation(generation_token: str | None) -> bool:
    return bool(generation_token) and st.session_state.get(REPORT_GENERATION_TOKEN_KEY) == generation_token


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


def _has_report_prerequisites() -> bool:
    return bool(
        st.session_state.get("summary_1")
        and st.session_state.get("summary_2")
        and st.session_state.get("summary_3")
        and st.session_state.get("summary_4")
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


def _finish_report_generation(generation_token: str | None) -> None:
    if _is_current_report_generation(generation_token):
        st.session_state[REPORT_GENERATION_RUNNING_KEY] = False


def _complete_auto_report(report_agent) -> None:
    report_agent.finish_auto()
    st.session_state.auto_mode = False

    planner = st.session_state.get("planner_agent")
    if planner is not None:
        planner.finish_report_auto()


def _has_report_prerequisites() -> bool:
    return bool(
        st.session_state.get("summary_1")
        and st.session_state.get("summary_2")
        and st.session_state.get("summary_3")
        and st.session_state.get("summary_4")
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

    selected_full_content, selected_source = _resolve_selected_full_content(
        visual_summary=visual_summary,
        allow_report_cache=False,
    )
    _log_selected_full_content("toc", selected_full_content, selected_source)

    return {
        "load_summary": load_summary,
        "preproc_summary": preproc_summary,
        "visual_summary": visual_summary,
        "coding_summary": coding_summary,
        "selected_full_conten": selected_full_content,
        "load_abstract": stringify_string(_resolve_loading_field(load_agent, "abstract_1", "")),
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


def _build_word_report_inputs(report_agent) -> dict[str, Any]:
    current_coding_abstract = stringify_string(
        st.session_state.get("abstract_4") or st.session_state.get("modeling_abstract_4")
    )
    if not current_coding_abstract:
        current_coding_abstract = stringify_string(st.session_state.get("report_coding_abstract"))

    visual_summary = maybe_json_loads(st.session_state.get("summary_3", {}))
    current_selected_full_content, selected_source = _resolve_selected_full_content(
        visual_summary=visual_summary if isinstance(visual_summary, dict) else {},
        allow_report_cache=True,
    )
    _log_selected_full_content("word", current_selected_full_content, selected_source)

    return {
        "toc_text": _normalize_multiline_text(report_agent.load_outline()),
        "title": "",
        "selected_full_conten": current_selected_full_content,
        "preference_selected": stringify_string(st.session_state.get("report_preference_selected")),
        "add_preference": stringify_string(st.session_state.get("report_add_preference")),
        "load_abstract": stringify_string(st.session_state.get("report_load_abstract")),
        "preproc_abstract": stringify_string(st.session_state.get("report_preproc_abstract")),
        "visual_abstract": stringify_string(st.session_state.get("report_visual_abstract")),
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
        return retriever.retrieve_and_format(
            f"报告撰写 业务背景 {inputs.get('add_preference', '')}",
            top_k=3,
        )
    except Exception as exc:
        print("[REPORT][JOB] reference retrieval failed:", repr(exc))
        return ""


def _build_report_worker_payload(report_agent) -> dict[str, Any]:
    inputs = _build_word_report_inputs(report_agent)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    inputs.setdefault("preference_select", st.session_state.get("preference_selected") or "")
    inputs["ref_context"] = _get_report_worker_ref_context(inputs)

    return {
        "inputs": inputs,
        "llm_config": {
            "api_key": st.session_state.get("llm_api_key") or os.getenv("OPENAI_API_KEY", ""),
            "base_url": st.session_state.get("llm_base_url") or os.getenv("OPENAI_BASE_URL", ""),
            "model": st.session_state.get("llm_model") or os.getenv("OPENAI_MODEL", ""),
        },
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


def _start_report_generation_process(report_agent, action: str) -> bool:
    _terminate_report_generation_process()
    generation_token = _begin_report_generation(report_agent)

    work_dir = tempfile.mkdtemp(prefix="autostat_report_")
    input_path = os.path.join(work_dir, "input.json")
    output_path = os.path.join(work_dir, "output.json")
    payload = _build_report_worker_payload(report_agent)

    try:
        with open(input_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

        env = os.environ.copy()
        llm_config = payload.get("llm_config") if isinstance(payload.get("llm_config"), dict) else {}
        if llm_config.get("api_key"):
            env["OPENAI_API_KEY"] = str(llm_config["api_key"])
        if llm_config.get("base_url"):
            env["OPENAI_BASE_URL"] = str(llm_config["base_url"])
        if llm_config.get("model"):
            env["OPENAI_MODEL"] = str(llm_config["model"])

        process = subprocess.Popen(
            [sys.executable, "-m", "workflows.reporting_partly_worker", input_path, output_path],
            cwd=str(_report_repo_root()),
            env=env,
        )
    except Exception as exc:
        _cleanup_report_job_files({"work_dir": work_dir})
        _finish_report_generation(generation_token)
        st.error(f"报告生成进程启动失败：{exc}")
        return False

    job = {
        "token": generation_token,
        "action": action,
        "work_dir": work_dir,
        "input_path": input_path,
        "output_path": output_path,
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
        return {"ok": False, "error": f"报告生成结果读取失败：{exc}"}


def _save_formatted_report_result(
    report_agent,
    action: str,
    workflow_result: dict[str, Any],
    generation_token: str | None,
) -> bool:
    status_placeholder = st.empty()
    raw_content = extract_report_html(workflow_result)
    if not raw_content:
        status_placeholder.empty()
        st.error("Word 报告工作流未返回 `final_html`。")
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

    html_content = _finalize_report_html(html_content, report_title)
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

    status_placeholder.info("正在生成报告文件，请稍后。")
    downloadable_reports = _prepare_downloadable_reports(report_agent, generation_token=generation_token)

    if _is_report_generation_cancelled(generation_token):
        return False

    if action == "Word" and downloadable_reports.get("word") is None:
        status_placeholder.empty()
        st.error("Word 报告内容已生成，但 Word 文件转换失败，请重试或切换为 HTML 报告。")
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
        st.error("报告生成进程状态丢失，请重新生成。")
        return "failed"

    return_code = process.poll()
    if return_code is None:
        st.info(f"正在生成{action}报告，请耐心等待")
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
        st.error(f"报告生成进程已退出（code={return_code}），但没有返回可用结果。")
        return "failed"

    if not worker_payload.get("ok"):
        _finish_report_generation(token)
        error_message = worker_payload.get("error") or "未知错误"
        st.error(f"报告生成失败：{error_message}")
        traceback_text = stringify_string(worker_payload.get("traceback"))
        if traceback_text:
            print("[REPORT][JOB] worker traceback:\n", traceback_text)
        return "failed"

    workflow_result = _merge_report_workflow_results([worker_payload.get("result")])
    if workflow_result is None:
        _finish_report_generation(token)
        st.error("Word 报告生成失败，未解析到有效输出，请重新生成。")
        return "failed"

    success = _save_formatted_report_result(report_agent, action, workflow_result, token)
    _finish_report_generation(token)
    if success:
        st.success(f"{action} 报告已生成，已在下方展示。")
        return "complete"
    return "failed"


def _is_report_generation_job_running() -> bool:
    job = st.session_state.get(REPORT_GENERATION_JOB_KEY)
    process = st.session_state.get(REPORT_GENERATION_PROCESS_KEY)
    poll = getattr(process, "poll", None)
    return isinstance(job, dict) and callable(poll) and process.poll() is None


def call_coze_workflow_report_stream(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """本地化版本：调用本地 Reporting_toc workflow。"""
    from utils.local_workflow_bridge import call_reporting_toc_bridge


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
        return retriever.retrieve_and_format(
            f"报告撰写 业务背景 {inputs.get('add_preference', '')}",
            top_k=3,
        )
    except Exception as exc:
        print("[REPORT][JOB] reference retrieval failed:", repr(exc))
        return ""


def _build_report_worker_payload(report_agent) -> dict[str, Any]:
    inputs = _build_word_report_inputs(report_agent)
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
    st.error("目录工作流已完成，但未解析到有效输出。")
    return None


def call_coze_workflow_word_stream(
    inputs: dict[str, Any],
    status_placeholder: Any | None = None,
    clear_on_success: bool = True,
    generation_token: str | None = None,
) -> dict[str, Any] | None:
    """本地化版本：调用本地 Reporting_partly workflow。"""
    from utils.local_workflow_bridge import call_reporting_partly_bridge

    if status_placeholder is None:
        status_placeholder = st.empty()
    status_placeholder.info("正在生成报告，请稍后。")

    inputs = dict(inputs)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    # report workflow 字段名是 preference_select
    inputs.setdefault("preference_select", st.session_state.get("preference_selected") or "")
    inputs["ref_context"] = _get_report_worker_ref_context(inputs)

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
    st.error("Word 报告生成失败，未解析到有效输出，请点击生成报告按钮重新生成。")
    return None


def _generate_formatted_report(report_agent, action: str) -> None:
    if _start_report_generation_process(report_agent, action):
        st.info(f"正在生成{action}报告，请耐心等待")
    return

    generation_token = _begin_report_generation(report_agent)
    status_placeholder = st.empty()
    workflow_result = call_coze_workflow_word_stream(
        _build_word_report_inputs(report_agent),
        status_placeholder=status_placeholder,
        clear_on_success=False,
        generation_token=generation_token,
    )
    if not workflow_result:
        _finish_report_generation(generation_token)
        return

    if _is_report_generation_cancelled(generation_token):
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


def _start_report_generation_process(report_agent, action: str) -> bool:
    _terminate_report_generation_process()
    generation_token = _begin_report_generation(report_agent)

    work_dir = tempfile.mkdtemp(prefix="autostat_report_")
    input_path = os.path.join(work_dir, "input.json")
    output_path = os.path.join(work_dir, "output.json")
    payload = _build_report_worker_payload(report_agent)

    try:
        with open(input_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

        env = os.environ.copy()
        llm_config = payload.get("llm_config") if isinstance(payload.get("llm_config"), dict) else {}
        if llm_config.get("api_key"):
            env["OPENAI_API_KEY"] = str(llm_config["api_key"])
        if llm_config.get("base_url"):
            env["OPENAI_BASE_URL"] = str(llm_config["base_url"])
        if llm_config.get("model"):
            env["OPENAI_MODEL"] = str(llm_config["model"])

        process = subprocess.Popen(
            [sys.executable, "-m", "workflows.reporting_partly_worker", input_path, output_path],
            cwd=str(_report_repo_root()),
            env=env,
        )
    except Exception as exc:
        _cleanup_report_job_files({"work_dir": work_dir})
        _finish_report_generation(generation_token)
        st.error(f"报告生成进程启动失败：{exc}")
        return False

    job = {
        "token": generation_token,
        "action": action,
        "work_dir": work_dir,
        "input_path": input_path,
        "output_path": output_path,
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
        return {"ok": False, "error": f"报告生成结果读取失败：{exc}"}


def _save_formatted_report_result(
    report_agent,
    action: str,
    workflow_result: dict[str, Any],
    generation_token: str | None,
) -> bool:
    status_placeholder = st.empty()
    raw_content = extract_report_html(workflow_result)
    if not raw_content:
        status_placeholder.empty()
        st.error("Word 报告工作流未返回 `final_html`。")
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

    html_content = _finalize_report_html(html_content, report_title)
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

    status_placeholder.info("正在生成报告，请稍后。")
    downloadable_reports = _prepare_downloadable_reports(report_agent, generation_token=generation_token)

    if _is_report_generation_cancelled(generation_token):
        return

    if action == "Word" and downloadable_reports.get("word") is None:
        status_placeholder.empty()
        st.error("Word 报告内容已生成，但 Word 文件转换失败，请重试或切换为 HTML 报告。")
        _finish_report_generation(generation_token)
        return

    st.session_state.report_final_html = html_content
    _clear_pending_report_preview()
    status_placeholder.empty()
    st.success(f"{action} 报告已生成，已在下侧展示。")
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
    outline_length = sac.segmented(
        items=[
            sac.SegmentedItem(label="简要"),
            sac.SegmentedItem(label="标准"),
            sac.SegmentedItem(label="详细"),
        ],
        label="详细程度",
        index=1,
        align="center",
        size="sm",
        radius="sm",
        use_container_width=True,
    )
    report_agent.save_outline_length(outline_length)

    report_format = sac.chip(
        items=[
            sac.ChipItem(label="Word", icon=sac.BsIcon(name="file-earmark-word", size=16)),
            sac.ChipItem(label="HTML", icon=sac.BsIcon(name="filetype-html", size=16)),
            sac.ChipItem(label="PDF", icon=sac.BsIcon(name="file-earmark-pdf", size=16)),
        ],
        label="选择报告生成格式",
        index=[0],
        align="start",
        radius="md",
        multiple=False,
    )
    if auto:
        report_format = "Word"
    report_agent.save_report_format(_normalize_report_format(report_format))

    user_input = st.text_input("报告生成要求", "默认")
    report_agent.save_user_input(user_input)
    visualization_agent = st.session_state.get("visualization_agent")

    if not auto and not _has_visualization_recommendation(visualization_agent):
        st.warning("请先完成可视化推荐部分。")

    not_generated = not _has_generated_outline(report_agent)
    if st.button("生成目录") or (auto and not_generated):
        if not auto and not _has_visualization_recommendation(visualization_agent):
            st.warning("请先完成可视化推荐部分。")
            return

        _clear_report_workflow_outputs(report_agent)
        report_agent.save_outline([])

        inputs = _build_report_inputs(load_agent, report_agent)
        workflow_result = call_coze_workflow_report_stream(inputs)

        if not workflow_result:
            return

        _save_report_workflow_outputs(report_agent, workflow_result)

        toc_text = _extract_toc_text_from_result(workflow_result)
        if not toc_text:
            st.error("报告工作流未返回 `toc_text`。")
            return

        report_agent.save_outline(toc_text)
        if auto:
            st.rerun()
        st.success("目录已生成，已在右侧显示文本。")


def report_outline(report_agent) -> None:
    st.subheader("目录结构预览与编辑")

    outline_value = report_agent.load_outline()
    if isinstance(outline_value, str):
        default_toc = stringify_string(outline_value).replace("\\r\\n", "\n").replace("\\n", "\n")
    else:
        default_toc = "\n".join(normalize_toc_list(outline_value))
    toc_text = st.text_area(
        "您可以在此处编辑目录结构，每行一个目录项",
        value=default_toc,
        height=260,
        placeholder="# 数据分析报告\n## 1. 数据导入",
    )
    report_agent.save_outline(toc_text)


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

    generate_clicked = st.button(f"生成 {action} 报告")
    if generate_clicked or (auto and not_generate):
        if st.session_state.get("report_selected_full_conten") is None and st.session_state.get("full") is None:
            st.warning("请先点击“生成目录”获取新 workflow 输出。")
            return

        if not auto and not _ensure_visualization_ready_for_report(visualization_agent):
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

    downloadable_reports = _prepare_downloadable_reports(report_agent)
    if action == "PDF":
        downloadable_reports = _ensure_pdf_download_ready(report_agent, downloadable_reports)

    html_content = (downloadable_reports.get("html") or "").strip()
    markdown_content = (downloadable_reports.get("markdown") or "").strip()

    if action == "Word":
        st.download_button(
            label="下载 Word 报告",
            data=downloadable_reports["word"] or b"",
            file_name="report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            disabled=downloadable_reports["word"] is None,
        )
        _render_report_preview(html_content, markdown_content)
        return

    if action == "HTML":
        st.download_button(
            label="下载 HTML 报告",
            data=html_content.encode("utf-8") if html_content else b"",
            file_name="report.html",
            mime="text/html",
            disabled=not bool(html_content),
        )
        _render_report_preview(html_content, markdown_content)
        return

    if action == "PDF":
        st.download_button(
            label="下载 PDF 报告",
            data=downloadable_reports["pdf"] or b"",
            file_name="report.pdf",
            mime="application/pdf",
            disabled=downloadable_reports["pdf"] is None,
        )

        _render_report_preview(html_content, markdown_content)
        return

    _render_report_preview(html_content, markdown_content)
            

if __name__ == "__main__":
    st.title("报告生成")
    st.markdown("---")

    load_agent = st.session_state.data_loading_agent
    preproc_agent = st.session_state.data_preprocess_agent
    planner = st.session_state.planner_agent
    auto = bool(st.session_state.auto_mode and planner.report_auto)

    if st.session_state.auto_mode and not _has_report_prerequisites():
        st.warning("自动模式需要在前序步骤都生成结果后，才会进入报告生成。")
        st.stop()

    processed_df = preproc_agent.load_processed_df()
    df = processed_df if processed_df is not None else load_agent.load_df()

    if df is None:
        st.warning("请先在数据导入页面加载数据。")
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
    with columns[0].expander("报告设置", expanded=True):
        report_basic_info(load_agent, report_agent, auto)

    with columns[1].expander("报告大纲", expanded=True):
        report_outline(report_agent)
        report_save(report_agent, auto)
        report_execution(report_agent)
