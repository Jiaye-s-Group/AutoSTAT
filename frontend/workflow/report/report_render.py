"""
report_render

"""

import json
import time
import re
import html
import base64
import hashlib
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import streamlit as st
import streamlit_antd_components as sac
from bs4 import BeautifulSoup, NavigableString, Tag
from workflow.report.report_constants import (
    FIG_PLACEHOLDER_CAPTURE_PATTERN,
    FIG_PLACEHOLDER_PATTERN,
    REPORT_EXPORT_IMAGE_PERCENT,
    REPORT_FIGURE_DATA_URI_CACHE_KEY,
    REPORT_GENERATION_JOB_KEY,
    REPORT_IMAGE_EXPORT_TIMEOUT_SECONDS,
)
from workflow.report.report_content_utils import (
    _split_markdown_heading_lines,
    extract_report_html,
    html_to_markdown,
    markdown_to_html,
    maybe_json_loads,
    normalize_figure_placeholders,
    normalize_trailing_punctuation_before_figure_placeholder,
    normalize_toc_list,
    remove_figure_placeholders,
    stringify_string,
)
from workflow.report.report_export import (
    ensure_pdf_download_ready as _ensure_pdf_download_ready,
    extract_report_title as _extract_report_title,
    looks_like_html as _looks_like_html,
    prepare_downloadable_reports as _prepare_downloadable_reports_base,
)
from workflow.report.report_generation import (
    is_report_generation_job_running as _is_report_generation_job_running,
    poll_report_generation_job as _poll_report_generation_job_base,
    start_report_generation_process as _start_report_generation_process,
)
from workflow.report.report_inputs import (
    build_report_inputs as _build_report_inputs,
    build_word_report_inputs as _build_word_report_inputs,
    ensure_visualization_ready_for_report as _ensure_visualization_ready_for_report,
    extract_toc_text_from_result as _extract_toc_text_from_result,
    has_generated_outline as _has_generated_outline,
    has_generated_word_report as _has_generated_word_report,
    has_report_prerequisites as _has_report_prerequisites,
    has_visualization_recommendation as _has_visualization_recommendation,
    normalize_report_format as _normalize_report_format,
    normalize_visualization_titles as _normalize_visualization_titles,
    resolve_visualization_dataframe_for_report as _resolve_visualization_dataframe_for_report,
)
from workflow.report.report_preview import (
    clear_pending_report_preview as _clear_pending_report_preview,
    render_pending_report_preview as _render_pending_report_preview,
    render_report_preview as _render_report_preview,
)
from workflow.report.report_state import (
    begin_report_generation as _begin_report_generation,
    clear_generated_report_files as _clear_generated_report_files,
    clear_report_workflow_outputs as _clear_report_workflow_outputs,
    complete_auto_report as _complete_auto_report,
    finish_report_generation as _finish_report_generation,
    is_current_report_generation as _is_current_report_generation,
    is_report_generation_cancelled as _is_report_generation_cancelled,
    save_report_workflow_outputs as _save_report_workflow_outputs,
)


def _merge_report_workflow_results(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    merged_result: dict[str, Any] = {}

    for result in results:
        if isinstance(result, dict):
            merged_result.update(result)

    return merged_result or None


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


def _prepare_downloadable_reports(report_agent, generation_token: str | None = None) -> dict[str, Any]:
    return _prepare_downloadable_reports_base(
        report_agent,
        generation_token=generation_token,
        finalize_report_html=_finalize_report_html,
    )


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
    return _poll_report_generation_job_base(
        report_agent,
        action,
        merge_report_workflow_results=_merge_report_workflow_results,
        save_formatted_report_result=_save_formatted_report_result,
    )


def call_report_workflow_stream(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Call the local report-outline workflow."""
    from utils.local_workflow_bridge import call_reporting_toc_bridge

    status_placeholder = st.empty()
    status_placeholder.info("正在生成目录，请稍后。")

    inputs = dict(inputs)
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


def call_word_report_workflow_stream(
    inputs: dict[str, Any],
    status_placeholder: Any | None = None,
    clear_on_success: bool = True,
    generation_token: str | None = None,
) -> dict[str, Any] | None:
    """Call the local report-writing workflow."""
    from utils.local_workflow_bridge import call_reporting_partly_bridge

    if status_placeholder is None:
        status_placeholder = st.empty()
    status_placeholder.info("正在生成报告，请稍后。")

    inputs = dict(inputs)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    # Legacy report workflow field name.
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
    st.error("Word 报告生成失败，未解析到有效输出，请点击生成报告按钮重新生成。")
    return None


def _generate_formatted_report(report_agent, action: str) -> None:
    if _start_report_generation_process(report_agent, action):
        st.info(f"正在生成{action}报告，请耐心等待")
    return

    generation_token = _begin_report_generation(report_agent)
    status_placeholder = st.empty()
    workflow_result = call_word_report_workflow_stream(
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
        workflow_result = call_report_workflow_stream(inputs)

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
