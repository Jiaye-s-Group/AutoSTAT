"""
report_render

"""

import time
import re
import html
import base64
import io
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
import streamlit_antd_components as sac
from bs4 import BeautifulSoup, NavigableString, Tag
try:  # cozepy 不再需要，但保留 import 以兼容可能残留的旧代码引用
    from cozepy import Coze, TokenAuth, WorkflowEventType  # type: ignore
except ImportError:  # 本地化后可能没装 cozepy
    Coze = None  # type: ignore
    TokenAuth = None  # type: ignore
    WorkflowEventType = None  # type: ignore

from utils.coze_runtime import resolve_coze_runtime
from workflow.visualization.viz_coding import (
    execute_visualization_code_once,
    generate_visualization_code_once,
)
from workflow.report.report_content_utils import (
    _split_markdown_heading_lines,
    build_markdown_preview_from_html,
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
    normalize_trailing_punctuation_before_figure_placeholder,
    normalize_toc_list,
    stringify_string,
)
from workflow.report.report_utils import convert_report_to_pdf_bytes

COZE_SPACE_ID = "7594748927577554949"
WORKFLOW_ID = "7619618199978508341"
WORD_REPORT_WORKFLOW_ID = "7619618317418446901"
BOT_ID = "7595403958269575173"
MAX_POLL_SECONDS = 1800
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
FIG_PLACEHOLDER_PATTERN = r"(?<![A-Za-z0-9_])[\[\uFF3B\u3010]?\s*FIG\s*:?\s*(?:\d+)\s*[\]\uFF3D\u3011]?(?![A-Za-z0-9_])"
FIG_PLACEHOLDER_CAPTURE_PATTERN = r"(?<![A-Za-z0-9_])[\[\uFF3B\u3010]?\s*FIG\s*:?\s*(\d+)\s*[\]\uFF3D\u3011]?(?![A-Za-z0-9_])"
REPORT_EXPORT_IMAGE_SCALE = 0.6
REPORT_EXPORT_IMAGE_PERCENT = f"{REPORT_EXPORT_IMAGE_SCALE * 100:.0f}%"
def _resolve_coze_base_url(coze_url: str) -> str:
    if "api.coze.cn" in coze_url:
        return "https://api.coze.cn"
    return "https://api.coze.com"


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


def _normalize_report_workflow_result(result: Any) -> dict[str, Any] | None:
    parsed_result = maybe_json_loads(result)

    if isinstance(parsed_result, dict):
        return parsed_result

    if isinstance(parsed_result, str):
        reparsed = maybe_json_loads(parsed_result)
        if isinstance(reparsed, dict):
            return reparsed

    return None


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


def _looks_like_markdown(text: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(r"(?m)^\s*#{1,6}\s+", text)
        or re.search(r"(?m)^\s*[-*]\s+", text)
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
    try:
        image_bytes = pio.to_image(fig, format="png", width=1400, height=900, scale=2)
        return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
    except Exception as exc:
        print("[REPORT][FIG] pio.to_image failed:", repr(exc))
        return None


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

        new_nodes: list[Tag] = []
        text_fragments: list[str] = []

        def flush_text_fragments() -> None:
            text = re.sub(r"\s+", " ", "".join(text_fragments)).strip()
            text_fragments.clear()
            if not text:
                return

            paragraph_tag = soup.new_tag("p")
            for attr_name, attr_value in paragraph.attrs.items():
                paragraph_tag[attr_name] = attr_value
            paragraph_tag.string = text
            new_nodes.append(paragraph_tag)

        def walk(node: Tag | NavigableString) -> None:
            if isinstance(node, NavigableString):
                text_fragments.append(str(node))
                return

            if not isinstance(node, Tag):
                return

            if is_figure_block(node):
                flush_text_fragments()
                new_nodes.append(node.extract())
                return

            if node.name in {"script", "style", "noscript"}:
                return

            if node.name == "br":
                text_fragments.append("\n")
                return

            for child in list(node.children):
                walk(child)

        for child in list(paragraph.children):
            walk(child)

        flush_text_fragments()

        if not new_nodes:
            paragraph.decompose()
        else:
            first_node = new_nodes[0]
            paragraph.replace_with(first_node)
            current_node = first_node
            for node in new_nodes[1:]:
                current_node.insert_after(node)
                current_node = node
        changed = True

    return str(soup) if changed else final_html


def _resolve_placeholder_figure_index(raw_index: int, fig_count: int, prefer_one_based: bool) -> int | None:
    if fig_count <= 0:
        return None

    candidates = [raw_index - 1, raw_index] if prefer_one_based else [raw_index, raw_index - 1]
    seen: set[int] = set()

    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if 0 <= candidate < fig_count:
            return candidate

    return None


def _build_figure_block_tag(
    soup: BeautifulSoup,
    fig_desc_list: list[Any],
    title_items: list[str],
    fig_index: int,
    display_number: int,
    image_uri_cache: dict[int, str],
) -> Tag | None:
    if fig_index < 0 or fig_index >= len(fig_desc_list):
        print(f"[REPORT][FIG] fig index out of range: {fig_index}")
        return None

    fig_item = fig_desc_list[fig_index]
    fig = _normalize_visual_figure(fig_item.get("fig") if isinstance(fig_item, dict) else fig_item)
    if fig is None:
        print(f"[REPORT][FIG] fig at index {fig_index} cannot be normalized")
        return None

    image_uri = image_uri_cache.get(fig_index)
    if not image_uri:
        image_uri = _figure_to_data_uri(fig)
        if not image_uri:
            print(f"[REPORT][FIG] fig at index {fig_index} cannot convert to image")
            return None
        image_uri_cache[fig_index] = image_uri

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


def _replace_remaining_placeholders_in_soup(
    soup: BeautifulSoup,
    fig_desc_list: list[Any],
    title_items: list[str],
    prefer_one_based: bool,
    image_uri_cache: dict[int, str],
) -> int:
    inserted_count = 0
    text_nodes = list(soup.find_all(string=True))

    for text_node in text_nodes:
        parent = text_node.parent
        if not isinstance(parent, Tag):
            continue
        if parent.name in {"script", "style", "noscript"}:
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

    for display_number, figure_block in enumerate(soup.find_all("div", class_="report-figure-block"), start=1):
        raw_fig_index = figure_block.get("data-fig-index", "")
        try:
            fig_index = int(raw_fig_index)
        except Exception:
            fig_index = -1

        figure_block["data-report-figure-number"] = str(display_number)

        image_tag = figure_block.find("img")
        if image_tag is not None:
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
        return re.sub(FIG_PLACEHOLDER_PATTERN, "", final_html, flags=re.IGNORECASE)

    final_html = normalize_trailing_punctuation_before_figure_placeholder(final_html)

    matches = re.findall(FIG_PLACEHOLDER_CAPTURE_PATTERN, final_html, flags=re.IGNORECASE)
    print("[REPORT][FIG] placeholders found in html =", matches)

    match_numbers = [int(item) for item in matches if str(item).isdigit()]
    prefer_one_based = bool(match_numbers) and 0 not in match_numbers and max(match_numbers) <= len(fig_desc_list)
    print("[REPORT][FIG] placeholder numbering mode =", "1-based" if prefer_one_based else "0-based")

    image_uri_cache: dict[int, str] = {}
    soup = BeautifulSoup(final_html, "html.parser")
    inserted_figure_count = _replace_remaining_placeholders_in_soup(
        soup=soup,
        fig_desc_list=fig_desc_list,
        title_items=title_items,
        prefer_one_based=prefer_one_based,
        image_uri_cache=image_uri_cache,
    )
    injected_html = str(soup)
    injected_html = _normalize_report_figure_layout(injected_html)
    injected_html = _renumber_report_figure_blocks(injected_html, title_items)
    injected_html = _remove_duplicate_figure_titles(injected_html)
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

    modeling_heading = None
    heading_tags = soup.find_all(re.compile(r"^h[1-6]$"))
    for heading in heading_tags:
        if _looks_like_chapter4_heading(heading.get_text(" ", strip=True)):
            modeling_heading = heading
            break

    if modeling_heading is None:
        candidate_tags = soup.find_all(["p", "div", "section", "article"])
        for tag in candidate_tags:
            if tag.find(["img", "table", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6", "p"]):
                continue
            if _looks_like_chapter4_heading(tag.get_text(" ", strip=True)):
                modeling_heading = tag
                break

    if modeling_heading is None:
        return final_html

    insert_after: Tag | None = modeling_heading
    if modeling_heading is not None:
        for sibling in modeling_heading.next_siblings:
            if not isinstance(sibling, Tag):
                continue
            if re.match(r"^h[1-6]$", sibling.name or ""):
                break
            if sibling.get_text(" ", strip=True):
                insert_after = sibling
                break

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


def _prepare_downloadable_reports(report_agent) -> dict[str, Any]:
    workflow_result = report_agent.load_report_workflow_result()
    html_content = report_agent.load_html()
    markdown_content = report_agent.load_markdown()
    word_bytes = report_agent.load_word()
    pdf_bytes = report_agent.load_pdf()
    pdf_export_method = report_agent.load_pdf_export_method()

    if workflow_result:
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

            report_agent.save_html(html_content)
            if markdown_content:
                report_agent.save_markdown(markdown_content)

    if not markdown_content:
        if html_content:
            markdown_content = html_to_markdown(html_content)
        elif workflow_result:
            markdown_content = extract_report_markdown(workflow_result) or extract_report_text(workflow_result)

        if markdown_content:
            report_agent.save_markdown(markdown_content)

    def _count_docx_media_files(docx_content: bytes | None) -> int:
        if not docx_content:
            return 0
        try:
            with zipfile.ZipFile(io.BytesIO(docx_content)) as archive:
                return sum(1 for name in archive.namelist() if name.startswith("word/media/"))
        except Exception:
            return 0

    if html_content:
        try:
            word_bytes = build_docx_from_html(html_content)
        except Exception as exc:
            print("[REPORT][WORD] build_docx_from_html failed:", repr(exc))
            word_bytes = None

    html_image_count = len(re.findall(r"<img\b", html_content or "", flags=re.IGNORECASE))
    docx_media_count = _count_docx_media_files(word_bytes)
    if word_bytes is not None and html_image_count > 0 and docx_media_count < html_image_count:
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

    if word_bytes is not None:
        print(
            f"[REPORT][WORD] final media count = {_count_docx_media_files(word_bytes)}, html_image_count = {html_image_count}"
        )
        report_agent.save_word(word_bytes)

    if pdf_bytes is not None:
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


def _describe_pdf_export_method(pdf_export_method: str | None) -> str | None:
    if not pdf_export_method:
        return None
    if pdf_export_method.startswith("word:"):
        return "PDF 已通过 Word 文档转换生成，排版和字体优先沿用 Word。"
    if pdf_export_method.startswith("html:"):
        return "当前环境未检测到可用的 Word 转 PDF 能力，已回退为 HTML 渲染生成 PDF，版式可能与 Word 略有差异。"
    return f"PDF 导出方式：{pdf_export_method}"


def _describe_pdf_export_method(pdf_export_method: str | None) -> str | None:
    if not pdf_export_method:
        return None
    if pdf_export_method.startswith("word:"):
        return "PDF 已通过 Word 文档转换生成，版式和字体优先沿用 Word。"
    if pdf_export_method.startswith("html:"):
        return "当前环境未检测到可用的 Word 转 PDF 能力，已回退为 HTML 渲染生成 PDF，样式已尽量对齐 Word 版。"
    return f"PDF 导出方式：{pdf_export_method}"


def _clear_generated_report_files(report_agent) -> None:
    report_agent.save_word(None)
    report_agent.save_pdf(None)
    report_agent.save_pdf_export_method(None)
    report_agent.save_html(None)
    report_agent.save_markdown(None)
    st.session_state.pop("report_final_html", None)


def _clear_report_workflow_outputs(report_agent) -> None:
    _clear_generated_report_files(report_agent)
    report_agent.save_report_workflow_result(None)
    report_agent.save_report(None)
    report_agent.save_report_content(None)

    for field_name in REPORT_WORKFLOW_OUTPUT_FIELDS:
        st.session_state.pop(f"report_{field_name}", None)

    st.session_state.pop("report_workflow_outputs", None)
    st.session_state.pop("report_preference_selected", None)


def _save_report_workflow_outputs(report_agent, workflow_result: dict[str, Any]) -> None:
    extracted_outputs = _extract_report_workflow_outputs(workflow_result)

    report_agent.save_report_workflow_result(workflow_result)
    report_agent.save_report(workflow_result)
    report_agent.save_report_content(None)

    st.session_state.report_workflow_outputs = extracted_outputs
    for field_name in REPORT_WORKFLOW_OUTPUT_FIELDS:
        st.session_state[f"report_{field_name}"] = extracted_outputs.get(field_name)
    st.session_state.report_preference_selected = extracted_outputs.get("preference_selected")


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

    return {
        "load_summary": load_summary,
        "preproc_summary": preproc_summary,
        "visual_summary": visual_summary,
        "coding_summary": coding_summary,
        "selected_full_conten": stringify_string(st.session_state.get("full")),
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

    return {
        "toc_text": _normalize_multiline_text(report_agent.load_outline()),
        "title": "",
        "selected_full_conten": stringify_string(st.session_state.get("report_selected_full_conten")),
        "preference_selected": stringify_string(st.session_state.get("report_preference_selected")),
        "add_preference": stringify_string(st.session_state.get("report_add_preference")),
        "load_abstract": stringify_string(st.session_state.get("report_load_abstract")),
        "preproc_abstract": stringify_string(st.session_state.get("report_preproc_abstract")),
        "visual_abstract": stringify_string(st.session_state.get("report_visual_abstract")),
        "coding_abstract": current_coding_abstract,
    }


def call_coze_workflow_report_stream(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """本地化版本：调用本地 Reporting_toc workflow。"""
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


def call_coze_workflow_word_stream(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """本地化版本：调用本地 Reporting_partly workflow。"""
    from utils.local_workflow_bridge import call_reporting_partly_bridge

    status_placeholder = st.empty()
    status_placeholder.info("正在生成报告，请稍后。")

    inputs = dict(inputs)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    # report workflow 字段名是 preference_select
    inputs.setdefault("preference_select", st.session_state.get("preference_selected") or "")

    result = call_reporting_partly_bridge(inputs)
    if result is None:
        status_placeholder.empty()
        return None

    merged_result = _merge_report_workflow_results([result])
    if merged_result is not None:
        status_placeholder.empty()
        return merged_result

    status_placeholder.empty()
    st.error("Word 报告生成失败，未解析到有效输出，请点击生成报告按钮重新生成。")
    return None


def _generate_formatted_report(report_agent, action: str) -> None:
    workflow_result = call_coze_workflow_word_stream(_build_word_report_inputs(report_agent))
    if not workflow_result:
        return

    raw_content = extract_report_html(workflow_result)
    if not raw_content:
        st.error("Word 报告工作流未返回 `final_html`。")
        return

    raw_content = raw_content.strip()
    report_title = _extract_report_title(workflow_result)
    if report_title:
        st.session_state.report_title = report_title

    if _looks_like_html(raw_content):
        html_content = raw_content
        markdown_content = html_to_markdown(html_content)
    else:
        markdown_content = raw_content
        markdown_content = _deduplicate_report_html_blocks(markdown_content)
        html_content = markdown_to_html(markdown_content, title="")

    html_content = _finalize_report_html(html_content, report_title)
    markdown_content = html_to_markdown(html_content) if html_content else markdown_content

    _clear_generated_report_files(report_agent)

    report_agent.save_report_workflow_result(workflow_result)
    report_agent.save_report(workflow_result)
    report_agent.save_report_content(markdown_content)
    report_agent.save_markdown(markdown_content)
    report_agent.save_html(html_content)

    _prepare_downloadable_reports(report_agent)
    st.session_state.report_final_html = html_content
    st.success(f"{action} 报告已生成，已在下侧展示。")
    

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
    report_agent.save_report_format(report_format)

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
        st.success("目录已生成，已在下侧显示目录文本。")


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
    action = report_agent.load_report_format()
    visualization_agent = st.session_state.get("visualization_agent")

    outline_generated = _has_generated_outline(report_agent)
    report_generated = _has_generated_word_report(report_agent)
    not_generate = outline_generated and not report_generated

    if auto and report_generated and not report_agent.finish_auto_task:
        _complete_auto_report(report_agent)
        st.rerun()

    if st.button(f"生成 {action} 报告") or (auto and not_generate):
        if st.session_state.get("report_selected_full_conten") is None:
            st.warning("请先点击“生成目录”获取新 workflow 输出。")
            return

        if not auto and not _ensure_visualization_ready_for_report(visualization_agent):
            return

        _generate_formatted_report(report_agent, action)

        if auto:
            current_action = report_agent.load_report_format()
            generated = (
                report_agent.load_html() is not None
                if current_action in {"Word", "HTML", "PDF"}
                else report_agent.load_html() is not None
            )
            if generated:
                _complete_auto_report(report_agent)
                st.rerun()


def report_execution(report_agent) -> None:
    action = report_agent.load_report_format()
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
        if html_content:
            components.html(html_content, height=720, scrolling=True)
        elif markdown_content:
            st.markdown(markdown_content)
        return

    if action == "HTML":
        st.download_button(
            label="下载 HTML 报告",
            data=html_content.encode("utf-8") if html_content else b"",
            file_name="report.html",
            mime="text/html",
            disabled=not bool(html_content),
        )
        if html_content:
            components.html(html_content, height=720, scrolling=True)
        elif markdown_content:
            st.markdown(markdown_content)
        return

    if action == "PDF":
        st.download_button(
            label="下载 PDF 报告",
            data=downloadable_reports["pdf"] or b"",
            file_name="report.pdf",
            mime="application/pdf",
            disabled=downloadable_reports["pdf"] is None,
        )

        if html_content:
            components.html(html_content, height=720, scrolling=True)
        elif markdown_content:
            st.markdown(markdown_content)
            

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
