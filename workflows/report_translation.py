"""Report outline and body translation helpers."""
from __future__ import annotations

import re
from typing import Any, Callable

from bs4 import BeautifulSoup, Tag

from core.llm_client import chat
from core.prompt_template import render_file
from core.report_language import (
    REPORT_LANGUAGE_EN,
    REPORT_LANGUAGE_ZH,
    normalize_report_language,
    report_language_html_lang,
    report_language_name,
)
from frontend.workflow.report.report_content_utils import truncate_text


ProgressCallback = Callable[[dict[str, Any]], None]


def _unwrap_plain_text(raw: str) -> str:
    text = str(raw or "").strip()
    match = re.match(r"^```(?:text|markdown)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
    text = re.sub(r"^(?:translated text|translation)\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    return text.strip().strip('"').strip("'").strip()


def _language_pair(source_language: Any, target_language: Any) -> tuple[str, str]:
    source = normalize_report_language(source_language)
    target = normalize_report_language(target_language)
    if source == target:
        source = REPORT_LANGUAGE_EN if target == REPORT_LANGUAGE_ZH else REPORT_LANGUAGE_ZH
    return source, target


def _should_translate_text(text: str) -> bool:
    clean = re.sub(r"\s+", "", str(text or ""))
    if not clean:
        return False
    if re.fullmatch(r"[\d\W_]+", clean, flags=re.UNICODE):
        return False
    if re.fullmatch(r"\[FIG:\d+\]", clean, flags=re.IGNORECASE):
        return False
    return True


def _progress_snippet(text: str, limit: int = 15) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return ""
    return f"{clean[:limit]}..."


def translate_report_text(
    text: str,
    *,
    source_language: Any,
    target_language: Any,
    context_hint: str = "",
) -> str:
    text = str(text or "")
    if not _should_translate_text(text):
        return text

    source, target = _language_pair(source_language, target_language)
    ctx = {
        "source_language": source,
        "target_language": target,
        "source_language_name": report_language_name(source),
        "target_language_name": report_language_name(target),
        "context_hint": truncate_text(context_hint, 800),
        "text": text,
    }
    sys_prompt = render_file("report_translation/text_translate_sys.txt", ctx)
    user_prompt = render_file("report_translation/text_translate_user.txt", ctx)
    translated = chat(sys_prompt, user_prompt, name="report_translate.text", temperature=0.1).strip()
    translated = _unwrap_plain_text(translated)
    return translated or text


def _line_prefix(line: str) -> str:
    match = re.match(r"^(\s*\d+(?:[\.．]\d+)*(?:[\.．、]|\s+)?)", line or "")
    return match.group(1).replace("．", ".") if match else ""


def _preserve_line_prefix(source_line: str, translated_line: str) -> str:
    source_prefix = _line_prefix(source_line)
    if not source_prefix:
        return translated_line.strip()
    translated = translated_line.strip()
    translated_prefix = _line_prefix(translated)
    if translated_prefix:
        translated = translated[len(translated_prefix) :].strip()
    return f"{source_prefix}{translated}".strip()


def translate_report_toc(
    toc_text: str,
    *,
    source_language: Any,
    target_language: Any,
) -> str:
    source, target = _language_pair(source_language, target_language)
    toc_text = str(toc_text or "").replace("\\r\\n", "\n").replace("\\n", "\n").strip()
    if not toc_text:
        return ""

    ctx = {
        "source_language": source,
        "target_language": target,
        "source_language_name": report_language_name(source),
        "target_language_name": report_language_name(target),
        "toc_text": toc_text,
    }
    sys_prompt = render_file("report_translation/toc_translate_sys.txt", ctx)
    user_prompt = render_file("report_translation/toc_translate_user.txt", ctx)
    raw = chat(sys_prompt, user_prompt, name="report_translate.toc", temperature=0.1).strip()
    translated = _unwrap_plain_text(raw).replace("\\r\\n", "\n").replace("\\n", "\n")

    source_lines = [line for line in toc_text.splitlines() if line.strip()]
    translated_lines = [line for line in translated.splitlines() if line.strip()]
    if len(source_lines) != len(translated_lines):
        translated_lines = [
            translate_report_text(
                line,
                source_language=source,
                target_language=target,
                context_hint="Report outline item. Preserve the numbering prefix exactly.",
            )
            for line in source_lines
        ]

    normalized_lines = [
        _preserve_line_prefix(source_line, translated_line)
        for source_line, translated_line in zip(source_lines, translated_lines)
    ]
    return "\n".join(normalized_lines).strip()


def _translatable_blocks(soup: BeautifulSoup) -> list[Tag]:
    block_names = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "figcaption"}
    blocks: list[Tag] = []
    seen: set[int] = set()

    for tag in soup.find_all(True):
        if not isinstance(tag, Tag):
            continue
        classes = set(tag.get("class") or [])
        is_caption = "report-figure-caption" in classes or "report-modeling-table-title" in classes
        if tag.name not in block_names and not is_caption:
            continue
        if tag.find(["script", "style", "noscript", "img"]):
            continue
        text = tag.get_text(" ", strip=True)
        if not _should_translate_text(text):
            continue
        tag_id = id(tag)
        if tag_id in seen:
            continue
        seen.add(tag_id)
        blocks.append(tag)

    return blocks


def translate_report_html(
    html_text: str,
    *,
    source_language: Any,
    target_language: Any,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    source, target = _language_pair(source_language, target_language)
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    html_tag = soup.find("html")
    if html_tag is not None:
        html_tag["lang"] = report_language_html_lang(target)

    blocks = _translatable_blocks(soup)
    total_blocks = len(blocks)

    def emit(phase: str, index: int = 0, block_text: str = "") -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "status": "translating" if phase != "completed" else "complete",
                "phase": phase,
                "block_index": index,
                "total_blocks": total_blocks,
                "completed_blocks": max(0, min(index, total_blocks)),
                "section_title": _progress_snippet(block_text),
                "html": str(soup),
            }
        )

    emit("translation_ready", 0)
    for index, tag in enumerate(blocks, start=1):
        original_text = tag.get_text(" ", strip=True)
        emit("block_started", index - 1, original_text)
        translated_text = translate_report_text(
            original_text,
            source_language=source,
            target_language=target,
            context_hint="Visible text from a data analysis report. Preserve numbers, metrics, and figure placeholders.",
        )
        tag.clear()
        tag.string = translated_text
        emit("block_completed", index, translated_text)

    emit("completed", total_blocks)
    return {
        "html": str(soup),
        "source_language": source,
        "target_language": target,
        "translated_blocks": total_blocks,
    }
