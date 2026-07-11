"""Report export preparation for HTML, Markdown, Word, and PDF."""

from __future__ import annotations

import io
import re
import hashlib
import zipfile
from typing import Any, Callable

import streamlit as st

from workflow.report.report_content_utils import (
    build_docx_from_html,
    build_docx_from_markdown,
    extract_report_html,
    extract_report_markdown,
    extract_report_text,
    extract_report_word_bytes,
    find_first_nested_field,
    html_to_markdown,
    markdown_to_html,
    stringify_string,
)
from workflow.report.report_utils import convert_report_to_pdf_bytes
from workflow.report.report_constants import (
    REPORT_GENERATION_TOKEN_KEY,
    REPORT_PDF_EXPORT_KEY,
    REPORT_WORD_EXPORT_KEY,
)


def clean_report_title_text(raw_title: Any) -> str:
    if raw_title is None:
        return ""

    if isinstance(raw_title, dict):
        for key in ("title", "标题", "题目", "text", "name", "label", "content"):
            cleaned = clean_report_title_text(raw_title.get(key))
            if cleaned:
                return cleaned
        return ""

    if isinstance(raw_title, list):
        for item in raw_title:
            cleaned = clean_report_title_text(item)
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

    parsed = None
    try:
        import json

        parsed = json.loads(text)
    except Exception:
        parsed = None
    if parsed is not None and parsed is not raw_title:
        cleaned = clean_report_title_text(parsed)
        if cleaned:
            return cleaned

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1:
        text = lines[0]
    elif lines:
        for line in lines:
            cleaned = clean_report_title_text(line)
            if cleaned:
                return cleaned

    text = re.sub(r"^(报告标题|标题|题目|title)\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("`'\"“”‘’ \t\r\n")
    text = re.sub(r"\s+", " ", text)
    return text[:80]


def extract_report_title(workflow_result: Any) -> str:
    if isinstance(workflow_result, dict):
        for field in ("title", "report_title", "name"):
            cleaned = clean_report_title_text(find_first_nested_field(workflow_result, [field]))
            if cleaned:
                return cleaned

    return clean_report_title_text(st.session_state.get("report_title"))


def looks_like_html(text: str) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip().lower()
    return bool(
        stripped.startswith("<!doctype html")
        or stripped.startswith("<html")
        or re.search(r"<(section|article|div|h1|h2|p|table|figure)\b", stripped)
    )


def prepare_downloadable_reports(
    report_agent,
    *,
    generation_token: str | None = None,
    finalize_report_html: Callable[[str, str], str] | None = None,
) -> dict[str, Any]:
    workflow_result = report_agent.load_report_workflow_result()
    html_content = report_agent.load_html()
    markdown_content = report_agent.load_markdown()
    word_bytes = report_agent.load_word()
    pdf_bytes = report_agent.load_pdf()
    pdf_export_method = report_agent.load_pdf_export_method()

    def can_save_prepared_output() -> bool:
        return (
            generation_token is None
            or st.session_state.get(REPORT_GENERATION_TOKEN_KEY) == generation_token
        )

    if workflow_result and not html_content:
        raw_content = extract_report_html(workflow_result)
        if raw_content:
            raw_content = raw_content.strip()
            report_title = extract_report_title(workflow_result)
            if report_title:
                st.session_state.report_title = report_title

            if looks_like_html(raw_content):
                html_content = raw_content
                markdown_content = html_to_markdown(html_content)
            else:
                markdown_content = raw_content
                html_content = markdown_to_html(markdown_content, title="")

            if finalize_report_html is not None:
                html_content = finalize_report_html(html_content, report_title)
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

    export_cache_key = _build_export_cache_key(html_content, markdown_content)
    if st.session_state.get(REPORT_WORD_EXPORT_KEY) != export_cache_key:
        word_bytes = None
    if st.session_state.get(REPORT_PDF_EXPORT_KEY) != export_cache_key:
        pdf_bytes = None
        pdf_export_method = None

    def count_docx_media_files(docx_content: bytes | None) -> int:
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
    docx_media_count = count_docx_media_files(word_bytes)
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
            f"[REPORT][WORD] final media count = {count_docx_media_files(word_bytes)}, html_image_count = {html_image_count}"
        )
        report_agent.save_word(word_bytes)
        st.session_state[REPORT_WORD_EXPORT_KEY] = export_cache_key

    if pdf_bytes is not None and can_save_prepared_output():
        report_agent.save_pdf(pdf_bytes)
        report_agent.save_pdf_export_method(pdf_export_method)
        st.session_state[REPORT_PDF_EXPORT_KEY] = export_cache_key

    return {
        "word": word_bytes,
        "html": html_content,
        "markdown": markdown_content,
        "pdf": pdf_bytes,
        "pdf_export_method": pdf_export_method,
        "export_cache_key": export_cache_key,
    }


def ensure_pdf_download_ready(report_agent, downloadable_reports: dict[str, Any]) -> dict[str, Any]:
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
    export_cache_key = downloadable_reports.get("export_cache_key")
    if export_cache_key:
        st.session_state[REPORT_PDF_EXPORT_KEY] = export_cache_key
    return downloadable_reports


def _build_export_cache_key(html_content: str | None, markdown_content: str | None) -> str:
    digest_source = "\n\n".join([html_content or "", markdown_content or ""])
    return hashlib.sha256(digest_source.encode("utf-8", errors="ignore")).hexdigest()
