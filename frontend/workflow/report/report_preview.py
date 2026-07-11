"""Report preview rendering helpers."""

from __future__ import annotations

import re

import streamlit as st
import streamlit.components.v1 as components

from workflow.report.report_constants import REPORT_PENDING_PREVIEW_KEY
from workflow.report.report_content_utils import markdown_to_html, stringify_string


def build_markdown_preview(markdown_text: str) -> str:
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


def render_report_preview(html_content: str, markdown_content: str) -> None:
    if (html_content or "").strip():
        components.html(html_content, height=720, scrolling=True)
        return

    if (markdown_content or "").strip():
        st.markdown(build_markdown_preview(markdown_content))


def insert_dim_preview_style(html_content: str) -> str:
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


def clear_pending_report_preview() -> None:
    st.session_state.pop(REPORT_PENDING_PREVIEW_KEY, None)


def render_pending_report_preview() -> None:
    preview = st.session_state.get(REPORT_PENDING_PREVIEW_KEY)
    if not isinstance(preview, dict):
        return

    is_live = bool(preview.get("live"))
    html_content = stringify_string(preview.get("html")).strip()
    markdown_content = stringify_string(preview.get("markdown")).strip()
    if html_content:
        display_html = html_content if is_live else insert_dim_preview_style(html_content)
        components.html(display_html, height=720, scrolling=True)
        return

    if markdown_content:
        preview_html = markdown_to_html(build_markdown_preview(markdown_content), title="")
        display_html = preview_html if is_live else insert_dim_preview_style(preview_html)
        components.html(display_html, height=720, scrolling=True)
