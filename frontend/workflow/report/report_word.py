import streamlit as st

from utils.i18n import bt
from workflow.report.report_content_utils import (
    build_docx_from_html,
    build_docx_from_markdown,
    extract_report_markdown,
    extract_report_text,
    extract_report_word_bytes,
    html_to_markdown,
)


def write_word(report_agent):
    workflow_result = report_agent.load_report_workflow_result()
    word_bytes = report_agent.load_word()

    if word_bytes is None and workflow_result:
        word_bytes = extract_report_word_bytes(workflow_result)

    if word_bytes is None:
        html_content = report_agent.load_html() or report_agent.load_report_content()
        markdown_text = html_to_markdown(html_content) if html_content else ""
        if not markdown_text:
            markdown_text = extract_report_markdown(workflow_result)
        if not markdown_text:
            markdown_text = extract_report_text(workflow_result)
        if not markdown_text:
            st.error(bt(
                "报告工作流未返回可用于导出 Word 的内容。",
                "The report workflow did not return content that can be exported as Word.",
            ))
            return
        if html_content:
            try:
                word_bytes = build_docx_from_html(html_content)
            except Exception:
                word_bytes = build_docx_from_markdown(markdown_text)
        else:
            word_bytes = build_docx_from_markdown(markdown_text)

    report_agent.save_word(word_bytes)
    st.success(bt("Word 报告已生成。", "The Word report has been generated."))
