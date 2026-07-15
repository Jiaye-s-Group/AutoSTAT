import streamlit as st

from utils.i18n import bt
from workflow.report.report_content_utils import (
    extract_report_html,
    extract_report_markdown,
    extract_report_text,
    markdown_to_html,
)


def write_html(report_agent):
    html_content = report_agent.load_html() or report_agent.load_report_content()
    workflow_result = report_agent.load_report_workflow_result()

    if not html_content and workflow_result:
        html_content = extract_report_html(workflow_result)

    if not html_content:
        markdown_text = extract_report_markdown(workflow_result)
        if not markdown_text:
            markdown_text = extract_report_text(workflow_result)
        if not markdown_text:
            st.error(bt(
                "报告工作流未返回可用于导出 HTML 的内容。",
                "The report workflow did not return content that can be exported as HTML.",
            ))
            return
        html_content = markdown_to_html(markdown_text, title="Analysis Report")

    report_agent.save_html(html_content)
    st.success(bt("HTML 报告已生成。", "The HTML report has been generated."))
