import streamlit as st

from workflow.report.report_content_utils import (
    extract_report_markdown,
    extract_report_text,
    html_to_markdown,
)


def write_markdown(report_agent):
    markdown_content = report_agent.load_markdown()
    workflow_result = report_agent.load_report_workflow_result()

    if not markdown_content:
        html_content = report_agent.load_html() or report_agent.load_report_content()
        if html_content:
            markdown_content = html_to_markdown(html_content)

    if not markdown_content:
        markdown_content = extract_report_markdown(workflow_result)
    if not markdown_content:
        markdown_content = extract_report_text(workflow_result)

    if not markdown_content:
        st.error("报告工作流未返回可用于导出 Markdown 的内容。")
        return

    report_agent.save_markdown(markdown_content)
    st.success("Markdown 报告已生成。")
