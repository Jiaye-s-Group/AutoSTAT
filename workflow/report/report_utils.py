import re
import base64

import streamlit as st
from playwright.sync_api import sync_playwright


def html_to_pdf_bytes_playwright(html: str) -> bytes:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="load")
        pdf_bytes = page.pdf(format="A4", print_background=True)
        browser.close()
        return pdf_bytes


def html_dowmload(full_report):

    try:
        pdf_bytes = html_to_pdf_bytes_playwright(full_report)
    except Exception as e:
        st.error(f"生成 PDF 出错：{e}")
    else:
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")

        auto_download_html = f"""
        <html>
        <body>
            <a id="dl_link"
            href="data:application/pdf;base64,{b64}"
            download="report.pdf"
            style="display:none">download</a>
            <script>
            (function() {{
                const a = document.getElementById('dl_link');
                try {{
                a.click();
                }} catch (err) {{
                // 如果自动点击被阻止，替换页面内容并露出手动链接
                document.body.innerHTML =
                    '<p>自动下载被浏览器阻止，请点击下面链接手动下载：</p>' + a.outerHTML;
                }}
            }})();
            </script>
        </body>
        </html>
        """

        st.components.v1.html(auto_download_html, height=120)

        st.download_button(
            label="⬇️ 手动下载 PDF（回退）",
            data=pdf_bytes,
            file_name="report.pdf",
            mime="application/pdf",
        )

        st.success("PDF 已生成（如未自动下载，请使用上方手动下载按钮）。")