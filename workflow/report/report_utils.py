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
        st.error(f"Error generating PDF: {e}")
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
                // If automatic click is blocked, replace page content and show manual link
                document.body.innerHTML =
                    '<p>Automatic download blocked by browser, please click the link below to download manually:</p>' + a.outerHTML;
                }}
            }})();
            </script>
        </body>
        </html>
        """

        st.components.v1.html(auto_download_html, height=120)

        st.download_button(
            label="⬇️ Manually Download PDF (Fallback)",
            data=pdf_bytes,
            file_name="report.pdf",
            mime="application/pdf",
        )

        st.success("PDF has been generated (if not downloaded automatically, please use the manual download button above).")