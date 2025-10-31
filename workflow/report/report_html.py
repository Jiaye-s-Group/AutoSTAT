import io
import re
import streamlit as st
import plotly.io as pio
from utils.sanitize_code import sanitize_code
import base64

def write_html(agents):
    report_agent = agents[-1]
    report_obj = report_agent.load_report()  # Reportcore

    # 图像分析列表
    analysis_list = agents[2].summary_fig_analysis_list()

    # 给 heading 加唯一 id
    heading_counter = {"count": 0}
    def _gen_id(text):
        heading_counter["count"] += 1
        return f"sec-{heading_counter['count']}"

    # 遍历树 → 正文 & TOC
    toc_items, content_items = [], []

    def _process_node(node):
        if node.type == "heading":
            sec_id = _gen_id(node.text)
            toc_items.append((sec_id, node.text, node.level))
            content_items.append(
                f"<h{node.level} id='{sec_id}' class='font-bold text-gray-800 mt-8 mb-4 text-{max(6-node.level,1)}xl'>{node.text}</h{node.level}>"
            )
            for ch in node.children:
                _process_node(ch)

        elif node.type == "paragraph":
            parts = re.split(r'(\[FIG:\d+\])', node.text)
            html_parts = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part.startswith("[FIG:") and part.endswith("]"):
                    idx = int(part[5:-1])
                    fig_html = ""
                    if 0 <= idx < len(analysis_list):
                        fig_obj = analysis_list[idx].get("figure")
                        try:
                            buf = io.BytesIO()
                            pio.write_image(fig_obj, buf, format="png")
                            data = buf.getvalue()
                            b64 = base64.b64encode(data).decode("utf-8")
                            fig_html = f"<div class='flex justify-center my-6'><img src='data:image/png;base64,{b64}' class='rounded-xl shadow-md max-w-3xl w-full'/></div>"
                        except Exception as e:
                            fig_html = f"<p class='text-red-500'>[图像插入失败: {e}]</p>"
                    html_parts.append(fig_html)
                else:
                    html_parts.append(f"<p class='text-gray-700 leading-relaxed mb-4'>{part}</p>")
            content_items.append("".join(html_parts))

        else:  # root
            for ch in node.children:
                _process_node(ch)

    _process_node(report_obj.root)

    # TOC HTML
    toc_html = ["<nav class='space-y-2'>"]
    prev_level = -1
    for sec_id, text, level in toc_items:
        indent = "ml-" + str(level * 4)
        toc_html.append(f"<a href='#{sec_id}' class='block {indent} text-gray-600 hover:text-blue-600 transition-colors'>{text}</a>")
    toc_html.append("</nav>")

    # 拼接完整 HTML
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
        document.addEventListener("DOMContentLoaded", function() {{
            const sections = document.querySelectorAll("h1, h2, h3, h4, h5, h6");
            const navLinks = document.querySelectorAll("nav a");

            function onScroll() {{
                let scrollPos = document.documentElement.scrollTop || document.body.scrollTop;
                let currentId = "";
                sections.forEach(sec => {{
                    if (sec.offsetTop - 80 <= scrollPos) {{
                        currentId = sec.id;
                    }}
                }});
                navLinks.forEach(link => {{
                    link.classList.remove("font-bold", "text-blue-600");
                    if (link.getAttribute("href") === "#" + currentId) {{
                        link.classList.add("font-bold", "text-blue-600");
                    }}
                }});
            }}
            window.addEventListener("scroll", onScroll);
            onScroll();
        }});
        </script>
    </head>
    <body class="flex font-sans">
        <aside class="fixed top-0 left-0 h-screen w-64 bg-gray-100 border-r border-gray-300 p-6 overflow-y-auto">
            <h2 class="text-xl font-bold mb-4">目录</h2>
            {''.join(toc_html)}
        </aside>
        <main class="ml-64 p-10 w-full max-w-5xl">
            {''.join(content_items)}
        </main>
    </body>
    </html>
    """

    report_agent.save_html(html_content)
    st.success("HTML 报告 (Tailwind 风格) 生成成功 ✅")
