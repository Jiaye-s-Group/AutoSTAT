import io
import re
import base64
import plotly.io as pio
import streamlit as st
def write_markdown(agents):
    report_agent = agents[-1]
    report_obj = report_agent.load_report()  # Reportcore

    # 图像分析列表
    analysis_list = agents[2].summary_fig_analysis_list()

    md_parts = []

    def _process_node(node):
        if node.type == "heading":
            prefix = "#" * (node.level if node.level > 0 else 1)
            md_parts.append(f"{prefix} {node.text}\n")
            for ch in node.children:
                _process_node(ch)

        elif node.type == "paragraph":
            parts = re.split(r'(\[FIG:\d+\])', node.text)
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                if part.startswith("[FIG:") and part.endswith("]"):
                    idx = int(part[5:-1])
                    if 0 <= idx < len(analysis_list):
                        fig_obj = analysis_list[idx].get("figure")
                        try:
                            buf = io.BytesIO()
                            pio.write_image(fig_obj, buf, format="png")
                            data = buf.getvalue()
                            b64 = base64.b64encode(data).decode("utf-8")
                            # 🔹 直接内嵌 base64
                            md_parts.append(
                                f"![Figure {idx}](data:image/png;base64,{b64})\n"
                            )
                        except Exception as e:
                            md_parts.append(f"> **图像插入失败**: {e}\n")
                else:
                    md_parts.append(f"{part}\n\n")

        else:  # root
            for ch in node.children:
                _process_node(ch)

    _process_node(report_obj.root)

    md_content = "\n".join(md_parts)
    report_agent.save_markdown(md_content)
    st.success("Markdown 报告生成成功 ✅")