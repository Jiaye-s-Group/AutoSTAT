from pathlib import Path
import sys

import plotly.graph_objects as go
import streamlit as st
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT))

from frontend.workflow.report import report_render


class DummyVisualizationAgent:
    def __init__(self, figures):
        self._figures = figures

    def load_fig(self):
        return self._figures


figures = [{"fig": go.Figure(data=go.Bar(y=[1, 2, 3]))} for _ in range(8)]
st.session_state["visualization_agent"] = DummyVisualizationAgent(figures)
st.session_state["tu_title"] = [
    "图1 示例1",
    "图2 示例2",
    "图3 示例3",
    "图4 示例4",
    "图5 示例5",
    "图6 花萼长度总体分布的小提琴图",
    "图7 示例7",
    "图8 按物种划分的花萼长度堆叠柱状图",
]

html_text = "<main><p>前文<span>[FIG:8]</span>后文</p><p><strong>[FIG:6]</strong></p></main>"
output_html = report_render._inject_visualizations_into_html(html_text)
soup = BeautifulSoup(output_html, "html.parser")

captions = [
    tag.get_text(" ", strip=True)
    for tag in soup.find_all("div", class_="report-figure-caption")
]
figure_blocks = soup.find_all("div", class_="report-figure-block")

assert "[FIG:" not in output_html
assert len(figure_blocks) == 2
assert captions == [
    "图1 按物种划分的花萼长度堆叠柱状图",
    "图2 花萼长度总体分布的小提琴图",
]
assert all(block.find_parent(["span", "strong"]) is None for block in figure_blocks)

print("OK: report render preserves inline placeholder figures and renumbers captions by report order")
