"""Deterministic visualization actions that do not require an LLM call."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st

from workflow.visualization.viz_color import apply_palette_to_figure


QUICK_ACTIONS = (
    "Histogram",
    "Bar Count",
    "Box Plot",
    "Line Chart",
    "Scatter Plot",
    "Pie Chart",
)


def render_quick_visualization(agent, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("加载数据后可使用快速可视化。")
        return

    column_labels = [str(column) for column in df.columns]
    label_to_column = dict(zip(column_labels, df.columns))
    numeric_labels = [str(column) for column in df.select_dtypes(include="number").columns]
    default_label = numeric_labels[0] if numeric_labels else column_labels[0]

    action = st.selectbox("图表类型", QUICK_ACTIONS, key="viz_quick_action_type")
    y_label = st.selectbox(
        "主变量",
        column_labels,
        index=column_labels.index(default_label),
        key="viz_quick_y_column",
    )
    y_column = label_to_column[y_label]
    x_column = None
    if action in {"Line Chart", "Scatter Plot"}:
        x_candidates = ["使用行序号", *column_labels]
        x_choice = st.selectbox("横轴", x_candidates, key="viz_quick_x_column")
        x_column = None if x_choice == "使用行序号" else label_to_column[x_choice]

    if st.button("生成快速图表", use_container_width=True, key="viz_quick_generate"):
        fig = build_quick_figure(df, action=action, y_column=y_column, x_column=x_column)
        if fig is None:
            st.warning("当前列组合不适合生成该图表。")
            return

        colors = agent.load_color() or []
        display_fig = apply_palette_to_figure(fig, colors, 0) if colors else fig
        title = _figure_title(action, str(y_column), str(x_column) if x_column is not None else None)
        display_fig.update_layout(title=title)
        agent.save_fig([
            {
                "fig": display_fig,
                "base_fig": fig,
                "desc": f"快速可视化：{title}",
                "title": title,
            }
        ])
        st.session_state.tu_title = [title]
        st.session_state.full = f"[FIG:0] 图题：{title}"
        st.session_state.abstract_3 = f"已生成快速可视化图表：{title}。"
        st.session_state.summary_3 = {
            "fig_analysis": [{"title": title, "analysis": f"快速可视化展示 {y_column} 的分布或关系。"}]
        }
        st.session_state.visual_recommendatio = f"快速可视化已生成：{title}"
        st.session_state.viz_suggestion = st.session_state.visual_recommendatio
        st.success("快速图表已生成，可在“可视化结果”中查看，也可进入报告。")


def build_quick_figure(
    df: pd.DataFrame,
    *,
    action: str,
    y_column: str,
    x_column: str | None = None,
) -> go.Figure | None:
    if y_column not in df.columns:
        return None

    if action == "Histogram":
        return px.histogram(df, x=y_column, title=_figure_title(action, y_column, x_column))

    if action == "Bar Count":
        counts = df[y_column].value_counts(dropna=False).reset_index()
        counts.columns = [y_column, "count"]
        return px.bar(counts, x=y_column, y="count", title=_figure_title(action, y_column, x_column))

    if action == "Box Plot":
        if not _is_numeric(df[y_column]):
            return None
        return px.box(df, y=y_column, title=_figure_title(action, y_column, x_column))

    if action == "Line Chart":
        if x_column and x_column in df.columns:
            return px.line(df, x=x_column, y=y_column, title=_figure_title(action, y_column, x_column))
        indexed_df = df.reset_index(names="row_index")
        return px.line(indexed_df, x="row_index", y=y_column, title=_figure_title(action, y_column, x_column))

    if action == "Scatter Plot":
        if not x_column or x_column not in df.columns:
            return None
        if not (_is_numeric(df[x_column]) and _is_numeric(df[y_column])):
            return None
        return px.scatter(df, x=x_column, y=y_column, title=_figure_title(action, y_column, x_column))

    if action == "Pie Chart":
        counts = df[y_column].value_counts(dropna=False).reset_index()
        counts.columns = [y_column, "count"]
        return px.pie(counts, names=y_column, values="count", title=_figure_title(action, y_column, x_column))

    return None


def _is_numeric(series: Any) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _figure_title(action: str, y_column: str, x_column: str | None = None) -> str:
    labels = {
        "Histogram": "直方图",
        "Bar Count": "类别计数",
        "Box Plot": "箱线图",
        "Line Chart": "折线图",
        "Scatter Plot": "散点图",
        "Pie Chart": "饼图",
    }
    chart_name = labels.get(action, action)
    if x_column:
        return f"{x_column} 与 {y_column} 的{chart_name}"
    return f"{y_column} 的{chart_name}"
