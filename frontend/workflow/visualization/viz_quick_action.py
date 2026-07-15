import streamlit as st
import plotly.express as px

from utils.i18n import bt


def plot_for_option(df, option: str, column: str):
    
    series = df[column]
    
    if option in {"直方图", "Histogram"}:
        fig = px.histogram(df, x=column, title=bt(f"{column} 的直方图", f"Histogram of {column}"))
    elif option in {"饼图", "Pie Chart"}:
        counts = series.value_counts().reset_index()
        counts.columns = [column, 'count']
        fig = px.pie(counts, names=column, values='count', title=bt(f"{column} 的饼图", f"Pie Chart of {column}"))
    elif option in {"折线图", "Line Chart"}:
        fig = px.line(df, y=column, title=bt(f"{column} 的折线图", f"Line Chart of {column}"))
    elif option in {"箱线图", "Box Plot"}:
        fig = px.box(df, y=column, title=bt(f"{column} 的箱线图", f"Box Plot of {column}"))
    else:
        st.error(bt("未知的图表类型", "Unknown chart type"))
        return
    
    return fig
