import streamlit as st
import plotly.express as px


def plot_for_option(df, option: str, column: str):
    
    series = df[column]
    
    if option == "直方图":
        fig = px.histogram(df, x=column, title=f"{column} 的直方图")
    elif option == "饼图":
        counts = series.value_counts().reset_index()
        counts.columns = [column, 'count']
        fig = px.pie(counts, names=column, values='count', title=f"{column} 的饼图")
    elif option == "折线图":
        fig = px.line(df, y=column, title=f"{column} 的折线图")
    elif option == "箱线图":
        fig = px.box(df, y=column, title=f"{column} 的箱线图")
    else:
        st.error("未知的图表类型")
        return
    
    return fig