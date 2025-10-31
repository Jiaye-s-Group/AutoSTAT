import streamlit as st
import plotly.express as px


def plot_for_option(df, option: str, column: str):
    
    series = df[column]
    
    if option == "Histogram":
        fig = px.histogram(df, x=column, title=f"Histogram of {column}")
    elif option == "Pie Chart":
        counts = series.value_counts().reset_index()
        counts.columns = [column, 'count']
        fig = px.pie(counts, names=column, values='count', title=f"Pie Chart of {column}")
    elif option == "Line Chart":
        fig = px.line(df, y=column, title=f"Line Chart of {column}")
    elif option == "Box Plot":
        fig = px.box(df, y=column, title=f"Box Plot of {column}")
    else:
        st.error("Unknown chart type")
        return
    
    return fig