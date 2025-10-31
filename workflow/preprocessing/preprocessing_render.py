import io
import traceback

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_ace import st_ace
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from utils.sanitize_code import sanitize_code
from workflow.preprocessing.preprocessing_core import prep_meta_execution, prep_code_gen


def prep_basic_info(agent):

    df = agent.load_df()

    # 展示基本统计
    r, c = df.shape
    missing = int(df.isnull().sum().sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("行数", r)
    col2.metric("列数", c)
    col3.metric("缺失值总数", missing)

    dtype_info = pd.DataFrame({
        '列名': df.columns,
        '类型': df.dtypes.astype(str),
        '非空值数量': df.count().values,
        '缺失值比例(%)': (df.isnull().mean() * 100).round(2).values,
    })
    dtype_info = dtype_info.reset_index(drop=True)
    st.dataframe(dtype_info, use_container_width=True)


def prep_execution(agent, auto=False):
    ''' 
    training data进行预处理
    '''

    code = agent.load_code()
    df = agent.load_df()

    process_df = prep_meta_execution(agent, code, df, auto=auto)


def prep_result(agent):
    
    process_df = agent.load_processed_df()
    df = agent.load_df()
    
    if process_df is not None:
        st.write("处理前数据预览：", df.head(10))
        st.write("处理后数据预览：", process_df.head(10))
            
        csv_buffer = io.StringIO()
        process_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        
        st.download_button(
            label="⬇️ 下载处理后数据",
            data=csv_bytes,
            file_name="processed_data.csv",
            mime="text/csv",
        )


def prep_chat(agent, auto=False):
    """渲染对话式建议区"""

    with st.chat_message("assistant"):
        st.write("我是 Autostat 数据分析助手，很高兴为您服务！\n\n"
            "您可以在下方输入预处理需求，或直接点击按钮获取预处理建议。")
        analyze_btn = st.button("🔍 预处理推荐", key='prep_suggest')

    # 对话历史渲染
    chat_history = agent.load_memory()

    for idx, entry in enumerate(chat_history):
        bubble = st.chat_message(entry["role"])
        content = entry["content"]
        if isinstance(content, str):
            bubble.write(content)

    already_generated = any(
        entry["role"] == "assistant" and "预处理" in str(entry["content"])
        for entry in chat_history
    )

    # 自动/手动触发
    if analyze_btn or (auto and not already_generated):

        st.chat_message("user").write("请给我预处理建议")
        agent.add_memory({'role': 'user', 'content': "请给我预处理建议"})

        with st.spinner("生成建议中…"):
            text = agent.get_preprocessing_suggestions()
            agent.save_preprocessing_suggestions(text)
            agent.refine_suggestions(df.head(10).to_string())
        st.chat_message("assistant").write(text)
        agent.add_memory({'role': 'assistant', 'content': text})

    # 用户自然语言交互
    user_input = st.chat_input("请输入您的问题")
    if user_input:
        st.chat_message("user").write(user_input)
        agent.add_memory({'role': 'user', 'content': user_input})
        agent.save_user_input(user_input)
        with st.spinner("处理中…"):
            reply = agent.get_preprocessing_suggestions(user_input)
            agent.save_preprocessing_suggestions(reply)
            agent.refine_suggestions(df.head(10).to_string())
        st.chat_message('assistant').write(reply)
        agent.add_memory({'role': 'assistant', 'content': reply})          


if __name__ == '__main__':

    st.title("数据预处理与标准化")

    st.markdown("---")

    data_loading_agent = st.session_state.data_loading_agent
    df = data_loading_agent.load_df()
    planner = st.session_state.planner_agent
    auto = planner.prep_auto

    if df is None:
        st.warning("⚠️ 请先在数据导入页面加载数据")
        st.stop()

    agent = st.session_state.data_preprocess_agent
    agent.add_df(df)

    if st.session_state.auto_mode == True:
        if (agent.finish_auto_task == True and planner.switched_prep == False) or planner.prep_auto == False:
            planner.finish_prep_auto()
            st.switch_page("workflow/visualization/viz_render.py")

    code = agent.load_code()
    if code is None:
        code_expand = False
    else:
        code_expand = True

    c = st.columns(2)
    with c[0].expander('预处理展示', True):
        prep_basic_info(agent)
    with c[1].expander('预处理建议', True):
        prep_chat(agent, auto)
        prep_code_gen(agent, auto=auto)
    with c[0].expander('预处理执行', code_expand):
        prep_execution(agent, auto)
    with c[0].expander('预处理结果', code_expand):
        prep_result(agent)