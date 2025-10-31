import os

import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objs as go
import streamlit as st
import streamlit_antd_components as sac

from utils.sanitize_code import sanitize_code
from workflow.visualization.viz_suggestion import vis_button_suggest, vis_talk_suggest
from workflow.visualization.viz_coding import vis_execution, vis_code_gen
from workflow.visualization.viz_quick_action import plot_for_option
from workflow.visualization.viz_color import vis_palette

def vis_quick_actions(agent):

    cols_list = agent.load_df().columns.tolist()
    options = ["直方图", "饼图", "箱线图", "折线图"]

    selected_col = st.selectbox("选择一个列:", cols_list)

    logo_dir = r"logo\sec3"
    logo_paths = {opt: os.path.join(logo_dir, f"{opt}.png") for opt in options}

    cols = st.columns(len(options))

    fig_placeholder = st.empty()

    for idx, opt in enumerate(options):
        with cols[idx]:
            left, center, right = st.columns([1, 8, 1]) 
            with center:
                st.write(opt)
                path = logo_paths.get(opt)
                if path and os.path.exists(path):
                    st.image(Image.open(path), width=80)
                else:
                    st.text("Logo 文件未找到")

                if st.button("Try me", key=f"try_{idx}"):
                    fig = plot_for_option(agent.load_df(), opt, selected_col)
                    fig_placeholder.plotly_chart(fig, use_container_width=True)


def vis_result(agent) -> None:
    
    fig_desc_list = agent.load_fig() 
    total = len(fig_desc_list)
    PAGE_SIZE = 5

    current_page = sac.pagination(
        total=total,
        page_size=PAGE_SIZE,
        align='center',
        jump=False,
        show_total=True,
        variant='filled',
        color='#44658C'
    )

    start_idx = (current_page - 1) * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, total)
    page_items = fig_desc_list[start_idx:end_idx]

    for offset, item in enumerate(page_items):
        
        idx = start_idx + offset
        fig = item["fig"]
        desc = item["desc"]

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"fig_{idx}"
        )
        if desc is not None:
            st.write(desc)


def vis_chat(agent, auto = False):
    
    msg = st.chat_message("assistant")
    msg.write(
        "我是 Autostat 数据分析助手，很高兴为您服务！\n\n"
        "您可以在下方对话框输入具体可视化需求，"
        "也可以点击下面的按钮，一键获取可视化建议并绘图。"
    )
    analyze_clicked = msg.button("🔍 可视化推荐", key="viz_suggest")
    reply_placeholder = msg.empty()

    chat_history = agent.load_memory()

    for idx, entry in enumerate(chat_history):
        bubble = st.chat_message(entry["role"])
        content = entry["content"]
        if isinstance(content, str):
            bubble.write(content)
        elif isinstance(content, go.Figure):
            bubble.plotly_chart(
                content,
                use_container_width=True,
                key=f"chart-{idx}"
            )

    already_generated = any(
        entry["role"] == "assistant" and "图" in str(entry["content"])
        for entry in chat_history
    )

    # 按钮路径
    if analyze_clicked or (auto and not already_generated):
        st.chat_message("user").write("请帮我做可视化分析")
        agent.add_memory({'role': 'user', 'content': "请帮我做可视化分析"})
        with st.spinner("正在处理您的请求..."):
            rec = vis_button_suggest(agent)
            agent.save_suggestion(rec)
            st.chat_message("assistant").write(rec)
            agent.add_memory({"role": "assistant", "content": str(rec)})

    # 对话路径
    reply = None
    user_input = None
    user_input = st.chat_input("请输入需求，例如'请给我一些可视化建议'")
    if user_input is not None:
        st.chat_message("user").write(user_input)
        with st.spinner("正在处理您的请求..."):
            agent.save_user_input(user_input)
            agent.add_memory({"role": "user", "content": user_input})
            rec = vis_talk_suggest(agent, user_input)
            agent.save_suggestion(rec)
            st.chat_message("assistant").write(rec)
            agent.add_memory({"role": "assistant", "content": str(rec)})
            st.rerun()


if __name__ == "__main__":

    st.title("统计可视化分析")
    st.markdown("---")

    preproc_agent = st.session_state.data_preprocess_agent
    load_agent   = st.session_state.data_loading_agent
    planner = st.session_state.planner_agent
    auto = planner.vis_auto

    processed_df = preproc_agent.load_processed_df()
    if processed_df is None:
        df = load_agent.load_df()
    else:
        df = processed_df

    if df is None:
        st.warning("⚠️ 请先在数据导入页面加载数据")
        st.stop()

    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    agent = st.session_state.visualization_agent
    agent.add_df(df_shuffled)

    if st.session_state.auto_mode == True:
        if (agent.finish_auto_task == True and planner.switched_vis == False) or planner.vis_auto == False:
            planner.finish_vis_auto()
            st.switch_page("workflow/modeling/modeling_render.py")

    code = agent.load_code()
    if code is None:
        code_expand = False
    else:
        code_expand = True
        
    fig = agent.load_fig()
    if fig is None:
        fig_expand = False
    else:
        fig_expand = True

    c = st.columns(2)
    # with c[1].expander('快速可视化', False):
    #     vis_quick_actions(agent)
    with c[0].expander('配色选择', True):
        vis_palette(agent)
    with c[1].expander('可视化建议', True):
        vis_chat(agent, auto)
        vis_code_gen(agent, auto = auto)
    with c[0].expander('可视化执行', code_expand):
        vis_execution(agent, auto = auto)
    with c[0].expander('可视化结果', fig_expand):
        vis_result(agent)