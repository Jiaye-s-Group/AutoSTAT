import os
from typing import List, Optional

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac

from workflow.dataloading.dataloading_core import process_complex_data, load_from_path, load_concat_file, PathFileWrapper

def loading_reference_docs(agent):
    """
    专门处理参考资料的上传逻辑
    """
    st.info("💡 提示：在此处上传业务背景、算法说明或数据手册 (PDF/Docx)，AI 会学习这些内容。")
    
    # 2. 文件上传组件
    uploaded_docs = st.file_uploader(
        "上传参考文档",
        type=['pdf', 'docx', 'txt', 'names'],
        accept_multiple_files=True,
        key="ref_doc_uploader" # 设置 key 防止与数据上传冲突
    )

    if uploaded_docs:
        # 记录已处理过的文件名，避免重复学习
        if 'learned_doc_names' not in st.session_state:
            st.session_state.learned_doc_names = set()

        new_files = [f for f in uploaded_docs if f.name not in st.session_state.learned_doc_names]
        
        if new_files:
            if st.button("🧠 学习选中的资料", use_container_width=True):
                with st.spinner("正在解析文档并提取知识点..."):
                    count = st.session_state.retriever.add_uploaded_files(new_files)
                    for f in new_files:
                        st.session_state.learned_doc_names.add(f.name)
                st.success(f"学习成功！新增 {len(new_files)} 份文档，提取了 {count} 条知识片段。")
        else:
            st.caption("✅ 当前上传的文件已全部在知识库中。")

    # 展示已加载的文档列表（可选）
    if 'learned_doc_names' in st.session_state and st.session_state.learned_doc_names:
        with st.expander("查看当前已加载的外部资料"):
            for name in st.session_state.learned_doc_names:
                st.write(f"- 📄 {name}")
def loading_data_file(agent):

    st.info(
        "💡 提示：\n"
        "1. 支持一次上传多个数据文件\n"
        "2. 自动使用大模型分析并处理数据\n"
        "3. 支持多种格式的文件类型上传\n"
    )

    selected_index = sac.tabs([
        sac.TabsItem(label='本地上传'),
        sac.TabsItem(label='路径导入'),
    ], color='#5980AE',)

    if selected_index == "本地上传":
        # 点击上传文件
        uploaded_files = st.file_uploader(
            "选择新文件",
            accept_multiple_files=True,
            help="拖拽或点击上传多个文件",
        )

        if uploaded_files:
            current_memory_file_name = agent.load_file_name()
            new_files = [f for f in uploaded_files if f.name not in current_memory_file_name]
            if new_files:
                try:
                    with st.spinner("正在处理数据..."):
                        df, dfs = process_complex_data(new_files, agent)
                    if df is not None:
                        agent.add_df(df)
                        agent.save_dfs(dfs)
                        for f in new_files:
                            agent.save_file_name(f.name)
                        st.rerun()
                except Exception as err:
                    st.error(f"导入失败：{err}")

    elif selected_index == "路径导入":
        # 路径上传文件
        raw_paths = st.text_area(
            "从路径导入数据 (每行一个文件路径)",
            placeholder=    "C:\\data\\iris.names\nC:\\data\\iris.data",
            height=100
        )

        if st.button("从路径加载文件", use_container_width=True):
            if raw_paths:

                path_list = [p.strip().strip("'\"") for p in raw_paths.strip().split('\n') if p.strip()]
                
                valid_paths = [p for p in path_list if os.path.exists(p)]
                invalid_paths = [p for p in path_list if not os.path.exists(p)]

                if invalid_paths:
                    st.warning(f"路径不存在，已跳过：\n- " + "\n- ".join(invalid_paths))

                if not valid_paths:
                    st.error("未找到任何有效的本地文件路径。")
                else:
                    current_memory_file_name = agent.load_file_name()
                    new_paths = [p for p in valid_paths if p not in current_memory_file_name]

                    if not new_paths:
                        st.info("所有指定的路径文件均已加载。")
                    else:
                        files_to_process = [PathFileWrapper(p) for p in new_paths]
                        try:
                            with st.spinner("正在处理数据..."):
                                df, dfs = process_complex_data(files_to_process, agent)
                            if df is not None:
                                agent.add_df(df)
                                agent.save_dfs(dfs)
                                for p in new_paths:
                                    agent.save_file_name(p)
                                st.rerun()
                        except Exception as err:
                            st.error(f"本地文件读取失败：{err}")
    
    dfs = agent.load_dfs()
    if dfs is not None and len(dfs) >= 2:
        load_concat_file(dfs, agent)


def loading_basic_info(agent):
    
    df = agent.load_df()
    if df is not None:
        r, c = df.shape
        missing = int(df.isnull().sum().sum())
        col1, col2, col3 = st.columns(3)
        col1.metric("行数", r)
        col2.metric("列数", c)
        col3.metric("缺失值总数", missing)

        dtype_info = pd.DataFrame({
            "列名": df.columns,
            "类型": df.dtypes.astype(str),
            "非空": df.count().values,
            "缺失%": (df.isnull().mean() * 100).round(2).values,
        }).reset_index(drop=True)

        selected_index = sac.tabs([
            sac.TabsItem(label='数据类型概览'),
            sac.TabsItem(label='数据预览'),
        ],color='#5980AE',)

        if selected_index == "数据类型概览":
            st.dataframe(dtype_info, use_container_width=True)
        elif selected_index == "数据预览":
            if st.button("🎲 随机抽样"):
                display_df = df.sample(10)
                st.dataframe(display_df, use_container_width=True)
            else:
                st.dataframe(df.head(10), use_container_width=True)


def loading_chat(agent, auto=False) -> None:

    df = agent.load_df()
    if df is None:
        return

    with st.chat_message("assistant"):
        st.write(
            "我是 Autostat 数据分析助手，很高兴为您服务！\n\n"
            "请先上传您的数据文件，上传完成后，您可以在下方和我对话，也可以直接点击按钮解析数据含义。"
        )
        analyze_btn = st.button("🔍 解析含义")
        result_placeholder = st.empty()
        
    # 渲染历史对话
    chat_history = agent.load_memory()

    for idx, entry in enumerate(chat_history):
        bubble = st.chat_message(entry["role"])
        content = entry["content"]
        if isinstance(content, str):
            bubble.write(content)

    already_generated = any(
        entry["role"] == "assistant" and "含义" in str(entry["content"])
        for entry in chat_history
    )

    if analyze_btn or (auto and not already_generated):
        st.chat_message("user").write("请帮我解析数据含义")
        agent.add_memory({"role": "user", "content": "请帮我解析数据含义"})
        with st.spinner("分析中..."):
            desc = agent.do_data_description(df)

        agent.finish_auto()
        st.chat_message("assistant").write(desc)
        agent.add_memory({"role": "assistant", "content": desc})
        st.rerun()

    # 用户自定义输入
    user_input = st.chat_input("请输入需求，例如“帮我分析xx列”")
    if user_input:
        st.chat_message("user").write(user_input)
        agent.add_memory({"role": "user", "content": user_input})
        with st.spinner("处理中…"):
            reply = agent.do_data_description(df, user_input)

        st.chat_message("assistant").write(reply)
        agent.add_memory({"role": "assistant", "content": reply})
        st.rerun()


if __name__ == "__main__":

    agent = st.session_state.data_loading_agent
    planner = st.session_state.planner_agent
    auto = planner.loading_auto

    if st.session_state.auto_mode == True:
        if (agent.finish_auto_task == True and planner.switched_prep == False) or planner.loading_auto == False:
            planner.finish_loading_auto()
            st.switch_page("workflow/preprocessing/preprocessing_render.py")

    c1,c2 = st.columns(2)
    with c1:
        st.title("数据导入")
    with c2:
        st.write("")  
        st.write("")  
        sac.buttons([
            sac.ButtonsItem(label='Github', icon='github', href='https://github.com/Automated-Statistician/AutoSTAT'),
            sac.ButtonsItem(label='Doc', icon=sac.BsIcon(name='bi bi-file-earmark-post-fill', size=16), href='https://automated-statistician.github.io/autostatdoc.github.io/'),
        ], align='end', color='dark', variant='filled', index=None)
    st.markdown("---")

    c = st.columns(3)
    with c[0].expander('数据上传', True):
        loading_data_file(agent)
    with c[0].expander('数据展示', True):
        loading_basic_info(agent)
    with c[1].expander('参考资料pdf/docx上传', True):
        loading_reference_docs(agent)
    with c[2].expander('数据建议', True):
        loading_chat(agent, auto)
