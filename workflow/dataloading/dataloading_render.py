import os
from typing import List, Optional

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac

from workflow.dataloading.dataloading_core import process_complex_data, load_from_path, load_concat_file, PathFileWrapper


def loading_data_file(agent):

    st.info(
        "💡 Tips:\n"
        "1. Supports uploading multiple data files at once\n"
        "2. Automatically uses large models to analyze and process data\n"
        "3. Supports uploading multiple file formats\n"
    )

    selected_index = sac.tabs([
        sac.TabsItem(label='Local Upload'),
        sac.TabsItem(label='Path Import'),
    ], color='#5980AE',)

    if selected_index == "Local Upload":
        # Click to upload files
        uploaded_files = st.file_uploader(
            "Select new files",
            accept_multiple_files=True,
            help="Drag and drop or click to upload multiple files",
        )

        if uploaded_files:
            current_memory_file_name = agent.load_file_name()
            new_files = [f for f in uploaded_files if f.name not in current_memory_file_name]
            if new_files:
                try:
                    with st.spinner("Processing data..."):
                        df, dfs = process_complex_data(new_files, agent)
                    if df is not None:
                        agent.add_df(df)
                        agent.save_dfs(dfs)
                        for f in new_files:
                            agent.save_file_name(f.name)
                        st.rerun()
                except Exception as err:
                    st.error(f"Loading failed: {err}")

    elif selected_index == "Path Import":
        # Upload file from path
        raw_paths = st.text_area(
            "Import data from path (one file path per line)",
            placeholder=    "C:\\data\\iris.names\nC:\\data\\iris.data",
            height=100
        )

        if st.button("Import Data from Path", use_container_width=True):
            if raw_paths:

                path_list = [p.strip().strip("'\"") for p in raw_paths.strip().split('\n') if p.strip()]
                
                valid_paths = [p for p in path_list if os.path.exists(p)]
                invalid_paths = [p for p in path_list if not os.path.exists(p)]

                if invalid_paths:
                    st.warning(f"Path does not exist, skipped:\n- " + "\n- ".join(invalid_paths))

                if not valid_paths:
                    st.error("No valid local file paths found.")
                else:
                    current_memory_file_name = agent.load_file_name()
                    new_paths = [p for p in valid_paths if p not in current_memory_file_name]

                    if not new_paths:
                        st.info("All specified path files have been loaded.")
                    else:
                        files_to_process = [PathFileWrapper(p) for p in new_paths]
                        try:
                            with st.spinner("Processing data..."):
                                df, dfs = process_complex_data(files_to_process, agent)
                            if df is not None:
                                agent.add_df(df)
                                agent.save_dfs(dfs)
                                for p in new_paths:
                                    agent.save_file_name(p)
                                st.rerun()
                        except Exception as err:
                            st.error(f"Failed to read local file: {err}")
    
    dfs = agent.load_dfs()
    if dfs is not None and len(dfs) >= 2:
        load_concat_file(dfs, agent)


def loading_basic_info(agent):
    
    df = agent.load_df()
    if df is not None:
        r, c = df.shape
        missing = int(df.isnull().sum().sum())
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", r)
        col2.metric("Columns", c)
        col3.metric("Total Missing Values", missing)

        dtype_info = pd.DataFrame({
            "Column Name": df.columns,
            "Type": df.dtypes.astype(str),
            "Non-null": df.count().values,
            "Missing %": (df.isnull().mean() * 100).round(2).values,
        }).reset_index(drop=True)

        selected_index = sac.tabs([
            sac.TabsItem(label='Data Type Overview'),
            sac.TabsItem(label='Data Preview'),
        ],color='#5980AE',)

        if selected_index == "Data Type Overview":
            st.dataframe(dtype_info, use_container_width=True)
        elif selected_index == "Data Preview":
            if st.button("🎲 Random Sampling"):
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
            "I am the Autostat data analysis assistant, pleased to serve you!\n\n"
            "Please upload your data file first. After the upload is complete, you can chat with me below or directly click the button to analyze the data meaning."
        )
        analyze_btn = st.button("🔍 Analyze Meaning")
        result_placeholder = st.empty()
        
    # Render chat history
    chat_history = agent.load_memory()

    for idx, entry in enumerate(chat_history):
        bubble = st.chat_message(entry["role"])
        content = entry["content"]
        if isinstance(content, str):
            bubble.write(content)

    already_generated = any(
        entry["role"] == "assistant" and 'meaning' in str(entry["content"]).lower()
        for entry in chat_history
    )

    if analyze_btn or (auto and not already_generated):
        st.chat_message("user").write("Please help me analyze the meaning of the data")
        agent.add_memory({"role": "user", "content": "Please help me analyze the meaning of the data"})
        with st.spinner("Analyzing..."):
            desc = agent.do_data_description(df)

        agent.finish_auto()
        st.chat_message("assistant").write(desc)
        agent.add_memory({"role": "assistant", "content": desc})
        st.rerun()

    # User custom input
    user_input = st.chat_input("Please enter your request, e.g., 'Help me analyze xx column'")
    if user_input:
        st.chat_message("user").write(user_input)
        agent.add_memory({"role": "user", "content": user_input})
        with st.spinner("Processing..."):
            reply = agent.do_data_description(df, user_input)

        st.chat_message("assistant").write(reply)
        agent.add_memory({"role": "assistant", "content": reply})
        st.rerun()


if __name__ == "__main__":

    agent = st.session_state.data_loading_agent
    planner = st.session_state.planner_agent
    auto = planner.loading_auto

    if st.session_state.auto_mode == True:
        if (agent.finish_auto_task == True and planner.switched_prep == False) or planner.prep_auto == False:
            planner.finish_loading_auto()
            st.switch_page("workflow/preprocessing/preprocessing_render.py")

    c1,c2 = st.columns(2)
    with c1:
        st.title("Data Loading")
    with c2:
        st.write("")  
        st.write("")  
        sac.buttons([
            sac.ButtonsItem(label='Github', icon='github', href='https://github.com/Automated-Statistician/AutoSTAT/tree/eng_version'),
            sac.ButtonsItem(label='Doc', icon=sac.BsIcon(name='bi bi-file-earmark-post-fill', size=16), href='https://automated-statistician.github.io/autostatdoc.github.io/'),
        ], align='end', color='dark', variant='filled', index=None)
    st.markdown("---")

    c = st.columns(2)
    with c[0].expander('Data Upload', True):
        loading_data_file(agent)
    with c[1].expander('Data Suggestions', True):
        loading_chat(agent, auto)
    with c[0].expander('Data Display', True):
        loading_basic_info(agent)

