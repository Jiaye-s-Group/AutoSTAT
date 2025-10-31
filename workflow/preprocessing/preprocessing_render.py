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

    # Display basic statistics
    r, c = df.shape
    missing = int(df.isnull().sum().sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", r)
    col2.metric("Columns", c)
    col3.metric("Total Missing Values", missing)

    dtype_info = pd.DataFrame({
        'Column Name': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-null Count': df.count().values,
        'Missing Ratio (%)': (df.isnull().mean() * 100).round(2).values,
    })
    dtype_info = dtype_info.reset_index(drop=True)
    st.dataframe(dtype_info, use_container_width=True)


def prep_execution(agent, auto=False):
    ''' 
    Preprocess training data
    '''

    code = agent.load_code()
    df = agent.load_df()

    process_df = prep_meta_execution(agent, code, df, auto=auto)


def prep_result(agent):
    
    process_df = agent.load_processed_df()
    df = agent.load_df()
    
    if process_df is not None:
        st.write("Raw data preview:", df.head(10))
        st.write("Processed data preview:", process_df.head(10))
            
        csv_buffer = io.StringIO()
        process_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        
        st.download_button(
            label="⬇️ Download processed data",
            data=csv_bytes,
            file_name="processed_data.csv",
            mime="text/csv",
        )


def prep_chat(agent, auto=False):
    """Render conversational suggestion area"""

    with st.chat_message("assistant"):
        st.write("I am the Autostat data analysis assistant, pleased to serve you!\n\n"
            "You can enter preprocessing requirements below or directly click the button to get preprocessing suggestions.")
        analyze_btn = st.button("🔍 Preprocessing Recommendation", key='prep_suggest')

    # Render chat history
    chat_history = agent.load_memory()

    for idx, entry in enumerate(chat_history):
        bubble = st.chat_message(entry["role"])
        content = entry["content"]
        if isinstance(content, str):
            bubble.write(content)

    already_generated = any(
        entry["role"] == "assistant" and "preprocessing" in str(entry["content"]).lower()
        for entry in chat_history
    )

    # Auto/manual trigger
    if analyze_btn or (auto and not already_generated):

        st.chat_message("user").write("Please give me preprocessing suggestions")
        agent.add_memory({'role': 'user', 'content': "Please give me preprocessing suggestions"})

        with st.spinner("Generating suggestions..."):
            text = agent.get_preprocessing_suggestions()
            agent.save_preprocessing_suggestions(text)
            agent.refine_suggestions(df.head(10).to_string())
        st.chat_message("assistant").write(text)
        agent.add_memory({'role': 'assistant', 'content': text})

    # Natural language interaction
    user_input = st.chat_input("Please enter your question")
    if user_input:
        st.chat_message("user").write(user_input)
        agent.add_memory({'role': 'user', 'content': user_input})
        agent.save_user_input(user_input)
        with st.spinner("Processing..."):
            reply = agent.get_preprocessing_suggestions(user_input)
            agent.save_preprocessing_suggestions(reply)
            agent.refine_suggestions(df.head(10).to_string())
        st.chat_message('assistant').write(reply)
        agent.add_memory({'role': 'assistant', 'content': reply})          


if __name__ == '__main__':

    st.title("Preprocessing")

    st.markdown("---")

    data_loading_agent = st.session_state.data_loading_agent
    df = data_loading_agent.load_df()
    planner = st.session_state.planner_agent
    auto = planner.prep_auto

    if df is None:
        st.warning("⚠️ Please load data on the data import page first")
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
    with c[0].expander('Preprocessing Display', True):
        prep_basic_info(agent)
    with c[1].expander('Preprocessing Suggestions', True):
        prep_chat(agent, auto)
        prep_code_gen(agent, auto=auto)
    with c[0].expander('Preprocessing Execution', code_expand):
        prep_execution(agent, auto)
    with c[0].expander('Preprocessing Results', code_expand):
        prep_result(agent)