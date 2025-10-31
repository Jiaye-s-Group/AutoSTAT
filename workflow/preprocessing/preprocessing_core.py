import time
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


def prep_meta_execution(agent, code, df, auto=False):

    edited = st_ace(
        value=code,
        height=400,
        theme="tomorrow_night",
        language="python",
        auto_update=True
    )

    not_generated = agent.load_processed_df() is None
    
    if code is not None:
        if st.button("▶️ Run Preprocessing") or (auto and not_generated):
            code = sanitize_code(edited)
            agent.save_code(code)

            exec_ns = {
                "df": df,
                "np": np,
                "pd": pd,
                "st": st,
                "SimpleImputer": SimpleImputer,
                "FunctionTransformer": FunctionTransformer,
                "StandardScaler": StandardScaler,
                "MinMaxScaler": MinMaxScaler,
                "RobustScaler": RobustScaler,
                "OneHotEncoder": OneHotEncoder,
                "OrdinalEncoder": OrdinalEncoder,
                "LabelEncoder": LabelEncoder,
                "ColumnTransformer": ColumnTransformer,
                "Pipeline": Pipeline,
            }

            try:
                with st.spinner("Running program..."):
                    exec(code, exec_ns)
            except Exception as exc:
                st.error(f"Error saved, please recall LLM to generate debug code")
                st.text(traceback.format_exc())
                agent.save_error(traceback.format_exc())
                prep_code_gen(agent, debug=True)
            else:
                process_df = exec_ns.get("process_df")
                if process_df is None:
                    st.error(
                        "The script did not write `process_df`. Please ensure the edited script assigns process_df at the end"
                    )
                else:
                    agent.save_processed_df(process_df)
                    agent.finish_auto()
                    st.rerun()
                    return process_df
                    

def prep_code_gen(agent, auto = False, debug = False):

    suggest = agent.load_preprocessing_suggestions()
    df = agent.load_df()

    chat_history = agent.load_memory()
    already_generated = any(
        entry["role"] == "assistant" and "Preprocessing script updated! Please rerun the code!" in str(entry["content"])
        for entry in chat_history
    )

    if suggest is not None:

        if debug == True or (auto and not already_generated):
            with st.spinner("Preprocessing Agent is writing script..."):
                raw = agent.code_generation(
                    df.head(10).to_string(),
                    suggest,
                )
                code = sanitize_code(raw)
                agent.save_code(code)

            st.chat_message("assistant").write("Preprocessing script updated! Please rerun the code!")
            agent.add_memory({"role": "assistant", "content": "Preprocessing script updated! Please rerun the code!"})

            st.rerun()

        analyze_btn = st.button("🔧 Generate Preprocessing Code", key='prep_code')
        if analyze_btn:
            with st.spinner("Requesting LLM to generate preprocessing script..."):
                raw = agent.code_generation(
                    df.head(10).to_string(),
                    suggest,
                )
                code = sanitize_code(raw)
                agent.save_code(code)

            st.chat_message("assistant").write("Preprocessing script updated! Please rerun the code!")
            agent.add_memory({"role": "assistant", "content": "Preprocessing script updated! Please rerun the code!"})

            st.rerun()