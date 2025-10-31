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
        if st.button("▶️ 执行预处理") or (auto and not_generated):
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
                with st.spinner("正在运行程序..."):
                    exec(code, exec_ns)
            except Exception as exc:
                st.error(f"已保存报错，正在重新调用llm生成代码debug")
                st.text(traceback.format_exc())
                agent.save_error(traceback.format_exc())
                prep_code_gen(agent, debug=True)
            else:
                process_df = exec_ns.get("process_df")
                if process_df is None:
                    st.error(
                        "脚本未写入 `process_df`。请确保编辑后的脚本在末尾赋值 process_df"
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
        entry["role"] == "assistant" and "预处理脚本已更新！请重新运行代码！" in str(entry["content"])
        for entry in chat_history
    )

    if suggest is not None:

        if debug == True or (auto and not already_generated):
            with st.spinner("预处理 Agent 正在编写脚本..."):
                raw = agent.code_generation(
                    df.head(10).to_string(),
                    suggest,
                )
                code = sanitize_code(raw)
                agent.save_code(code)

            st.chat_message("assistant").write("预处理脚本已更新！请重新运行代码！")
            agent.add_memory({"role": "assistant", "content": "预处理脚本已更新！请重新运行代码！"})

            st.rerun()

        analyze_btn = st.button("🔧 生成预处理代码", key='prep_code')
        if analyze_btn:
            with st.spinner("向 LLM 请求生成预处理脚本..."):
                raw = agent.code_generation(
                    df.head(10).to_string(),
                    suggest,
                )
                code = sanitize_code(raw)
                agent.save_code(code)

            st.chat_message("assistant").write("预处理脚本已更新！请重新运行代码！")
            agent.add_memory({"role": "assistant", "content": "预处理脚本已更新！请重新运行代码！"})

            st.rerun()