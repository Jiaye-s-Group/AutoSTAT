import traceback

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from streamlit_ace import st_ace

from utils.sanitize_code import sanitize_code


def _show_execution_error(message: str, error_text: str) -> None:
    st.error(message)
    if error_text:
        st.code(error_text, language="text")


def prep_meta_execution(agent, code, df, auto=False):
    if not st.session_state.get("prep_code_visible") or code is None:
        return None

    edited = st_ace(
        value=code,
        height=400,
        theme="tomorrow_night",
        language="python",
        auto_update=True,
    )

    not_generated = agent.load_processed_df() is None

    if code is not None:
        execute_clicked = st.button("▶️ 执行预处理") or (auto and not_generated)
        if execute_clicked:
            summary_2 = st.session_state.get("summary_2")
            if isinstance(summary_2, dict) and summary_2.get("processed_df"):
                st.session_state.prep_result_from_summary_2 = summary_2["processed_df"]

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
                with st.spinner("正在运行预处理脚本..."):
                    exec(code, exec_ns)
            except Exception:
                error_text = traceback.format_exc()
                agent.save_error(error_text)
                _show_execution_error(
                    "出现报错，正在重新调用 LLM 生成调试后的代码。",
                    error_text,
                )
                prep_code_gen(agent, debug=True)
            else:
                process_df = exec_ns.get("process_df")
                if process_df is None:
                    message = "脚本未写入 `process_df`。请重新生成代码，并确保脚本末尾为 `process_df` 赋值。"
                    agent.save_error(message)
                    _show_execution_error(message, message)
                    prep_code_gen(agent, debug=True)
                else:
                    if not isinstance(process_df, pd.DataFrame):
                        if isinstance(process_df, np.ndarray):
                            process_df = pd.DataFrame(process_df)
                        else:
                            message = (
                                f"期望 pandas.DataFrame 或 numpy.ndarray，收到 {type(process_df)}。"
                                "请重新生成代码。"
                            )
                            agent.save_error(message)
                            _show_execution_error(message, message)
                            prep_code_gen(agent, debug=True)
                            return None

                    agent.save_processed_df(process_df)
                    agent.finish_auto()
                    st.rerun()
                    return process_df


def prep_code_gen(agent, auto=False, debug=False):
    suggest = agent.load_preprocessing_suggestions()
    df = agent.load_df()

    chat_history = agent.load_memory()
    already_generated = any(
        entry["role"] == "assistant" and "预处理脚本已更新，请重新运行代码！" in str(entry["content"])
        for entry in chat_history
    )

    summary_2 = st.session_state.get("summary_2")
    workflow_code = ""
    if isinstance(summary_2, dict):
        workflow_code = str(summary_2.get("code") or "").strip()

    if workflow_code:
        analyze_btn = st.button("🔡 生成预处理代码", key="prep_code")
        if analyze_btn or (auto and not already_generated):
            agent.save_code(workflow_code)
            st.session_state.prep_code_visible = True
            st.chat_message("assistant").write("预处理代码已从工作流加载，请在下方执行。")
            agent.add_memory(
                {"role": "assistant", "content": "预处理代码已从工作流加载，请在下方执行。"}
            )
            st.rerun()
        return

    if suggest is not None:
        if debug or (auto and not already_generated):
            with st.spinner("预处理 Agent 正在编写脚本..."):
                raw = agent.code_generation(
                    df.head(10).to_string(),
                    suggest,
                )
                code = sanitize_code(raw)
                agent.save_code(code)
                st.session_state.prep_code_visible = True

            st.chat_message("assistant").write("预处理脚本已更新，请重新运行代码！")
            agent.add_memory({"role": "assistant", "content": "预处理脚本已更新，请重新运行代码！"})
            st.rerun()

        analyze_btn = st.button("🔡 生成预处理代码", key="prep_code")
        if analyze_btn:
            with st.spinner("向 LLM 请求生成预处理脚本..."):
                raw = agent.code_generation(
                    df.head(10).to_string(),
                    suggest,
                )
                code = sanitize_code(raw)
                agent.save_code(code)
                st.session_state.prep_code_visible = True

            st.chat_message("assistant").write("预处理脚本已更新，请重新运行代码！")
            agent.add_memory({"role": "assistant", "content": "预处理脚本已更新，请重新运行代码！"})
            st.rerun()
