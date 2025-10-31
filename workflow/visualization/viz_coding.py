import time
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from stqdm import stqdm
from streamlit_ace import st_ace
import streamlit_antd_components as sac

from utils.sanitize_code import sanitize_code


def vis_code_gen(agent, debug = False, auto = False) -> None:

    df = agent.load_df()
    suggest = agent.load_suggestion()
    user_input = agent.load_user_input()

    chat_history = agent.load_memory()
    already_generated = any(
        entry["role"] == "assistant" and "训练脚本已更新！请重新运行代码！" in str(entry["content"])
        for entry in chat_history
    )

    if suggest is not None:
        if debug == True or (auto and not already_generated):
            with st.spinner("可视化 Agent 正在编写脚本..."):
                raw = agent.code_generation(
                    df.head().to_string(),
                    suggest,
                )
                code = sanitize_code(raw)
                agent.save_code(code)
            st.chat_message("assistant").write("训练脚本已更新！请重新运行代码！")
            agent.add_memory({"role": "assistant", "content": "训练脚本已更新！请重新运行代码！"})
            st.rerun()
        
        analyze_btn = st.button("🔧 生成可视化代码", key="viz_code")
        if analyze_btn:
            with st.spinner("可视化 Agent 正在编写脚本..."):
                raw = agent.code_generation(
                    df.head().to_string(),
                    suggest,
                )
                code = sanitize_code(raw)
                agent.save_code(code)
            st.chat_message("assistant").write("训练脚本已更新！请重新运行代码！")
            agent.add_memory({"role": "assistant", "content": "训练脚本已更新！请重新运行代码！"})
            st.rerun()
            

def vis_execution(agent, auto = False):

    df = agent.load_df()

    exec_ns = {
        "df": df,
        "np": np,
        "pd": pd,
        "px": px,
        "go": go,
    }

    code = agent.load_code()
    edited = st_ace(
            value=code,
            height=450,
            theme="tomorrow_night",
            language="python",
            auto_update=True
        )
    desc_switch = sac.switch(label='附加分析', value=False, off_label='Off')
    if code is not None:
        not_executed = agent.load_fig() == []
        # 当点击按钮，或者 auto=True 且尚未执行过时才执行
        if st.button("▶️ 执行可视化") or (auto and not_executed):
            code = sanitize_code(edited)
            agent.save_code(code)
            try:
                with st.spinner("正在运行可视化脚本..."):
                    exec(code, exec_ns)
            except Exception as exc:
                st.error(f"已记录报错内容，正在为您debug...")
                st.text(traceback.format_exc())
                agent.save_error(traceback.format_exc())
                vis_code_gen(agent, debug=True)
            else:
                fig_dict = exec_ns.get("fig_dict")
                if not fig_dict or not isinstance(fig_dict, dict):
                    st.error(
                        "脚本未写入 `fig_dict` 或格式不正确。请确保末尾赋值 `fig_dict`，且它是一个 {列名: 图表} 的 dict。"
                    )
                    agent.save_error(traceback.format_exc())
                    vis_code_gen(agent, debug=True)
                else:
                    with st.spinner("正在处理可视化结果..."):
                        for col_name, fig in stqdm(fig_dict.items()):
                            dtype_info = ", ".join(
                                f"{c}: {df[c].dtype}" for c in df.columns
                            )
                            if desc_switch == True:
                                desc = agent.desc_fig(fig, dtype_info)
                            else:
                                desc = None
                            agent.add_fig(fig, desc)
                        agent.finish_auto()
                        st.rerun()