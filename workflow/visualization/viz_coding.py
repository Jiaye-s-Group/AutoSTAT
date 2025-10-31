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
        entry["role"] == "assistant" and "Training script updated! Please rerun the code!" in str(entry["content"])
        for entry in chat_history
    )

    if suggest is not None:
        if debug == True or (auto and not already_generated):
            with st.spinner("Visualization Agent is writing script..."):
                raw = agent.code_generation(
                    df.head().to_string(),
                    suggest,
                )
                code = sanitize_code(raw)
                agent.save_code(code)
            st.chat_message("assistant").write("Training script updated! Please rerun the code!")
            agent.add_memory({"role": "assistant", "content": "Training script updated! Please rerun the code!"})
            st.rerun()
        
        analyze_btn = st.button("🔧 Generate Visualization Code", key="viz_code")
        if analyze_btn:
            with st.spinner("Visualization Agent is writing script..."):
                raw = agent.code_generation(
                    df.head().to_string(),
                    suggest,
                )
                code = sanitize_code(raw)
                agent.save_code(code)
            st.chat_message("assistant").write("Training script updated! Please rerun the code!")
            agent.add_memory({"role": "assistant", "content": "Training script updated! Please rerun the code!"})
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
    desc_switch = sac.switch(label='Additional Analysis', value=False, off_label='Off')
    if code is not None:
        not_executed = agent.load_fig() == []
        # Execute when button clicked, or when auto=True and not yet executed
        if st.button("▶️ Run Visualization") or (auto and not_executed):
            code = sanitize_code(edited)
            agent.save_code(code)
            try:
                with st.spinner("Running visualization script..."):
                    exec(code, exec_ns)
            except Exception as exc:
                st.error(f"Error recorded, debugging for you...")
                st.text(traceback.format_exc())
                agent.save_error(traceback.format_exc())
                vis_code_gen(agent, debug=True)
            else:
                fig_dict = exec_ns.get("fig_dict")
                if not fig_dict or not isinstance(fig_dict, dict):
                    st.error(
                        "Script did not write `fig_dict` or format is incorrect. Ensure assignment of `fig_dict` at the end, and that it is a {column name: chart} dict."
                    )
                    agent.save_error(traceback.format_exc())
                    vis_code_gen(agent, debug=True)
                else:
                    with st.spinner("Processing visualization results..."):
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