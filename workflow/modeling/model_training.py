import importlib
import json
import traceback
import base64
import gzip
import pickle
import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from streamlit_ace import st_ace
import torch
import torchvision
import xgboost
import lightgbm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.sanitize_code import sanitize_code, to_json_serializable


def train_execution(agent):

    code = agent.load_code()
    df = agent.load_df()

    torch = importlib.import_module("torch")
    torchvision = importlib.import_module("torchvision")

    exec_ns = {
        "df": df,
        "np": np,
        "pd": pd,
        "torch": torch,
        "torchvision": torchvision,
        "train_test_split": train_test_split,
        "StandardScaler": StandardScaler,
        "LinearRegression": LinearRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "RandomForestClassifier": RandomForestClassifier,
        "xgboost": xgboost,
        "lightgbm": lightgbm,
    }

    try:
        with st.spinner("Running program..."):
            exec(code, exec_ns)
    except Exception as exc:
        st.error(f"Error saved, please recall LLM to generate debug code")
        # st.error(f"Script execution failed: {exc}")
        st.text(traceback.format_exc())
        agent.save_error(traceback.format_exc())
        modeling_code_gen(agent, debug=True)
    else:
        result_dict = exec_ns.get("result_dict")
        if result_dict is None:
            st.error(
                "The script did not write `result_dict`. Please ensure the edited script assigns result_dict at the end."
            )
        else:
            art = result_dict.get('artifacts', {})
            b64 = art.pop('best_model_b64', None)
            artifact_warning = result_dict.pop('artifact_warning', None)

            if not art:
                result_dict.pop('artifacts', None)

            serializable = to_json_serializable(result_dict)
            try:
                result_json = json.dumps(serializable, ensure_ascii=False)
            except Exception:
                result_json = json.dumps(serializable, default=str, ensure_ascii=False)

            with st.spinner("Requesting LLM to format result as Markdown..."):
                formatted = agent.result_format_prompt(result_json)
                agent.save_modeling_result(formatted)

            if b64:

                gz_bytes = base64.b64decode(b64)

                try:
                    agent.save_best_model_gz_bytes(gz_bytes)
                    model_obj = pickle.loads(gzip.decompress(gz_bytes))
                    st.success("Best model loaded into memory, available for instant inference (example).")
                    agent.save_best_model(model_obj)
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    
        
def modeling_code_gen(agent, debug = False, auto = False, ) -> None:

    df = agent.load_df()
    suggest = agent.load_suggestion()
    print(suggest)
    chat_history = agent.load_memory()
    already_generated = any(
        entry["role"] == "assistant" and "Training script updated! Please rerun the code!" in str(entry["content"])
        for entry in chat_history
    )

    if suggest is not None:
        if debug == True or (auto and not already_generated):
            with st.spinner("Modeling Agent is generating training script..."):
                raw = agent.code_generation(
                    df.head().to_string(),
                    suggest,
                )
                code = sanitize_code(raw)
                agent.save_code(code)
            st.chat_message("assistant").write("Training script updated! Please rerun the code!")
            agent.add_memory({"role": "assistant", "content": "Training script updated! Please rerun the code!"})
            st.rerun()        

        analyze_btn = st.button("🔧 Generate Modeling Code", key='modeling_code')
        if analyze_btn:
            with st.spinner("Modeling Agent is generating training script..."):
                raw = agent.code_generation(
                    df.head().to_string(),
                    suggest,
                )
                code = sanitize_code(raw)
                agent.save_code(code)
            st.chat_message("assistant").write("Training script updated! Please rerun the code!")
            agent.add_memory({"role": "assistant", "content": "Training script updated! Please rerun the code!"})
            st.rerun()

    
def train_download_model(agent):

    model = agent.load_best_model_gz_bytes()
    if model is not None:
        st.download_button(
        label="⬇️ Download the Best Model",
        data=model,
        file_name="best_model.pkl.gz",
        mime="application/gzip"
        )