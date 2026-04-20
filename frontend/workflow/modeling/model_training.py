import base64
import gzip
import importlib
import json
import pickle
import traceback

import lightgbm
import numpy as np
import pandas as pd
import streamlit as st
import xgboost
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.sanitize_code import sanitize_code, to_json_serializable


def _code_requires_torch(code: str) -> bool:
    text = (code or "").lower()
    keywords = [
        "import torch",
        "from torch",
        "torch.",
        "torch.nn",
        "torchvision",
        "torch.utils.data",
        "nn.",
        "optim.",
        "tensor(",
        ".backward(",
        "dataloader",
    ]
    return any(k in text for k in keywords)


def _load_optional_torch_modules(code: str):
    if not _code_requires_torch(code):
        return None, None, False

    try:
        torch_module = importlib.import_module("torch")
    except ModuleNotFoundError:
        return None, None, True

    try:
        torchvision_module = importlib.import_module("torchvision")
    except ModuleNotFoundError:
        torchvision_module = None

    return torch_module, torchvision_module, False


def train_download_model(agent) -> None:
    gz_bytes = None

    load_best_model_gz_bytes = getattr(agent, "load_best_model_gz_bytes", None)
    if callable(load_best_model_gz_bytes):
        gz_bytes = load_best_model_gz_bytes()

    if not gz_bytes:
        load_best_model = getattr(agent, "load_best_model", None)
        model_obj = load_best_model() if callable(load_best_model) else None
        if model_obj is not None:
            try:
                gz_bytes = gzip.compress(pickle.dumps(model_obj))
                save_best_model_gz_bytes = getattr(agent, "save_best_model_gz_bytes", None)
                if callable(save_best_model_gz_bytes):
                    save_best_model_gz_bytes(gz_bytes)
            except Exception:
                gz_bytes = None

    if not gz_bytes:
        return

    st.download_button(
        label="下载训练模型（best_model.pkl.gz）",
        data=gz_bytes,
        file_name="best_model.pkl.gz",
        mime="application/gzip",
        key="download_best_model",
    )


def train_execution(agent):
    code = agent.load_code()
    df = agent.load_df()

    torch_module, torchvision_module, missing_torch = _load_optional_torch_modules(code)

    if missing_torch:
        st.error("当前训练代码使用了 PyTorch，但环境未安装 `torch`。")
        st.code(
            "python -m pip install torch torchvision "
            "-i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple "
            "--timeout 120 --retries 10"
        )
        return

    exec_ns = {
        "df": df,
        "np": np,
        "pd": pd,
        "train_test_split": train_test_split,
        "StandardScaler": StandardScaler,
        "LinearRegression": LinearRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "RandomForestClassifier": RandomForestClassifier,
        "xgboost": xgboost,
        "lightgbm": lightgbm,
    }

    if torch_module is not None:
        exec_ns["torch"] = torch_module
    if torchvision_module is not None:
        exec_ns["torchvision"] = torchvision_module

    try:
        with st.spinner("正在运行程序..."):
            exec(code, exec_ns)
    except Exception:
        st.error("出现报错，请重新生成调试后的建模代码。")
        st.text(traceback.format_exc())
        modeling_code_gen(agent, debug=True)
        return

    result_dict = exec_ns.get("result_dict")
    if result_dict is None:
        st.error("脚本未写入 `result_dict`。请确保脚本末尾赋值 `result_dict`。")
        return

    artifacts = result_dict.get("artifacts", {})
    best_model_b64 = artifacts.pop("best_model_b64", None)
    result_dict.pop("artifact_warning", None)
    if not artifacts:
        result_dict.pop("artifacts", None)

    serializable = to_json_serializable(result_dict)
    try:
        result_json = json.dumps(serializable, ensure_ascii=False)
    except Exception:
        result_json = json.dumps(serializable, default=str, ensure_ascii=False)

    with st.spinner("正在格式化训练结果..."):
        formatted = agent.result_format_prompt(result_json)
        agent.save_modeling_result(formatted)

    if best_model_b64:
        gz_bytes = base64.b64decode(best_model_b64)
        try:
            agent.save_best_model_gz_bytes(gz_bytes)
            model_obj = pickle.loads(gzip.decompress(gz_bytes))
            agent.save_best_model(model_obj)
            st.success("最佳模型已加载到内存，可用于后续推理。")
        except Exception as exc:
            st.error(f"加载模型失败：{exc}")


def modeling_code_gen(agent, debug=False, auto=False) -> None:
    df = agent.load_df()
    suggest = agent.load_suggestion()
    summary_4 = st.session_state.get("summary_4") or st.session_state.get("modeling_summary_4")
    workflow_code = ""
    if isinstance(summary_4, dict):
        workflow_code = str(summary_4.get("code") or "").strip()

    chat_history = agent.load_memory()
    already_generated = any(
        entry["role"] == "assistant" and "训练脚本已更新，请重新运行代码！" in str(entry["content"])
        for entry in chat_history
    )

    if workflow_code:
        analyze_btn = st.button("🎯 生成模型建议代码", key="modeling_code")
        if analyze_btn:
            agent.save_code(workflow_code)
            st.chat_message("assistant").write("建模代码已完成加载，请在下方执行。")
            agent.add_memory(
                {"role": "assistant", "content": "建模代码已完成加载，请在下方执行。"}
            )
            st.rerun()
        return

    if suggest is not None:
        if debug or (auto and not already_generated):
            with st.spinner("建模 Agent 正在编写脚本..."):
                raw = agent.code_generation(
                    df.head(10).to_string(),
                    suggest,
                )
                code = sanitize_code(raw)
                agent.save_code(code)

            st.chat_message("assistant").write("训练脚本已更新，请重新运行代码！")
            agent.add_memory({"role": "assistant", "content": "训练脚本已更新，请重新运行代码！"})
            st.rerun()

        analyze_btn = st.button("🎯 生成模型建议代码", key="modeling_code")
        if analyze_btn:
            with st.spinner("向 LLM 请求生成建模脚本..."):
                raw = agent.code_generation(
                    df.head(10).to_string(),
                    suggest,
                )
                code = sanitize_code(raw)
                agent.save_code(code)

            st.chat_message("assistant").write("训练脚本已更新，请重新运行代码！")
            agent.add_memory({"role": "assistant", "content": "训练脚本已更新，请重新运行代码！"})
            st.rerun()
