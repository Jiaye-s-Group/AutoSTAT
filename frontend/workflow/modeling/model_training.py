import base64
from contextlib import contextmanager
import gzip
import importlib
import pickle
import traceback
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import streamlit as st
import xgboost
from core.modeling_table_utils import (
    build_model_comparison_table_bundle,
    build_modeling_execution_summary_markdown,
)
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.sanitize_code import sanitize_code, to_json_serializable


def _show_execution_error(message: str, error_text: str) -> None:
    st.error(message)
    if error_text:
        st.code(error_text, language="text")


def _update_modeling_summary_state(agent, code: str, formatted: str, table_bundle: dict[str, Any]) -> None:
    current_summary = st.session_state.get("summary_4") or st.session_state.get("modeling_summary_4")
    if isinstance(current_summary, dict):
        updated_summary = dict(current_summary)
    else:
        updated_summary = {
            "title": "建模分析",
            "desc": "",
        }

    updated_summary["code"] = code or updated_summary.get("code", "")
    updated_summary["result"] = formatted
    updated_summary["table_title"] = table_bundle.get("title", "")
    updated_summary["table_markdown"] = table_bundle.get("markdown_table", "")
    updated_summary["table_html"] = table_bundle.get("html_table", "")

    st.session_state.summary_4 = updated_summary
    st.session_state.modeling_summary_4 = updated_summary
    st.session_state.modeling_result_from_summary_4 = formatted
    st.session_state.abstract_4 = formatted
    st.session_state.modeling_abstract_4 = formatted
    agent.save_modeling_result(formatted)


def _build_modeling_result_summary(
    current_summary: Any,
    serializable_result: Any,
    table_bundle: dict[str, Any],
) -> str:
    _ = current_summary
    return build_modeling_execution_summary_markdown(serializable_result, table_bundle)


def _load_workflow_modeling_code(agent, workflow_code: str) -> None:
    agent.save_code(workflow_code)
    message = "建模代码已生成，请点击下方执行。"
    st.chat_message("assistant").write(message)
    agent.add_memory({"role": "assistant", "content": message})


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


def _ensure_lightgbm_quiet_params(params: Any) -> Any:
    if not isinstance(params, dict):
        return params

    quiet_params = dict(params)
    if "verbosity" not in quiet_params and "verbose" not in quiet_params:
        quiet_params["verbosity"] = -1
    return quiet_params


@contextmanager
def _temporary_quiet_lightgbm():
    patched_attrs: list[tuple[Any, str, Any]] = []

    try:
        for estimator_name in ("LGBMRegressor", "LGBMClassifier", "LGBMRanker"):
            estimator_cls = getattr(lightgbm, estimator_name, None)
            if estimator_cls is None or not hasattr(estimator_cls, "__init__"):
                continue

            original_init = estimator_cls.__init__

            def quiet_init(self, *args, __orig_init=original_init, **kwargs):
                if "verbosity" not in kwargs and "verbose" not in kwargs:
                    kwargs["verbosity"] = -1
                return __orig_init(self, *args, **kwargs)

            quiet_init.__name__ = original_init.__name__
            quiet_init.__qualname__ = original_init.__qualname__
            quiet_init.__wrapped__ = original_init

            setattr(estimator_cls, "__init__", quiet_init)
            patched_attrs.append((estimator_cls, "__init__", original_init))

        for fn_name in ("train", "cv"):
            original_fn = getattr(lightgbm, fn_name, None)
            if not callable(original_fn):
                continue

            def quiet_fn(params, *args, __orig_fn=original_fn, **kwargs):
                return __orig_fn(_ensure_lightgbm_quiet_params(params), *args, **kwargs)

            quiet_fn.__name__ = original_fn.__name__
            quiet_fn.__qualname__ = original_fn.__qualname__
            quiet_fn.__wrapped__ = original_fn

            setattr(lightgbm, fn_name, quiet_fn)
            patched_attrs.append((lightgbm, fn_name, original_fn))

        yield
    finally:
        for target, attr_name, original_value in reversed(patched_attrs):
            setattr(target, attr_name, original_value)


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
            with _temporary_quiet_lightgbm():
                exec(code, exec_ns)
    except Exception:
        error_text = traceback.format_exc()
        agent.save_error(error_text)
        _show_execution_error(
            "出现报错，请重新生成调试后的建模代码。",
            error_text,
        )
        modeling_code_gen(agent, debug=True)
        return

    result_dict = exec_ns.get("result_dict")
    if result_dict is None:
        error_text = "脚本执行完成，但未写入 `result_dict`。请确保脚本末尾赋值 `result_dict = {...}`。"
        agent.save_error(error_text)
        _show_execution_error(
            "脚本未写入 `result_dict`。请确保脚本末尾赋值 `result_dict`。",
            error_text,
        )
        return

    artifacts = result_dict.get("artifacts", {})
    best_model_b64 = artifacts.pop("best_model_b64", None)
    result_dict.pop("artifact_warning", None)
    if not artifacts:
        result_dict.pop("artifacts", None)

    serializable = to_json_serializable(result_dict)

    with st.spinner("正在格式化训练结果..."):
        current_summary = st.session_state.get("summary_4") or st.session_state.get("modeling_summary_4")
        table_bundle = build_model_comparison_table_bundle(
            serializable,
            target=getattr(agent, "load_target", lambda: "")() or "",
            user_input=getattr(agent, "load_user_input", lambda: "")() or "",
            additional_preference=st.session_state.get("add_preference") or "",
        )
        formatted = _build_modeling_result_summary(current_summary, serializable, table_bundle)
        _update_modeling_summary_state(agent, code, formatted, table_bundle)

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
        if debug:
            _load_workflow_modeling_code(agent, workflow_code)
            return

        analyze_btn = st.button("🎯 生成模型建议代码", key="modeling_code_from_workflow")
        if analyze_btn:
            _load_workflow_modeling_code(agent, workflow_code)
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

        analyze_btn = st.button("🎯 生成模型建议代码", key="modeling_code_from_suggest")
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
