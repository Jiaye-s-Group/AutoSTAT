import base64
import gzip
import importlib
import pickle
import traceback
from typing import Any

import streamlit as st
from core.bounded_code_execution import (
    MODELING_TIMEOUT_SECONDS,
    run_bounded_safe_exec,
)
from core.modeling_table_utils import (
    build_model_comparison_table_bundle,
    build_modeling_execution_summary_markdown,
)
from core.modeling_contract import (
    build_analysis_contract,
    format_contract_violations,
    has_primary_analysis_outputs,
    validate_code_against_contract,
    validate_modeling_result,
)
from core.modeling_runtime_compat import validate_modeling_runtime_compatibility
from core.safe_code import restricted_pickle_loads

from utils.i18n import bt, get_language
from utils.sanitize_code import to_json_serializable
from utils.suggestion_state import get_suggestion_state, record_execution_failure
from utils.workflow_state import (
    current_dataset_fingerprint,
    invalidate_from,
    record_stage_status,
    stable_fingerprint,
)


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
            "title": bt("建模分析", "Modeling Analysis"),
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
    message = bt(
        "建模代码已生成，请点击下方执行。",
        "Modeling code has been generated. Run it below.",
    )
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
        label=bt("下载训练模型（best_model.pkl.gz）", "Download Trained Model (best_model.pkl.gz)"),
        data=gz_bytes,
        file_name="best_model.pkl.gz",
        mime="application/gzip",
        key="download_best_model",
    )


def _record_modeling_failure(agent, error_text: str) -> None:
    agent.save_error(error_text)
    record_execution_failure(get_suggestion_state(st.session_state, "modeling"), error_text)
    if agent.load_modeling_result() is None:
        record_stage_status(
            st.session_state,
            "modeling",
            "failed",
            input_fingerprint=stable_fingerprint(
                st.session_state.get("analysis_dataset_fingerprint")
                or current_dataset_fingerprint(st.session_state),
                st.session_state.get("modeling_analysis_contract") or {},
            ),
            error=error_text,
        )


def _analysis_contract_for_execution(agent, df) -> dict[str, Any]:
    phase1_ctx = st.session_state.get("_model_phase1_ctx")
    if isinstance(phase1_ctx, dict):
        from workflows.modeling import ensure_analysis_contract

        phase1_ctx = ensure_analysis_contract(phase1_ctx)
        st.session_state._model_phase1_ctx = phase1_ctx
        st.session_state.modeling_analysis_contract = phase1_ctx.get("analysis_contract") or {}

    analysis_contract = st.session_state.get("modeling_analysis_contract") or {}
    if not analysis_contract:
        analysis_contract = build_analysis_contract(
            target=getattr(agent, "load_target", lambda: "")() or "",
            columns=list(map(str, df.columns)),
            user_input=getattr(agent, "load_user_input", lambda: "")() or "",
            add_preference=st.session_state.get("add_preference") or "",
            task_type=getattr(agent, "load_task_type", lambda: "auto")() or "auto",
        )
        st.session_state.modeling_analysis_contract = analysis_contract
    return analysis_contract


def train_execution(agent) -> bool:
    code = agent.load_code()
    df = agent.load_df()

    analysis_contract = _analysis_contract_for_execution(agent, df)
    if not analysis_contract.get("valid", False):
        _record_modeling_failure(
            agent,
            format_contract_violations(list(analysis_contract.get("issues") or [])),
        )
        return False

    static_contract_issues = validate_code_against_contract(
        code=code,
        contract=analysis_contract,
    )
    if static_contract_issues:
        _record_modeling_failure(
            agent,
            format_contract_violations(static_contract_issues),
        )
        return False

    compatibility_issues = validate_modeling_runtime_compatibility(code, n_rows=len(df))
    if compatibility_issues:
        _record_modeling_failure(
            agent,
            "Modeling runtime compatibility validation failed:\n- "
            + "\n- ".join(compatibility_issues),
        )
        return False

    _, _, missing_torch = _load_optional_torch_modules(code)

    if missing_torch:
        _record_modeling_failure(
            agent,
            bt(
                "当前训练代码使用了 PyTorch，但环境未安装 `torch`。",
                "The current training code uses PyTorch, but `torch` is not installed in this environment.",
            ),
        )
        return False

    try:
        with st.spinner(bt("正在运行程序...", "Running the program...")):
            execution_result = run_bounded_safe_exec(
                kind="modeling",
                code=code,
                dataframe=df,
                timeout_seconds=MODELING_TIMEOUT_SECONDS,
            )
        if not execution_result["is_success"]:
            raise RuntimeError(str(execution_result["error"]))
    except Exception:
        error_text = traceback.format_exc()
        _record_modeling_failure(agent, error_text)
        return False

    result_dict = execution_result.get("value")
    if not isinstance(result_dict, dict):
        error_text = bt(
            "脚本执行完成，但未生成字典类型的 `result_dict`。请确保脚本末尾赋值 `result_dict = {...}`。",
            "The script finished without a dictionary `result_dict`. Make sure it assigns `result_dict = {...}` at the end.",
        )
        _record_modeling_failure(agent, error_text)
        return False

    contract_issues = validate_modeling_result(
        code=code,
        result_json=result_dict,
        contract=analysis_contract,
    )
    if contract_issues:
        error_text = format_contract_violations(contract_issues)
        _record_modeling_failure(agent, error_text)
        return False

    result_dict = dict(result_dict)
    artifacts = result_dict.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        result_dict.pop("artifacts", None)
    else:
        artifacts = dict(artifacts)
        result_dict["artifacts"] = artifacts
    best_model_b64 = artifacts.pop("best_model_b64", None)
    result_dict.pop("artifact_warning", None)
    if not artifacts:
        result_dict.pop("artifacts", None)

    serializable = to_json_serializable(result_dict)

    try:
        with st.spinner(bt("正在格式化训练结果...", "Formatting training results...")):
            current_summary = st.session_state.get("summary_4") or st.session_state.get("modeling_summary_4")
            table_bundle = build_model_comparison_table_bundle(
                serializable,
                target=getattr(agent, "load_target", lambda: "")() or "",
                user_input=getattr(agent, "load_user_input", lambda: "")() or "",
                additional_preference=st.session_state.get("add_preference") or "",
                language=get_language(),
            )
            association_primary_result = (
                str(analysis_contract.get("task_type") or "").strip().lower() == "association_inference"
                and has_primary_analysis_outputs(serializable)
            )
            if not table_bundle.get("has_table") and not association_primary_result:
                error_text = format_contract_violations(
                    [
                        "result_dict must contain model metrics or primary analysis tables that can produce reportable results."
                    ]
                )
                _record_modeling_failure(agent, error_text)
                return False
            formatted = _build_modeling_result_summary(current_summary, serializable, table_bundle)
            _update_modeling_summary_state(agent, code, formatted, table_bundle)
    except Exception:
        error_text = traceback.format_exc()
        _record_modeling_failure(agent, error_text)
        return False

    if best_model_b64:
        try:
            gz_bytes = base64.b64decode(best_model_b64, validate=True)
            agent.save_best_model_gz_bytes(gz_bytes)
            model_obj = restricted_pickle_loads(gzip.decompress(gz_bytes))
            agent.save_best_model(model_obj)
            st.success(
                bt(
                    "最佳模型已加载到内存，可用于后续推理。",
                    "The best model has been loaded into memory and is ready for inference.",
                )
            )
        except Exception as exc:
            st.error(bt(f"加载模型失败：{exc}", f"Failed to load the model: {exc}"))

    invalidate_from(
        st.session_state,
        "modeling",
        reason="modeling result replaced",
    )
    st.session_state.pop("modeling_failure", None)
    record_stage_status(
        st.session_state,
        "modeling",
        "succeeded",
        input_fingerprint=stable_fingerprint(
            st.session_state.get("analysis_dataset_fingerprint")
            or current_dataset_fingerprint(st.session_state),
            analysis_contract,
        ),
        output_fingerprint=stable_fingerprint(serializable),
    )
    return True


def modeling_code_gen(agent, debug=False, auto=False) -> None:
    suggest = agent.load_suggestion()
    control_slot = st.empty()
    summary_4 = st.session_state.get("summary_4") or st.session_state.get("modeling_summary_4")
    workflow_code = ""
    if isinstance(summary_4, dict):
        workflow_code = str(summary_4.get("code") or "").strip()

    if workflow_code:
        if debug:
            _load_workflow_modeling_code(agent, workflow_code)
            return

        code_is_loaded = str(agent.load_code() or "").strip() == workflow_code
        analyze_btn = False
        with control_slot.container():
            if not code_is_loaded:
                analyze_btn = st.button(
                    bt("🎯 生成模型建议代码", "🎯 Generate Modeling Code"),
                    key="modeling_code_from_workflow",
                )
        if analyze_btn:
            control_slot.empty()
            _load_workflow_modeling_code(agent, workflow_code)
            st.rerun()
        return

    if suggest is not None:
        code_is_loaded = bool(agent.load_code())
        analyze_btn = False
        with control_slot.container():
            if not code_is_loaded:
                analyze_btn = st.button(
                    bt("🎯 生成模型建议代码", "🎯 Generate Modeling Code"),
                    key="modeling_code_from_suggest",
                )
        if analyze_btn or (auto and not code_is_loaded):
            if (
                isinstance(st.session_state.get("_model_phase1_ctx"), dict)
                and isinstance(st.session_state.get("_model_phase2_inputs"), dict)
            ):
                st.session_state._model_phase2_requested = True
            else:
                st.error(
                    bt(
                        "当前建模建议上下文已失效，请清除后重新生成建议。",
                        "The modeling recommendation context has expired. Clear it and generate a new recommendation.",
                    )
                )
                return
            control_slot.empty()
            st.rerun()
