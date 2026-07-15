import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from streamlit_ace import st_ace

from core.modeling_contract import build_analysis_contract

from utils.i18n import bt, get_language, is_english_ui
from utils.page_paths import page_file
from utils.suggestion_state import (
    add_requirement,
    base_requirements_text,
    begin_code_execution,
    clear_suggestion_state,
    can_auto_repair,
    confirm_active_suggestion,
    finish_code_execution,
    get_suggestion_state,
    mark_code_draft,
    queue_revision_request,
    record_auto_repair,
    record_successful_code,
    record_validated_code,
    record_validation_failure,
    replace_active_suggestion,
    take_pending_revision,
    visible_messages,
)
from utils.sanitize_code import sanitize_code
from utils.workflow_state import (
    current_dataset_fingerprint,
    invalidate_from,
    record_stage_status,
    stable_fingerprint,
    stage_is_current,
)
from workflow.modeling.model_training import (
    modeling_code_gen,
    train_download_model,
    train_execution,
)

_MODEL_TARGET_WIDGET_KEY = "modeling_target_widget"


def _maybe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return value

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _stringify_content(value: Any) -> str:
    value = _maybe_json_loads(value)

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return json.dumps(value, ensure_ascii=False, indent=2)


def _find_nested_field(data: Any, field_name: str) -> Any:
    if isinstance(data, dict):
        if field_name in data:
            return data[field_name]

        for nested_value in data.values():
            nested = _find_nested_field(nested_value, field_name)
            if nested is not None:
                return nested

    if isinstance(data, list):
        for item in data:
            nested = _find_nested_field(item, field_name)
            if nested is not None:
                return nested

    return None


def clean_and_parse(raw_data: Any):
    if isinstance(raw_data, list):
        return raw_data
    if not isinstance(raw_data, str):
        return None

    content = raw_data.strip()
    try:
        return json.loads(content)
    except Exception:
        try:
            cleaned = content.replace('\\"', '"')
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = json.loads(cleaned)
            return json.loads(cleaned)
        except Exception:
            return None


def _serialize_dataframe_for_workflow(df: pd.DataFrame) -> str:
    safe_df = df.copy()

    for column in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[column]):
            safe_df[column] = safe_df[column].astype(str)

    return safe_df.to_json(orient="records", force_ascii=False)


def _source_to_dataframe(source: Any) -> pd.DataFrame | None:
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, np.ndarray):
        return pd.DataFrame(source)

    if isinstance(source, str):
        records = clean_and_parse(source)
        if records is None:
            return None
        try:
            return pd.DataFrame(records)
        except Exception:
            return None

    return None


def _has_usable_data(source: Any) -> bool:
    if source is None:
        return False

    if isinstance(source, pd.DataFrame):
        return not source.empty

    if isinstance(source, np.ndarray):
        return source.size > 0

    if isinstance(source, str):
        return bool(source.strip())

    if isinstance(source, (list, dict)):
        return bool(source)

    return True


def _resolve_modeling_source(preproc_agent, load_agent) -> tuple[Any, str | None]:
    dataset_fingerprint = current_dataset_fingerprint(st.session_state)
    stage_states = st.session_state.get("workflow_stage_states") or {}
    prep_state = stage_states.get("preprocessing") if isinstance(stage_states, dict) else None
    prep_succeeded = stage_is_current(
        st.session_state,
        "preprocessing",
        input_fingerprint=dataset_fingerprint,
    )

    if prep_succeeded:
        processed_df = preproc_agent.load_processed_df()
        if _has_usable_data(processed_df):
            return processed_df, "processed"

        summary_2 = st.session_state.get("summary_2")
        if isinstance(summary_2, dict):
            summary_processed_df = summary_2.get("processed_df")
            if _has_usable_data(summary_processed_df):
                return summary_processed_df, "processed"

        cached_processed_df = st.session_state.get("prep_result_from_summary_2")
        if _has_usable_data(cached_processed_df):
            return cached_processed_df, "processed"

    if isinstance(prep_state, dict) and prep_state.get("status") in {"failed", "running"}:
        return None, "preprocessing_failed"

    raw_df = load_agent.load_df()
    if _has_usable_data(raw_df):
        return raw_df, "raw"

    return None, None


def _agent_load_value(agent, method_name: str, attr_name: str, default: Any = None) -> Any:
    method = getattr(agent, method_name, None)
    if callable(method):
        return method()
    return getattr(agent, attr_name, default)


def _agent_save_value(agent, method_name: str, attr_name: str, value: Any) -> None:
    method = getattr(agent, method_name, None)
    if callable(method):
        method(value)
        return
    setattr(agent, attr_name, value)


def _sync_history_train_code_from_execution(agent) -> None:
    st.session_state.history_train_code_input = agent.load_code() or ""
    _agent_save_value(
        agent,
        "save_history_train_code",
        "history_train_code",
        st.session_state.history_train_code_input,
    )


def _format_user_prompt(user_selection: Any) -> str:
    if isinstance(user_selection, list):
        values = [str(item).strip() for item in user_selection if str(item).strip()]
        return ", ".join(values) if is_english_ui() else "，".join(values)

    if isinstance(user_selection, str):
        return user_selection.strip()

    return ""


def _resolve_effective_target(target_value: str, user_prompt: str) -> str:
    return (target_value or "").strip()


def _promote_contract_outcome(
    inputs: dict[str, Any],
    phase1_ctx: dict[str, Any],
    analysis_contract: dict[str, Any],
) -> str:
    """Use only a contract outcome that exactly matches an available dataset column."""
    columns = [str(column) for column in inputs.get("columns") or []]
    selected_target = str(inputs.get("target") or "").strip()
    if selected_target in columns:
        return selected_target

    outcome = str(analysis_contract.get("outcome") or "").strip()
    if not analysis_contract.get("valid", False) or outcome not in columns:
        return ""

    inputs["target"] = outcome
    phase1_ctx["target"] = outcome
    phase1_ctx["analysis_contract"] = analysis_contract
    return outcome


def _build_modeling_inputs(
    source_data: Any,
    agent,
    user_input: str,
    target_value: str,
    history_train_code: str,
    modeling_auto: bool = True,
    task_type: str = "auto",
) -> dict[str, Any] | None:
    df_obj = _source_to_dataframe(source_data)
    if df_obj is None:
        return None

    if isinstance(source_data, str):
        data_str = source_data
    else:
        data_str = _serialize_dataframe_for_workflow(df_obj)

    columns = df_obj.columns.astype(str).tolist()
    df_head = json.dumps(df_obj.head(5).to_dict(orient="list"), ensure_ascii=False)
    preference_selected = st.session_state.get("preference_selected")
    add_preference = st.session_state.get("add_preference")
    train_code = (history_train_code or "").strip()
    user_selection = _agent_load_value(agent, "load_user_selection", "user_selection", None)
    user_prompt = _format_user_prompt(user_selection)
    effective_target = _resolve_effective_target(target_value, user_prompt)

    return {
        "user_input": user_input or "",
        "df_head": df_head,
        "columns": columns,
        "target": effective_target,
        "train_code": train_code,
        "preference_selected": _stringify_content(preference_selected),
        "add_preference": add_preference or "",
        "user_prompt": user_prompt,
        "data": data_str,
        "modeling_auto": bool(modeling_auto),
        "language": get_language(),
        "task_type": task_type or "auto",
    }


def _normalize_modeling_workflow_result(result: Any) -> dict[str, Any] | None:
    result = _maybe_json_loads(result)
    if not isinstance(result, dict):
        return None

    summary_value = _find_nested_field(result, "summary_4")
    abstract_value = _find_nested_field(result, "abstract_4")
    model_suggestion = _find_nested_field(result, "model_suggestion")

    normalized = dict(result)
    normalized["summary_4"] = _maybe_json_loads(summary_value)
    normalized["abstract_4"] = _stringify_content(abstract_value)
    normalized["model_suggestion"] = _stringify_content(model_suggestion)
    return normalized


def _extract_summary_4_result() -> Any:
    summary_4 = st.session_state.get("summary_4") or st.session_state.get("modeling_summary_4")
    if not isinstance(summary_4, dict):
        return None
    return summary_4.get("result")


def _render_modeling_result(result_value: Any) -> None:
    parsed = _maybe_json_loads(result_value)

    if isinstance(parsed, (dict, list)):
        st.json(parsed)
        return

    text = _stringify_content(parsed)
    if not text:
        st.info(bt("暂无结果内容。", "No result content yet."))
        return

    normalized = text.replace("\r\n", "\n").strip()
    if "\n" in normalized:
        paragraphs = [segment.strip() for segment in normalized.split("\n\n") if segment.strip()]
        pretty_text = "\n\n".join(paragraphs) if paragraphs else normalized
        st.markdown(pretty_text)
        return

    st.markdown(f"> {normalized}")


def call_modeling_workflow(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Run the local modeling workflow."""
    from utils.local_workflow_bridge import call_modeling_bridge

    inputs = dict(inputs)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    inputs.setdefault("preference_selected", st.session_state.get("preference_selected") or "")

    result = call_modeling_bridge(inputs)
    if result is None:
        return None

    normalized = _normalize_modeling_workflow_result(result)
    if normalized is None:
        st.error(
            bt(
                "建模工作流返回结构异常，未解析到有效结果。",
                "The modeling workflow returned an invalid structure, and no usable result was found.",
            )
        )
        return None
    return normalized


def _call_modeling_phase1(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Phase 1: 仅生成 suggestion，快速返回给前端展示。"""
    from utils.local_workflow_bridge import call_modeling_phase1_bridge

    inputs = dict(inputs)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    inputs.setdefault("preference_selected", st.session_state.get("preference_selected") or "")
    return call_modeling_phase1_bridge(inputs)


def _call_modeling_phase2(inputs: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Phase 2: RAG + 代码生成 + 验证 + 分析。"""
    from utils.local_workflow_bridge import call_modeling_phase2_bridge

    inputs = dict(inputs)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    inputs.setdefault("preference_selected", st.session_state.get("preference_selected") or "")

    result = call_modeling_phase2_bridge(inputs, ctx)
    if result is None:
        return None

    normalized = _normalize_modeling_workflow_result(result)
    if normalized is None:
        st.error(
            bt(
                "建模代码生成阶段返回结构异常。",
                "The modeling code-generation stage returned an invalid structure.",
            )
        )
        return None
    return normalized


def _generate_modeling_code_draft(agent) -> None:
    from utils.local_workflow_bridge import call_modeling_validated_code_bridge

    state = get_suggestion_state(st.session_state, "modeling")
    ctx = st.session_state.get("_model_phase1_ctx")
    if not isinstance(ctx, dict):
        st.error(bt("当前建模建议上下文已失效，请重新生成建议。", "The modeling recommendation context has expired. Generate it again."))
        return
    inputs = st.session_state.get("_model_phase2_inputs") or {}
    with st.spinner(bt("正在生成并验证建模代码，失败时将自动修复，最多尝试5次...", "Generating and validating modeling code; up to five repair attempts...")):
        result = call_modeling_validated_code_bridge(inputs, ctx)
    error = str((result or {}).get("error") or "").strip()
    code = str((result or {}).get("code") or "").strip()
    if not code:
        st.error(error or bt("未能生成建模代码。", "No modeling code was generated."))
        return
    st.session_state._model_phase1_ctx = (result or {}).get("_ctx") or ctx
    agent.save_code(code)
    attempts = int((result or {}).get("attempts") or 0)
    if (result or {}).get("success"):
        st.session_state._model_phase1_ctx = (result or {}).get("_ctx") or ctx
        record_validated_code(state, code, attempts=attempts)
        st.success(bt(f"代码已生成并通过验证（第{attempts}/5次）。请点击执行建模。", f"Code passed validation on attempt {attempts}/5. Run modeling to publish the result."))
    else:
        record_validation_failure(state, code, str((result or {}).get("error") or "未生成可执行代码。"), attempts=attempts or 5)
        st.error(bt("建模代码连续5次未通过验证，已停止自动修复。", "Modeling code failed validation five times; auto-repair stopped."))
    st.rerun()


def _repair_modeling_code_draft(agent) -> None:
    from utils.local_workflow_bridge import call_modeling_validated_code_bridge

    state = get_suggestion_state(st.session_state, "modeling")
    ctx = st.session_state.get("_model_phase1_ctx")
    error_text = str(state.get("last_execution_error") or "")
    if not isinstance(ctx, dict) or not error_text or not can_auto_repair(state):
        return
    inputs = st.session_state.get("_model_phase2_inputs") or {}
    state["repair_in_progress"] = True
    with st.spinner(bt("正在自动修复代码，失败时将继续修复，最多尝试5次...", "Automatically repairing code; up to five attempts...")):
        result = call_modeling_validated_code_bridge(inputs, ctx, str(agent.load_code() or ""))
    code = str((result or {}).get("code") or "").strip()
    if not code:
        state["repair_in_progress"] = False
        st.error(bt("未能生成修复后的代码。", "No repaired code was generated."))
        return
    agent.save_code(code)
    attempts = int((result or {}).get("attempts") or 0)
    if (result or {}).get("success"):
        record_validated_code(state, code, attempts=attempts)
        st.success(bt("代码已修复并通过验证，请点击执行建模。", "Code was repaired and validated. Run modeling to publish the result."))
    else:
        record_validation_failure(state, code, str((result or {}).get("error") or error_text), attempts=attempts or 5)
        st.error(bt("自动修复已达到5次，未能生成可执行代码。", "Auto-repair reached five attempts without a valid script."))
    st.rerun()


def _clear_modeling_workflow_state(agent) -> None:
    invalidate_from(
        st.session_state,
        "modeling",
        include_source=True,
        reason="modeling analysis cleared",
    )
    agent.clear_memory()
    agent.save_suggestion(None)
    agent.save_code(None)
    agent.save_modeling_result(None)
    _agent_save_value(agent, "save_user_input", "user_input", None)
    _agent_save_value(agent, "save_user_selection", "user_selection", None)
    _agent_save_value(agent, "save_target", "target", "")
    _agent_save_value(agent, "save_task_type", "task_type", "auto")
    _agent_save_value(agent, "save_history_train_code", "history_train_code", "")
    clear_suggestion_state(st.session_state, "modeling")
    st.session_state.history_train_code_reset_pending = True
    for key in (
        "modeling_workflow_result", "modeling_suggestion", "model_suggestion",
        "modeling_summary_4", "modeling_abstract_4", "summary_4", "abstract_4",
        "modeling_result_from_summary_4", "modeling_user_prompt",
        "_model_phase1_ctx", "_model_phase2_inputs", "_model_phase2_pending",
        "_model_phase2_requested",
    ):
        st.session_state.pop(key, None)
    st.session_state._model_target_sync = ""


def _reset_modeling_outputs(agent) -> None:
    invalidate_from(
        st.session_state,
        "modeling",
        include_source=True,
        reason="modeling analysis restarted",
    )
    agent.save_suggestion(None)
    agent.save_code(None)
    agent.save_modeling_result(None)
    for key in (
        "modeling_workflow_result", "modeling_suggestion", "model_suggestion",
        "modeling_summary_4", "modeling_abstract_4", "summary_4", "abstract_4",
        "modeling_result_from_summary_4",
        "_model_phase1_ctx", "_model_phase2_inputs", "_model_phase2_pending",
        "_model_phase2_requested",
    ):
        st.session_state.pop(key, None)


def _request_modeling_recommendation(
    agent,
    source_data: Any,
    user_input: str,
    target_value: str,
    history_train_code: str,
    task_type: str,
    auto: bool,
) -> None:
    state = get_suggestion_state(st.session_state, "modeling")
    inputs = _build_modeling_inputs(
        source_data=source_data,
        agent=agent,
        user_input=user_input,
        target_value=target_value,
        history_train_code=history_train_code,
        modeling_auto=True,
        task_type=task_type,
    )
    if inputs is None:
        st.error(
            bt(
                "无法从当前可用数据构造建模工作流输入，请检查预处理结果或原始上传数据是否可解析。",
                "Unable to build modeling workflow input from the available data. Check whether the preprocessing result or original uploaded data can be parsed.",
            )
        )
        return

    request_contract = build_analysis_contract(
        target=str(inputs.get("target") or ""),
        columns=list(inputs.get("columns") or []),
        user_input="\n".join(
            part
            for part in (
                str(inputs.get("user_input") or ""),
                str(inputs.get("user_prompt") or ""),
            )
            if part
        ),
        add_preference=str(inputs.get("add_preference") or ""),
        task_type=str(inputs.get("task_type") or "auto"),
    )
    if not request_contract.get("valid", False):
        st.error(bt(
            "当前任务需要从数据字段中选择目标变量后才能继续。",
            "Choose an outcome from the dataset columns before continuing with this task.",
        ))
        return

    _reset_modeling_outputs(agent)

    # ── Phase 1: 快速获取 suggestion 并展示 ──────────────────────
    with st.spinner(bt("正在生成建模推荐方案...", "Generating modeling recommendations...")):
        phase1_result = _call_modeling_phase1(inputs)

    if not phase1_result:
        return

    suggestion = phase1_result.get("model_suggestion", "")
    phase1_ctx = phase1_result.get("_ctx", {})
    analysis_contract = phase1_result.get("analysis_contract") or phase1_ctx.get("analysis_contract") or {}
    had_selected_target = bool(str(inputs.get("target") or "").strip())
    resolved_target = _promote_contract_outcome(inputs, phase1_ctx, analysis_contract)
    if resolved_target and not had_selected_target:
        agent.save_target(resolved_target)
        st.session_state._model_target_sync = resolved_target

    # 先把 suggestion 写入 session_state，让前端立刻可以显示
    st.session_state.modeling_suggestion = suggestion
    st.session_state.model_suggestion = suggestion
    st.session_state.modeling_analysis_contract = analysis_contract
    agent.save_suggestion(suggestion)

    # 缓存 phase1 上下文和 inputs，供 phase2 使用
    st.session_state._model_phase1_ctx = phase1_ctx
    st.session_state._model_phase2_inputs = inputs
    replace_active_suggestion(state, suggestion)
    if auto:
        confirm_active_suggestion(state)
        st.session_state._model_phase2_pending = True
    st.rerun()


def _revise_modeling_recommendation(agent, revision_instruction: str) -> None:
    from workflows.modeling import revise_modeling_phase1

    state = get_suggestion_state(st.session_state, "modeling")
    phase1_ctx = st.session_state.get("_model_phase1_ctx")
    phase2_inputs = st.session_state.get("_model_phase2_inputs")
    if not isinstance(phase1_ctx, dict):
        return
    with st.spinner(bt("正在修改建模建议...", "Revising modeling recommendations...")):
        revised = revise_modeling_phase1(
            ctx=phase1_ctx,
            original_requirements=base_requirements_text(state),
            revision_instruction=revision_instruction,
        )
    if state.get("confirmed_version") is not None:
        invalidate_from(
            st.session_state,
            "modeling",
            include_source=True,
            reason="confirmed modeling recommendation is being revised",
        )
    suggestion = str(revised.get("model_suggestion") or "")
    st.session_state._model_phase1_ctx = revised.get("_ctx")
    st.session_state._model_phase2_inputs = phase2_inputs
    st.session_state.modeling_analysis_contract = revised.get("analysis_contract") or {}
    st.session_state.modeling_suggestion = suggestion
    st.session_state.model_suggestion = suggestion
    agent.save_suggestion(suggestion)
    replace_active_suggestion(state, suggestion, revision_instruction=revision_instruction)
    st.rerun()


def _continue_modeling_phase2(agent) -> None:
    """在 suggestion 已展示的前提下，继续执行 phase2（RAG + 代码生成 + 训练 + 分析）。"""
    state = get_suggestion_state(st.session_state, "modeling")
    if state.get("status") != "confirmed":
        return
    phase1_ctx = st.session_state.get("_model_phase1_ctx")
    inputs = st.session_state.get("_model_phase2_inputs")
    st.session_state.pop("_model_phase2_pending", None)

    if not phase1_ctx or not inputs:
        return

    with st.spinner(
        bt(
            "建模建议已生成，正在生成代码与训练分析...",
            "Modeling recommendations are ready. Generating code and training analysis...",
        )
    ):
        workflow_result = _call_modeling_phase2(inputs, phase1_ctx)

    if not workflow_result:
        return

    analysis_contract = (
        workflow_result.get("_analysis_contract")
        or workflow_result.get("analysis_contract")
        or phase1_ctx.get("analysis_contract")
        or {}
    )
    if workflow_result.get("_status") != "succeeded":
        summary_4 = workflow_result.get("summary_4")
        error_text = str(
            (summary_4.get("desc") if isinstance(summary_4, dict) else "")
            or workflow_result.get("abstract_4")
            or bt("建模执行失败。", "Modeling execution failed.")
        )
        final_code = str(
            (summary_4.get("code") if isinstance(summary_4, dict) else "")
            or workflow_result.get("_final_code")
            or ""
        ).strip()
        attempts = int(workflow_result.get("_fix_attempts") or 5)
        if final_code:
            agent.save_code(final_code)
            record_validation_failure(state, final_code, error_text, attempts=attempts)
        st.session_state.modeling_analysis_contract = analysis_contract
        st.session_state.modeling_failure = error_text
        record_stage_status(
            st.session_state,
            "modeling",
            "failed",
            input_fingerprint=stable_fingerprint(
                st.session_state.get("analysis_dataset_fingerprint")
                or current_dataset_fingerprint(st.session_state),
                analysis_contract,
            ),
            error=error_text,
        )
        agent.save_error(error_text)
        if st.session_state.get("auto_mode"):
            st.session_state.auto_mode = False
            st.session_state.auto_mode_paused_stage = "modeling"
        st.rerun()

    invalidate_from(
        st.session_state,
        "modeling",
        reason="modeling result replaced",
    )

    st.session_state.modeling_workflow_result = workflow_result
    st.session_state.modeling_summary_4 = workflow_result.get("summary_4")
    st.session_state.modeling_abstract_4 = workflow_result.get("abstract_4")
    st.session_state.summary_4 = workflow_result.get("summary_4")
    st.session_state.abstract_4 = workflow_result.get("abstract_4")
    st.session_state.modeling_analysis_contract = analysis_contract
    st.session_state.pop("modeling_failure", None)

    summary_4 = workflow_result.get("summary_4")
    if isinstance(summary_4, dict):
        workflow_code = str(summary_4.get("code") or workflow_result.get("_final_code") or "").strip()
        if workflow_code:
            agent.save_code(workflow_code)
            record_successful_code(state, workflow_code)
        result_text = summary_4.get("result")
        if result_text is not None:
            agent.save_modeling_result(result_text)

    agent.add_memory({"role": "assistant", "content": workflow_result})
    model_input_fingerprint = stable_fingerprint(
        st.session_state.get("analysis_dataset_fingerprint")
        or current_dataset_fingerprint(st.session_state),
        analysis_contract,
    )
    record_stage_status(
        st.session_state,
        "modeling",
        "succeeded",
        input_fingerprint=model_input_fingerprint,
        output_fingerprint=stable_fingerprint(workflow_result.get("summary_4")),
    )
    st.rerun()


def _has_modeling_result(agent) -> bool:
    suggestion = st.session_state.get("model_suggestion") or st.session_state.get("modeling_suggestion")
    if suggestion is None:
        suggestion = agent.load_suggestion()
    return bool(suggestion)


def _has_completed_modeling_result(agent) -> bool:
    summary_4 = st.session_state.get("summary_4") or st.session_state.get("modeling_summary_4")
    if not isinstance(summary_4, dict):
        return False

    has_code = bool(str(summary_4.get("code") or agent.load_code() or "").strip())
    has_result = bool(summary_4.get("result") or agent.load_modeling_result())
    return bool(
        has_code
        and has_result
        and stage_is_current(st.session_state, "modeling")
    )


def modeling_quick_actions(agent):
    st.write(bt("选择一个或多个 model:", "Select one or more models:"))
    selected_models = sac.chip(
        items=[
            sac.ChipItem(label=bt("线性回归", "Linear Regression")),
            sac.ChipItem(label="XGBoost"),
            sac.ChipItem(label=bt("随机森林", "Random Forest")),
            sac.ChipItem(label=bt("神经网络", "Neural Network")),
        ],
        index=[0, 2],
        align="center",
        direction="horizontal",
        size="sm",
        radius="md",
        color="#44658C",
        multiple=True,
    )

    if st.button(bt("🖋️ 快速建模", "🖋️ Quick Modeling"), key="quick_modeling"):
        if not selected_models:
            st.error(bt("请先选择训练 model。", "Please select at least one training model."))
        elif not str(_agent_load_value(agent, "load_target", "target", "") or "").strip():
            st.error(bt(
                "快速建模属于监督学习，请先从数据字段中选择目标变量。",
                "Quick modeling is supervised; choose an outcome from the dataset columns first.",
            ))
        else:
            agent.save_user_selection(selected_models)
            st.session_state.modeling_user_prompt = _format_user_prompt(selected_models)
            st.success(
                bt(
                    "已保存快速建模选择，后续会作为 user_prompt 传入建模工作流。",
                    "Quick modeling selections have been saved and will be passed to the modeling workflow as user_prompt.",
                )
            )
            st.rerun()

    return selected_models


def modeling_execution(agent, auto=False) -> None:
    code = agent.load_code()

    edited = st_ace(
        value=code,
        height=450,
        theme="tomorrow_night",
        language="python",
        auto_update=True,
    )

    state = get_suggestion_state(st.session_state, "modeling")

    if edited is not None:
        _, result_is_current = mark_code_draft(state, edited)
        tracked_execution = bool(state.get("executed_code_fingerprint"))
        stale_result = tracked_execution and not result_is_current
        not_executed = agent.load_modeling_result() is None or stale_result
        if agent.load_modeling_result() is not None and stale_result:
            st.warning(
                bt(
                    "代码已修改，下方保留的是上一次成功代码生成的结果。",
                    "The code has changed. The result below is from the last successful code.",
                )
            )
        if st.button(bt("▶️ 执行建模", "▶️ Run Modeling"), key="modeling_run_code") or (auto and not_executed):
            code = sanitize_code(edited)
            run_id = begin_code_execution(state, code)
            agent.save_code(code)
            if train_execution(agent):
                finish_code_execution(state, run_id, success=True)
                st.session_state.modeling_result_from_summary_4 = _extract_summary_4_result()
                agent.finish_auto()
                st.rerun()
            else:
                finish_code_execution(state, run_id, success=False)

        modeling_result = agent.load_modeling_result()
        summary_result = st.session_state.get("modeling_result_from_summary_4")
        _, result_is_current = mark_code_draft(state, edited)
        if summary_result is not None or modeling_result is not None:
            train_download_model(agent)
            with st.container():
                st.subheader(bt("训练结果", "Training Results"))
                if summary_result is not None:
                    _render_modeling_result(summary_result)
                else:
                    _render_modeling_result(modeling_result)

        error_text = str(state.get("last_execution_error") or "")
        if error_text:
            st.error(bt("执行失败", "Execution failed"))
            with st.expander(bt("查看错误详情", "View error details")):
                st.code(error_text, language="text")
            attempts = int(state.get("auto_repair_attempts") or 0)
            if can_auto_repair(state):
                repair_slot = st.empty()
                with repair_slot.container():
                    repair_requested = st.button(
                        bt("自动修复代码", "Auto-fix Code"),
                        key="modeling_auto_fix_code",
                    )
                    if attempts:
                        st.caption(bt(f"本轮已自动修复 {attempts} 次，最多 5 次。", f"This round has used {attempts} of 5 automatic repairs."))
                if repair_requested:
                    repair_slot.empty()
                    st.session_state._model_code_repair_requested = True
                    st.rerun()
            else:
                st.caption(bt("已达到自动修复上限，请手动修改代码或调整建议后重新生成。", "The automatic repair limit has been reached. Edit the code or revise the suggestion before generating again."))


def modeling_chat(agent, source_data: Any, auto: bool) -> None:
    state = get_suggestion_state(st.session_state, "modeling")
    df_obj = _source_to_dataframe(source_data)
    available_columns = df_obj.columns.astype(str).tolist() if df_obj is not None else []

    task_type_options = ["auto", "association_inference", "prediction", "unsupervised"]
    task_type_labels = {
        "auto": bt("自动判断", "Auto Detect"),
        "association_inference": bt("关联分析 / 统计推断", "Association / Inference"),
        "prediction": bt("监督预测", "Supervised Prediction"),
        "unsupervised": bt("无监督 / 探索分析", "Unsupervised / Exploratory"),
    }
    current_task_type = str(
        _agent_load_value(agent, "load_task_type", "task_type", "auto") or "auto"
    )
    if current_task_type not in task_type_options:
        current_task_type = "auto"
    task_type = st.selectbox(
        bt("建模任务类型", "Modeling Task Type"),
        options=task_type_options,
        index=task_type_options.index(current_task_type),
        format_func=lambda value: task_type_labels[value],
    )
    if task_type != current_task_type:
        invalidate_from(
            st.session_state,
            "modeling",
            include_source=True,
            reason="modeling task type changed",
        )
        clear_suggestion_state(st.session_state, "modeling")
        state = get_suggestion_state(st.session_state, "modeling")
    _agent_save_value(agent, "save_task_type", "task_type", task_type)

    current_target = _agent_load_value(agent, "load_target", "target", "") or ""
    target_options = [""] + available_columns
    if current_target not in target_options:
        current_target = ""
    pending_target = st.session_state.pop("_model_target_sync", None)
    widget_target = st.session_state.get(_MODEL_TARGET_WIDGET_KEY)
    if pending_target is not None:
        st.session_state[_MODEL_TARGET_WIDGET_KEY] = (
            pending_target if pending_target in target_options else ""
        )
    elif widget_target not in target_options:
        st.session_state[_MODEL_TARGET_WIDGET_KEY] = current_target
    target_value = st.selectbox(
        bt("建模目标", "Modeling Target"),
        options=target_options,
        format_func=lambda value: value or bt("请选择目标字段（无监督任务可留空）", "Choose an outcome (optional for unsupervised tasks)"),
        key=_MODEL_TARGET_WIDGET_KEY,
    )
    if target_value != current_target:
        invalidate_from(
            st.session_state,
            "modeling",
            include_source=True,
            reason="modeling target changed",
        )
        clear_suggestion_state(st.session_state, "modeling")
        state = get_suggestion_state(st.session_state, "modeling")
    agent.save_target(target_value)

    current_history_train_code = _agent_load_value(
        agent,
        "load_history_train_code",
        "history_train_code",
        "",
    ) or ""
    if st.session_state.pop("history_train_code_reset_pending", False):
        st.session_state.history_train_code_input = ""
    if "history_train_code_input" not in st.session_state:
        st.session_state.history_train_code_input = current_history_train_code

    history_train_code = st.text_area(
        bt("历史训练代码", "Historical Training Code"),
        key="history_train_code_input",
        placeholder=bt(
            "若有历史训练代码可在此输入，也可点击下方按钮同步当前执行区代码。",
            "Paste historical training code here, or use the button below to sync the current execution code.",
        ),
        height=180,
    )
    if history_train_code != current_history_train_code:
        invalidate_from(
            st.session_state,
            "modeling",
            include_source=True,
            reason="historical modeling code changed",
        )
        clear_suggestion_state(st.session_state, "modeling")
        state = get_suggestion_state(st.session_state, "modeling")
    _agent_save_value(agent, "save_history_train_code", "history_train_code", history_train_code)

    st.button(
        bt("获取当前执行区代码", "Get Current Execution Code"),
        key="sync_history_train_code",
        on_click=_sync_history_train_code_from_execution,
        args=(agent,),
    )

    with st.chat_message("assistant"):
        st.write(
            bt(
                "我是 Autostat 数据分析助手。\n\n"
                "你可以在下方输入建模相关问题，或直接点击按钮获取建模推荐。",
                "I am the Autostat data analysis assistant.\n\n"
                "Enter a modeling question below, or get automatic modeling recommendations.",
            )
        )

        if st.session_state.get("modeling_failure"):
            st.error(bt("上一次建模代码未通过验证。", "The previous modeling code did not pass validation."))
            with st.expander(bt("查看错误详情", "View error details")):
                st.code(str(st.session_state.modeling_failure), language="text")

        columns = st.columns(2)
        with columns[0]:
            analyze_btn = st.button(
                bt("🔍 建模推荐", "🔍 Modeling Recommendation"),
                key="modeling_suggest",
                use_container_width=True,
                disabled=bool(state.get("active_suggestion")),
            )
        with columns[1]:
            clear_modeling_suggest = st.button(
                bt("♻️ 清除建模分析", "♻️ Clear Modeling Analysis"),
                key="clear_modeling_suggest",
                use_container_width=True,
            )

        if clear_modeling_suggest:
            _clear_modeling_workflow_state(agent)
            st.rerun()

    for entry in visible_messages(state):
        role = entry.get("role")
        content = entry.get("content")
        st.chat_message(role).write(str(content))

    pending_revision = take_pending_revision(state)
    if pending_revision:
        _revise_modeling_recommendation(agent, pending_revision)
        return

    already_generated = bool(state.get("active_suggestion"))
    saved_user_input = _agent_load_value(agent, "load_user_input", "user_input", "") or ""

    # ── Phase 2 自动续接：suggestion 已展示，继续生成代码 ──────────
    if st.session_state.get("_model_phase2_pending"):
        _continue_modeling_phase2(agent)
        return

    if st.session_state.pop("_model_code_repair_requested", False):
        _repair_modeling_code_draft(agent)
        return

    if st.session_state.pop("_model_phase2_requested", False):
        if confirm_active_suggestion(state):
            _generate_modeling_code_draft(agent)
        return

    if auto and _has_completed_modeling_result(agent) and not agent.finish_auto_task:
        agent.finish_auto()
        st.rerun()

    if analyze_btn or (auto and not already_generated):
        prompt_text = bt("请帮我获取建模建议", "Please provide modeling recommendations.")
        if saved_user_input and not state.get("base_requirements"):
            add_requirement(state, saved_user_input)
        if not state.get("base_requirements"):
            add_requirement(state, prompt_text)
        request_text = base_requirements_text(state, prompt_text)
        agent.save_user_input(request_text)
        _request_modeling_recommendation(
            agent=agent,
            source_data=source_data,
            user_input=request_text,
            target_value=target_value,
            history_train_code=history_train_code,
            task_type=task_type,
            auto=auto,
        )
        return

    user_input = st.chat_input(
        bt(
            "请输入建模要求；建议生成后可继续提出修改意见",
            "Enter modeling requirements; after generation, request revisions here",
        )
    )
    if user_input:
        if state.get("active_suggestion"):
            queue_revision_request(state, user_input)
            st.rerun()
        else:
            add_requirement(state, user_input)
            agent.save_user_input(base_requirements_text(state))
            st.rerun()


if __name__ == "__main__":
    st.title(bt("数据建模", "Data Modeling"))
    st.markdown("---")

    preproc_agent = st.session_state.data_preprocess_agent
    load_agent = st.session_state.data_loading_agent
    source_data, source_kind = _resolve_modeling_source(preproc_agent, load_agent)
    df = _source_to_dataframe(source_data)

    if df is None:
        if source_kind == "preprocessing_failed":
            st.error(bt(
                "预处理尚未成功，不能把原始数据作为处理后数据继续建模。请先修复或明确跳过预处理。",
                "Preprocessing has not succeeded. Raw data cannot silently continue as processed data. Fix or explicitly skip preprocessing first.",
            ))
        else:
            st.warning(bt("请先在数据导入页面加载数据。", "Please load data on the data import page first."))
        st.stop()

    agent = st.session_state.modeling_coding_agent
    agent.add_df(df)
    planner = st.session_state.planner_agent
    auto = bool(st.session_state.auto_mode and planner.modeling_auto)

    if st.session_state.auto_mode is True:
        if planner.modeling_auto and _has_completed_modeling_result(agent):
            next_page = planner.finish_modeling_auto()
            if next_page is not None:
                st.switch_page(next_page)
            st.session_state.auto_mode = False
            st.rerun()

    code = agent.load_code()

    if source_kind == "raw":
        st.caption(
            bt(
                "当前未对原始数据进行预处理，后续将基于原始数据进行分析",
                "The original data has not been preprocessed. The following analysis will use the raw data.",
            )
        )

    columns = st.columns(2)
    with columns[0].expander(bt("快速建模", "Quick Modeling"), True):
        modeling_quick_actions(agent)
    with columns[1].expander(bt("建模建议", "Modeling Suggestions"), True):
        modeling_chat(agent, source_data, auto)
        modeling_code_gen(agent, auto=auto)
    with columns[0].expander(bt("建模执行", "Modeling Execution"), code is not None):
        modeling_execution(agent, auto)
