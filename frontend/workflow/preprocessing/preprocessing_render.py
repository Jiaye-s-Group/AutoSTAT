import io
import json
from typing import Any

import pandas as pd
import streamlit as st

from utils.i18n import bt, get_language
from utils.page_paths import page_file
from utils.suggestion_state import (
    add_requirement,
    base_requirements_text,
    clear_suggestion_state,
    confirm_active_suggestion,
    can_auto_repair,
    get_suggestion_state,
    mark_code_draft,
    queue_initial_request,
    queue_revision_request,
    record_auto_repair,
    record_successful_code,
    record_validated_code,
    record_validation_failure,
    replace_active_suggestion,
    revision_fallback_text,
    take_pending_code_revision,
    take_pending_initial_request,
    take_pending_revision,
    visible_messages,
)
from utils.workflow_state import (
    current_dataset_fingerprint,
    dataframe_fingerprint,
    invalidate_from,
    record_stage_status,
    stable_fingerprint,
    stage_is_current,
)
from workflow.preprocessing.preprocessing_core import prep_code_gen, prep_meta_execution

# Local workflow configuration.
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


def _find_nested_field(data: Any, field_name: str) -> Any:
    if isinstance(data, dict):
        if field_name in data:
            return data[field_name]

        for value in data.values():
            nested = _find_nested_field(value, field_name)
            if nested is not None:
                return nested

    if isinstance(data, list):
        for item in data:
            nested = _find_nested_field(item, field_name)
            if nested is not None:
                return nested

    return None


def _normalize_prep_workflow_result(result: Any) -> dict[str, Any] | None:
    result = _maybe_json_loads(result)
    if not isinstance(result, dict):
        return None

    summary_2 = _find_nested_field(result, "summary_2")
    abstract_2 = _find_nested_field(result, "abstract_2")
    suggestion = _find_nested_field(result, "suggestion")

    normalized = dict(result)
    normalized["abstract_2"] = _maybe_json_loads(abstract_2)
    normalized["summary_2"] = _maybe_json_loads(summary_2)
    suggestion = _maybe_json_loads(suggestion)

    if isinstance(suggestion, (dict, list)):
        normalized["suggestion"] = json.dumps(suggestion, ensure_ascii=False, indent=2)
    else:
        normalized["suggestion"] = suggestion

    return normalized


def _stringify_content(value: Any) -> str | None:
    value = _maybe_json_loads(value)

    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        return value or None

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)

    return str(value)


def _coerce_processed_dataframe(payload: Any) -> pd.DataFrame | None:
    if isinstance(payload, pd.DataFrame):
        return payload.copy()

    value = payload
    for _ in range(2):
        parsed = _maybe_json_loads(value)
        if parsed is value:
            break
        value = parsed

    if isinstance(value, list):
        try:
            return pd.DataFrame(value)
        except (TypeError, ValueError):
            return None

    if isinstance(value, dict):
        try:
            if {"columns", "data"}.issubset(value):
                return pd.DataFrame(
                    data=value.get("data") or [],
                    columns=value.get("columns") or None,
                    index=value.get("index") or None,
                )
            return pd.DataFrame(value)
        except (TypeError, ValueError):
            try:
                return pd.DataFrame([value])
            except (TypeError, ValueError):
                return None

    return None


def _extract_suggestion_text(workflow_result: dict[str, Any]) -> str | None:
    suggestion = _stringify_content(workflow_result.get("suggestion"))
    if suggestion:
        return suggestion

    summary_2 = workflow_result.get("summary_2")
    if isinstance(summary_2, dict):
        suggestion = _stringify_content(summary_2.get("desc"))
        if suggestion:
            return suggestion

    suggestion = _stringify_content(workflow_result.get("abstract_2"))
    if suggestion:
        return suggestion

    suggestion = _stringify_content(_find_nested_field(workflow_result, "desc"))
    if suggestion:
        return suggestion

    return None


def _serialize_dataframe_for_workflow(df: pd.DataFrame) -> str:
    safe_df = df.copy()

    # Workflow inputs use JSON, so serialize the in-memory DataFrame.
    for column in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[column]):
            safe_df[column] = safe_df[column].astype(str)

    return safe_df.to_json(orient="records", force_ascii=False)


def call_preprocessing_workflow(
    df: pd.DataFrame,
    prep_auto: bool = True,
    user_input: str = "",
    phase: str = "",
    phase1_ctx: dict[str, Any] | None = None,
    code: str = "",
    error: str = "",
) -> dict[str, Any] | None:
    """本地化版本：改走本地 Preprocessing workflow。"""
    from utils.local_workflow_bridge import call_preprocessing_bridge

    preview_df = df.head(10)
    inputs = {
        "shape_0": int(df.shape[0]),
        "shape_1": int(df.shape[1]),
        "dtype_info_str": df.dtypes.astype(str).to_json(),
        "head_dict_str": preview_df.to_json(orient="records"),
        "df": _serialize_dataframe_for_workflow(df),
        "user_input": user_input or "",
        "prep_auto": bool(prep_auto),
        "add_preference": st.session_state.get("add_preference") or "",
        "preference_selected": st.session_state.get("preference_selected") or "",
        "language": get_language(),
        "_phase": phase,
        "_phase1_ctx": phase1_ctx or {},
        "_code": code,
        "_error": error,
    }

    result = call_preprocessing_bridge(inputs)
    if result is None:
        return None

    normalized = _normalize_prep_workflow_result(result)
    if normalized is None:
        st.error(
            bt(
                "预处理工作流返回结构异常，未解析到有效结果。",
                "The preprocessing workflow returned an invalid structure, and no usable result was found.",
            )
        )
        return None
    return normalized


def prep_basic_info(agent) -> None:
    df = agent.load_df()

    row_count, col_count = df.shape
    missing_count = int(df.isnull().sum().sum())

    col1, col2, col3 = st.columns(3)
    col1.metric(bt("行数", "Rows"), row_count)
    col2.metric(bt("列数", "Columns"), col_count)
    col3.metric(bt("缺失值总数", "Missing Values"), missing_count)

    dtype_info = pd.DataFrame(
        {
            bt("列名", "Column"): df.columns,
            bt("类型", "Type"): df.dtypes.astype(str),
            bt("非空值数量", "Non-null Count"): df.count().values,
            bt("缺失值比例(%)", "Missing Ratio (%)"): (df.isnull().mean() * 100).round(2).values,
        }
    ).reset_index(drop=True)
    st.dataframe(dtype_info, use_container_width=True)


def prep_execution(agent, auto: bool = False) -> None:
    code = agent.load_code()
    df = agent.load_df()
    prep_meta_execution(agent, code, df, auto=auto)


def prep_result(agent) -> None:
    process_df = agent.load_processed_df()
    df = agent.load_df()
    workflow_processed_df = st.session_state.get("prep_result_from_summary_2")

    if process_df is None:
        process_df = _coerce_processed_dataframe(workflow_processed_df)
        if process_df is not None:
            agent.save_processed_df(process_df)
            st.session_state.prep_result_from_summary_2 = process_df

    if process_df is None:
        if workflow_processed_df is not None:
            st.warning(
                bt(
                    "处理结果格式异常，无法转换为数据表。请重新执行预处理。",
                    "The processed result could not be converted into a data table. Run preprocessing again.",
                )
            )
        return

    state = get_suggestion_state(st.session_state, "preprocessing")
    executed_fingerprint = state.get("executed_code_fingerprint")
    current_fingerprint = state.get("current_code_fingerprint")
    if executed_fingerprint and current_fingerprint != executed_fingerprint:
        st.info(
            bt(
                "下方结果来自上一次成功代码；当前草稿尚未执行。",
                "The result below is from the last successful code; the current draft has not run.",
            )
        )

    st.write(bt("处理前数据预览：", "Preview Before Processing:"), df.head(10))

    st.write(bt("处理后数据预览：", "Preview After Processing:"), process_df.head(10))

    qc_summary = st.session_state.get("preprocessing_qc_summary")
    if not isinstance(qc_summary, dict):
        summary_2 = st.session_state.get("summary_2")
        qc_summary = summary_2.get("qc_summary") if isinstance(summary_2, dict) else None
    if isinstance(qc_summary, dict) and qc_summary:
        st.caption(bt("预处理 QC 摘要（由执行代码产生并校验）", "Preprocessing QC summary (produced and validated by the executed code)"))
        st.json(qc_summary, expanded=False)

    csv_buffer = io.StringIO()
    process_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    st.download_button(
        label=bt("⬇️ 下载处理后数据", "⬇️ Download Processed Data"),
        data=csv_bytes,
        file_name="processed_data.csv",
        mime="text/csv",
    )


def _clear_prep_workflow_state(agent) -> None:
    invalidate_from(
        st.session_state,
        "preprocessing",
        include_source=True,
        reason="preprocessing analysis cleared",
    )
    agent.clear_memory()
    agent.preprocessing_suggestions = None
    agent.code = None
    agent.processed_df = None
    agent.user_input = None
    agent.error = None
    agent.finish_auto_task = False
    clear_suggestion_state(st.session_state, "preprocessing")

    for key in ("suggestion", "abstract_2", "summary_2", "prep_result_from_summary_2", "preprocessing_qc_summary", "prep_code_visible"):
        st.session_state.pop(key, None)


def _has_prep_result(agent) -> bool:
    dataset_fingerprint = current_dataset_fingerprint(st.session_state)
    summary_2 = st.session_state.get("summary_2")
    return bool(
        dataset_fingerprint
        and stage_is_current(
            st.session_state,
            "preprocessing",
            input_fingerprint=dataset_fingerprint,
        )
        and isinstance(summary_2, dict)
        and summary_2.get("status") != "failed"
        and summary_2.get("processed_df")
    )


def _request_prep_recommendation(agent, df: pd.DataFrame, user_input: str, *, auto: bool) -> None:
    state = get_suggestion_state(st.session_state, "preprocessing")
    with st.spinner(bt("正在生成预处理建议...", "Generating preprocessing recommendations...")):
        phase1_result = call_preprocessing_workflow(
            df, prep_auto=True, user_input=user_input, phase="phase1"
        )
    if not phase1_result or not phase1_result.get("_ctx"):
        st.error(bt("预处理建议生成失败。", "Failed to generate preprocessing recommendations."))
        return
    suggestion = _extract_suggestion_text(phase1_result) or ""
    state["phase1_ctx"] = phase1_result.get("_ctx")
    replace_active_suggestion(state, suggestion)
    st.session_state.suggestion = suggestion
    agent.save_preprocessing_suggestions(suggestion)
    if auto:
        confirm_active_suggestion(state)
        _continue_prep_phase2(agent, df)
        return
    st.rerun()


def _revise_prep_recommendation(agent, revision_instruction: str) -> None:
    from workflows.preprocessing import revise_preprocessing_phase1

    state = get_suggestion_state(st.session_state, "preprocessing")
    ctx = state.get("phase1_ctx")
    if not isinstance(ctx, dict):
        return
    with st.spinner(bt("正在修改预处理建议...", "Revising preprocessing recommendations...")):
        revised = revise_preprocessing_phase1(
            ctx=ctx,
            original_requirements=base_requirements_text(state),
            revision_instruction=revision_instruction,
        )
    if state.get("confirmed_version") is not None:
        invalidate_from(
            st.session_state,
            "preprocessing",
            include_source=True,
            reason="confirmed preprocessing recommendation is being revised",
        )
    suggestion = _extract_suggestion_text(revised) or ""
    state["phase1_ctx"] = revised.get("_ctx")
    replace_active_suggestion(state, suggestion, revision_instruction=revision_instruction)
    st.session_state.suggestion = suggestion
    agent.save_preprocessing_suggestions(suggestion)
    st.rerun()


def _continue_prep_phase2(agent, df: pd.DataFrame) -> None:
    state = get_suggestion_state(st.session_state, "preprocessing")
    ctx = state.get("phase1_ctx")
    if not isinstance(ctx, dict):
        return
    with st.spinner(bt("正在生成并执行预处理代码...", "Generating and executing preprocessing code...")):
        workflow_result = call_preprocessing_workflow(
            df,
            prep_auto=True,
            user_input=base_requirements_text(state),
            phase="phase2",
            phase1_ctx=ctx,
        )
    if workflow_result:
        _handle_prep_workflow_result(agent, workflow_result)


def _generate_prep_code_draft(agent, df: pd.DataFrame) -> None:
    state = get_suggestion_state(st.session_state, "preprocessing")
    ctx = state.get("phase1_ctx")
    if not isinstance(ctx, dict):
        st.error(bt("当前预处理建议上下文已失效，请重新生成建议。", "The preprocessing recommendation context has expired. Generate it again."))
        return
    with st.spinner(bt("正在生成并验证预处理代码，失败时将自动修复，最多尝试5次...", "Generating and validating preprocessing code; up to five repair attempts...")):
        result = call_preprocessing_workflow(df, phase="validated_code", phase1_ctx=ctx)
    code = str((result or {}).get("code") or "").strip()
    if not code:
        st.error(bt("未能生成预处理代码。", "No preprocessing code was generated."))
        return
    agent.save_code(code)
    st.session_state.prep_code_visible = True
    attempts = int((result or {}).get("attempts") or 0)
    if (result or {}).get("success"):
        state["phase1_ctx"] = (result or {}).get("_ctx") or ctx
        record_validated_code(state, code, attempts=attempts)
        st.success(bt(f"代码已生成并通过验证（第{attempts}/5次）。请点击执行预处理。", f"Code passed validation on attempt {attempts}/5. Run preprocessing to publish the result."))
    else:
        error_text = str((result or {}).get("error") or "未生成可执行代码。")
        record_validation_failure(state, code, error_text, attempts=attempts or 5)
        st.error(bt("预处理代码连续5次未通过验证，已停止自动修复。", "Preprocessing code failed validation five times; auto-repair stopped."))
    st.rerun()


def _repair_prep_code_draft(agent, df: pd.DataFrame) -> None:
    state = get_suggestion_state(st.session_state, "preprocessing")
    ctx = state.get("phase1_ctx")
    error_text = str(state.get("last_execution_error") or "")
    if not isinstance(ctx, dict) or not error_text or not can_auto_repair(state):
        return
    state["repair_in_progress"] = True
    with st.spinner(bt("正在自动修复代码，失败时将继续修复，最多尝试5次...", "Automatically repairing code; up to five attempts...")):
        result = call_preprocessing_workflow(df, phase="validated_code", phase1_ctx=ctx, code=str(agent.load_code() or ""))
    code = str((result or {}).get("code") or "").strip()
    if not code:
        state["repair_in_progress"] = False
        st.error(bt("未能生成修复后的代码。", "No repaired code was generated."))
        return
    agent.save_code(code)
    attempts = int((result or {}).get("attempts") or 0)
    if (result or {}).get("success"):
        record_validated_code(state, code, attempts=attempts)
        st.success(bt("代码已修复并通过验证，请点击执行预处理。", "Code was repaired and validated. Run preprocessing to publish the result."))
    else:
        record_validation_failure(state, code, str((result or {}).get("error") or error_text), attempts=attempts or 5)
        st.error(bt("自动修复已达到5次，未能生成可执行代码。", "Auto-repair reached five attempts without a valid script."))
    st.session_state.prep_code_visible = True
    st.rerun()


def _revise_prep_code_draft(agent, df: pd.DataFrame, revision_instruction: str) -> None:
    state = get_suggestion_state(st.session_state, "preprocessing")
    ctx = state.get("phase1_ctx")
    current_code = str(agent.load_code() or "").strip()
    if not isinstance(ctx, dict) or not current_code:
        st.error(bt("当前预处理代码上下文已失效，请重新生成代码。", "The preprocessing code context has expired. Generate the code again."))
        return

    repair_prompt = (
        "User requested a code revision. Modify the current code to satisfy this instruction "
        "while preserving the confirmed preprocessing suggestion:\n"
        f"{revision_instruction}"
    )
    with st.spinner(bt("正在按你的意见修改并验证预处理代码...", "Revising and validating preprocessing code...")):
        repaired = call_preprocessing_workflow(
            df,
            phase="repair_code",
            phase1_ctx=ctx,
            code=current_code,
            error=repair_prompt,
        )
        revised_code = str((repaired or {}).get("code") or "").strip()
        if not revised_code:
            st.error(bt("未能生成修改后的预处理代码。", "No revised preprocessing code was generated."))
            return
        result = call_preprocessing_workflow(
            df,
            phase="validated_code",
            phase1_ctx=ctx,
            code=revised_code,
        )

    code = str((result or {}).get("code") or revised_code).strip()
    agent.save_code(code)
    st.session_state.prep_code_visible = True
    attempts = int((result or {}).get("attempts") or 0)
    if (result or {}).get("success"):
        record_validated_code(state, code, attempts=attempts)
        st.success(bt("代码已按你的意见修改并通过验证，请点击执行预处理。", "The code was revised and validated. Run preprocessing to publish the result."))
    else:
        record_validation_failure(
            state,
            code,
            str((result or {}).get("error") or "修改后的代码未通过验证。"),
            attempts=attempts or 5,
        )
        st.error(bt("修改后的预处理代码未通过验证。", "The revised preprocessing code did not pass validation."))
    st.rerun()


def _handle_prep_workflow_result(agent, workflow_result: dict[str, Any]) -> None:
    abstract_2 = workflow_result.get("abstract_2")
    summary_2 = workflow_result.get("summary_2")
    suggestion = _extract_suggestion_text(workflow_result)
    processed_payload = summary_2.get("processed_df") if isinstance(summary_2, dict) else None
    process_df = _coerce_processed_dataframe(processed_payload)
    workflow_succeeded = bool(
        workflow_result.get("_status") == "succeeded"
        and workflow_result.get("_code_success") is not False
        and isinstance(summary_2, dict)
        and process_df is not None
    )

    if not workflow_succeeded:
        error_text = str(
            workflow_result.get("_code_error")
            or (summary_2.get("error") if isinstance(summary_2, dict) else "")
            or (
                bt(
                    "预处理工作流返回的处理后数据格式无效。",
                    "The preprocessing workflow returned an invalid processed-data payload.",
                )
                if processed_payload is not None and process_df is None
                else ""
            )
            or bt("预处理执行失败。", "Preprocessing execution failed.")
        )
        record_stage_status(
            st.session_state,
            "preprocessing",
            "failed",
            input_fingerprint=current_dataset_fingerprint(st.session_state),
            error=error_text,
        )
        failed_code = str(summary_2.get("code") or "") if isinstance(summary_2, dict) else ""
        attempts = int(workflow_result.get("_fix_attempts") or 5)
        if failed_code:
            agent.save_code(failed_code)
            st.session_state.prep_code_visible = True
            record_validation_failure(
                get_suggestion_state(st.session_state, "preprocessing"),
                failed_code,
                error_text,
                attempts=attempts,
            )
        st.session_state.preprocessing_failure = {
            "error": error_text,
            "code": failed_code,
        }
        agent.save_error(error_text)
        if suggestion:
            agent.save_preprocessing_suggestions(suggestion)
        if st.session_state.get("auto_mode"):
            st.session_state.auto_mode = False
            st.session_state.auto_mode_paused_stage = "preprocessing"
        st.rerun()

    invalidate_from(
        st.session_state,
        "preprocessing",
        include_source=True,
        reason="preprocessing result replaced",
    )

    if abstract_2 is not None:
        st.session_state.abstract_2 = abstract_2

    if summary_2 is not None:
        st.session_state.summary_2 = summary_2
        st.session_state.prep_result_from_summary_2 = process_df
        st.session_state.preprocessing_qc_summary = summary_2.get("qc_summary") or {}
        agent.save_processed_df(process_df)

    if suggestion:
        st.session_state.suggestion = suggestion
        agent.save_preprocessing_suggestions(suggestion)

    agent.add_memory(
        {
            "role": "assistant",
            "content": {
                "abstract_2": abstract_2,
                "summary_2": summary_2,
                "suggestion": suggestion,
            },
        }
    )

    if isinstance(summary_2, dict) and summary_2.get("code"):
        code = str(summary_2.get("code") or "")
        agent.save_code(code)
        st.session_state.prep_code_visible = True
        record_successful_code(get_suggestion_state(st.session_state, "preprocessing"), code)
    st.session_state.pop("preprocessing_failure", None)

    output_fingerprint = dataframe_fingerprint(process_df)
    st.session_state.analysis_dataset_fingerprint = output_fingerprint
    record_stage_status(
        st.session_state,
        "preprocessing",
        "succeeded",
        input_fingerprint=current_dataset_fingerprint(st.session_state),
        output_fingerprint=output_fingerprint,
    )

    agent.finish_auto()
    st.rerun()


def prep_chat(agent, auto: bool = False) -> None:
    state = get_suggestion_state(st.session_state, "preprocessing")
    with st.chat_message("assistant"):
        st.write(
            bt(
                "我是 Autostat 数据分析助手。\n\n"
                "你可以在下方输入预处理需求，或者直接点击按钮获取预处理推荐。",
                "I am the Autostat data analysis assistant.\n\n"
                "Enter your preprocessing requirements below, or get an automatic preprocessing recommendation.",
            )
        )

        failure = st.session_state.get("preprocessing_failure")
        if isinstance(failure, dict) and failure.get("error"):
            st.error(bt("上一次预处理代码未通过验证。", "The previous preprocessing code did not pass validation."))
            # This function is rendered inside the outer "Preprocessing Suggestions"
            # expander. Streamlit does not allow expanders to be nested.
            st.caption(bt("错误详情", "Error details"))
            st.code(str(failure["error"]), language="text")

        columns = st.columns(2)
        with columns[0]:
            analyze_btn = st.button(
                bt("🔍 预处理推荐", "🔍 Preprocessing Recommendation"),
                key="prep_suggest",
                use_container_width=True,
                disabled=bool(state.get("active_suggestion")),
            )
        with columns[1]:
            clear_prep_suggest = st.button(
                bt("♻️ 清除预处理分析", "♻️ Clear Preprocessing Analysis"),
                key="clear_prep_suggest",
                use_container_width=True,
            )

        if clear_prep_suggest:
            _clear_prep_workflow_state(agent)
            st.rerun()

    for entry in visible_messages(state):
        role = entry.get("role")
        content = entry.get("content")
        with st.chat_message(role):
            st.write(str(content))

    pending_initial_request = take_pending_initial_request(state)
    if pending_initial_request:
        df = agent.load_df()
        if df is None:
            st.warning(bt("请先在数据导入页面加载数据。", "Please load data on the data import page first."))
        else:
            request_text = base_requirements_text(state, pending_initial_request)
            agent.save_user_input(request_text)
            _request_prep_recommendation(agent, df, request_text, auto=False)
        return

    pending_revision = take_pending_revision(state)
    if pending_revision:
        if isinstance(state.get("phase1_ctx"), dict):
            _revise_prep_recommendation(agent, pending_revision)
        else:
            df = agent.load_df()
            if df is None:
                st.warning(bt("请先在数据导入页面加载数据。", "Please load data on the data import page first."))
            else:
                st.warning(bt(
                    "上一轮预处理建议上下文已失效，正在基于当前数据和这条消息重新生成建议。",
                    "The previous preprocessing context expired. Regenerating from the current data and this message.",
                ))
                request_text = revision_fallback_text(
                    state,
                    pending_revision,
                    default=bt("请给我预处理建议", "Please provide preprocessing recommendations."),
                )
                agent.save_user_input(request_text)
                _request_prep_recommendation(agent, df, request_text, auto=False)
        return

    pending_code_revision = take_pending_code_revision(state)
    if pending_code_revision:
        df = agent.load_df()
        if df is None:
            st.warning(bt("请先在数据导入页面加载数据。", "Please load data on the data import page first."))
        else:
            _revise_prep_code_draft(agent, df, pending_code_revision)
        return

    already_generated = bool(state.get("active_suggestion"))

    if st.session_state.pop("_prep_code_repair_requested", False):
        df = agent.load_df()
        if df is not None:
            _repair_prep_code_draft(agent, df)
        return

    if st.session_state.pop("_prep_phase2_requested", False):
        if confirm_active_suggestion(state):
            df = agent.load_df()
            if df is not None:
                _generate_prep_code_draft(agent, df)
        return

    if auto and _has_prep_result(agent) and not agent.finish_auto_task:
        agent.finish_auto()
        st.rerun()

    if analyze_btn or (auto and not already_generated):
        df = agent.load_df()
        if df is None:
            st.warning(bt("请先在数据导入页面加载数据。", "Please load data on the data import page first."))
            return

        prompt_text = bt("请给我预处理建议", "Please provide preprocessing recommendations.")
        if not state.get("base_requirements"):
            add_requirement(state, prompt_text)
        request_text = base_requirements_text(state, prompt_text)
        agent.save_user_input(request_text)
        _request_prep_recommendation(agent, df, request_text, auto=auto)
        return

    user_input = st.chat_input(bt(
        "请输入预处理要求；建议生成后可继续提出修改意见",
        "Enter preprocessing requirements; after generation, request revisions here",
    ))
    if user_input:
        if state.get("active_suggestion"):
            queue_revision_request(state, user_input)
            st.rerun()
        else:
            queue_initial_request(state, user_input)
            st.rerun()
        return


if __name__ == "__main__":
    st.title(bt("数据预处理与标准化", "Data Preprocessing and Standardization"))
    st.markdown("---")

    data_loading_agent = st.session_state.data_loading_agent
    df = data_loading_agent.load_df()
    planner = st.session_state.planner_agent
    auto = bool(st.session_state.auto_mode and planner.prep_auto)

    if df is None:
        st.warning(bt("请先在数据导入页面加载数据。", "Please load data on the data import page first."))
        st.stop()

    agent = st.session_state.data_preprocess_agent
    agent.add_df(df)

    if st.session_state.auto_mode:
        if planner.prep_auto and _has_prep_result(agent):
            next_page = planner.finish_prep_auto()
            if next_page is not None:
                st.switch_page(next_page)
            st.session_state.auto_mode = False
            st.rerun()

    code = agent.load_code()
    code_expand = bool(st.session_state.get("prep_code_visible") and code is not None)

    columns = st.columns(2)
    with columns[0].expander(bt("预处理展示", "Preprocessing Overview"), True):
        prep_basic_info(agent)
    with columns[1].expander(bt("预处理建议", "Preprocessing Suggestions"), True):
        prep_chat(agent, auto)
        prep_code_gen(agent, auto=False)
    with columns[0].expander(bt("预处理执行", "Preprocessing Execution"), code_expand):
        prep_execution(agent, auto)
    with columns[0].expander(bt("预处理结果", "Preprocessing Result"), code_expand):
        prep_result(agent)
