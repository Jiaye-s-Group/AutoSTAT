import traceback

import numpy as np
import pandas as pd
import streamlit as st
from core.bounded_code_execution import (
    PREPROCESSING_TIMEOUT_SECONDS,
    run_bounded_safe_exec,
)
from core.preprocessing_contract import (
    format_preprocessing_contract_violations,
    validate_preprocessing_result,
)
from streamlit_ace import st_ace

from utils.i18n import bt, get_language
from utils.sanitize_code import sanitize_code
from utils.suggestion_state import (
    begin_code_execution,
    can_auto_repair,
    code_matches_current_suggestion,
    finish_code_execution,
    get_suggestion_state,
    mark_code_draft,
    record_execution_failure,
    record_validated_code,
    record_validation_failure,
)
from utils.workflow_state import (
    current_dataset_fingerprint,
    dataframe_fingerprint,
    invalidate_from,
    record_stage_status,
    stage_is_current,
)


def _show_execution_error(message: str, error_text: str) -> None:
    st.error(message)
    if error_text:
        st.code(error_text, language="text")


def _serialize_dataframe_for_workflow(df: pd.DataFrame) -> str:
    safe_df = df.copy()
    for column in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[column]):
            safe_df[column] = safe_df[column].astype(str)
    return safe_df.to_json(orient="records", force_ascii=False)


def _call_preprocessing_code_revision(
    df: pd.DataFrame,
    *,
    phase1_ctx: dict,
    code: str,
    error: str = "",
    phase: str,
) -> dict | None:
    from utils.local_workflow_bridge import call_preprocessing_bridge

    preview_df = df.head(10)
    return call_preprocessing_bridge(
        {
            "shape_0": int(df.shape[0]),
            "shape_1": int(df.shape[1]),
            "dtype_info_str": df.dtypes.astype(str).to_json(),
            "head_dict_str": preview_df.to_json(orient="records"),
            "df": _serialize_dataframe_for_workflow(df),
            "user_input": "",
            "prep_auto": True,
            "add_preference": st.session_state.get("add_preference") or "",
            "preference_selected": st.session_state.get("preference_selected") or "",
            "language": get_language(),
            "_phase": phase,
            "_phase1_ctx": phase1_ctx or {},
            "_code": code,
            "_error": error,
        }
    )


def _revise_preprocessing_code_from_execution(
    agent,
    df: pd.DataFrame,
    revision_instruction: str,
    *,
    current_code_override: str = "",
) -> None:
    state = get_suggestion_state(st.session_state, "preprocessing")
    ctx = state.get("phase1_ctx")
    current_code = str(current_code_override or agent.load_code() or "").strip()
    if not isinstance(ctx, dict) or not current_code:
        st.error(bt("当前预处理代码上下文已失效，请重新生成代码。", "The preprocessing code context has expired. Generate the code again."))
        return

    repair_prompt = (
        "User requested a code revision. Modify the current code to satisfy this instruction "
        "while preserving the confirmed preprocessing suggestion:\n"
        f"{revision_instruction}"
    )
    with st.spinner(bt("正在按你的意见修改并验证预处理代码...", "Revising and validating preprocessing code...")):
        agent.save_code(current_code)
        repaired = _call_preprocessing_code_revision(
            df,
            phase1_ctx=ctx,
            code=current_code,
            error=repair_prompt,
            phase="repair_code",
        )
        revised_code = str((repaired or {}).get("code") or "").strip()
        if not revised_code:
            st.session_state.prep_code_revision_flash = {
                "level": "error",
                "message": bt("未能生成修改后的预处理代码。", "No revised preprocessing code was generated."),
            }
            st.rerun()
        result = _call_preprocessing_code_revision(
            df,
            phase1_ctx=ctx,
            code=revised_code,
            phase="validated_code",
        )

    code = str((result or {}).get("code") or revised_code).strip()
    agent.save_code(code)
    st.session_state.prep_code_visible = True
    attempts = int((result or {}).get("attempts") or 0)
    if (result or {}).get("success"):
        record_validated_code(state, code, attempts=attempts)
        st.session_state.prep_code_revision_flash = {
            "level": "success",
            "message": bt(
                "代码已按你的意见修改并通过验证，请点击执行预处理。",
                "The code was revised and validated. Run preprocessing to publish the result.",
            ),
        }
    else:
        record_validation_failure(
            state,
            code,
            str((result or {}).get("error") or "修改后的代码未通过验证。"),
            attempts=attempts or 5,
        )
        st.session_state.prep_code_revision_flash = {
            "level": "error",
            "message": bt(
                "修改后的预处理代码未通过验证。",
                "The revised preprocessing code did not pass validation.",
            ),
        }
    st.rerun()


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
    state = get_suggestion_state(st.session_state, "preprocessing")
    tracked_execution = bool(state.get("executed_code_fingerprint"))
    result_is_current = bool(
        tracked_execution
        and state.get("current_code_fingerprint") == state.get("executed_code_fingerprint")
    )
    if edited is not None:
        _, result_is_current = mark_code_draft(state, edited)
        if code_matches_current_suggestion(state):
            st.caption(bt("代码状态：已同步当前建议。", "Code status: synced with the current suggestion."))
        else:
            st.warning(
                bt(
                    "建议已更新，当前代码仍基于旧建议。请重新生成代码，或让 AI 将当前代码迁移到新建议。",
                    "The suggestion was updated; the current code is still based on an older suggestion. Regenerate the code or ask AI to migrate it.",
                )
            )
        if agent.load_processed_df() is not None and tracked_execution and not result_is_current:
            st.warning(
                bt(
                    "代码已修改，当前预处理结果来自旧代码。请重新执行后再继续。",
                    "The code has changed. The current preprocessing result came from older code; run it again before continuing.",
                )
            )

    stale_result = tracked_execution and not result_is_current
    not_generated = agent.load_processed_df() is None or stale_result

    if code is not None:
        flash = st.session_state.pop("prep_code_revision_flash", None)
        if isinstance(flash, dict):
            message = str(flash.get("message") or "")
            if flash.get("level") == "success":
                st.success(message)
            elif flash.get("level") == "error":
                st.error(message)
            elif message:
                st.info(message)

        with st.form("prep_code_revision_form", clear_on_submit=True):
            revision_text = st.text_area(
                bt("代码修改要求", "Code Change Request"),
                placeholder=bt(
                    "例如：保留所有原始记录，仅对缺失的数值字段进行简单填补。",
                    "For example, retain all original records and apply simple imputation only to missing numeric fields.",
                ),
                height=90,
            )
            revise_clicked = st.form_submit_button(bt("让 AutoSTAT 修改代码", "Ask AutoSTAT to Revise Code"))
        if revise_clicked:
            if revision_text.strip():
                _revise_preprocessing_code_from_execution(
                    agent,
                    df,
                    revision_text,
                    current_code_override=sanitize_code(edited),
                )
            else:
                st.warning(bt("请输入代码修改要求。", "Enter a code change request."))

        execute_clicked = st.button(bt("▶️ 执行预处理", "▶️ Run Preprocessing")) or (auto and not_generated)
        if execute_clicked:
            code = sanitize_code(edited)
            run_id = begin_code_execution(state, code)
            agent.save_code(code)

            try:
                with st.spinner(bt("正在运行预处理脚本...", "Running the preprocessing script...")):
                    execution_result = run_bounded_safe_exec(
                        kind="preprocessing",
                        code=code,
                        dataframe=df,
                        timeout_seconds=PREPROCESSING_TIMEOUT_SECONDS,
                    )
                if not execution_result["is_success"]:
                    raise RuntimeError(str(execution_result["error"]))
            except Exception:
                error_text = traceback.format_exc()
                finish_code_execution(state, run_id, success=False)
                agent.save_error(error_text)
                record_execution_failure(state, error_text)
                if agent.load_processed_df() is None:
                    record_stage_status(
                        st.session_state,
                        "preprocessing",
                        "failed",
                        input_fingerprint=current_dataset_fingerprint(st.session_state),
                        error=error_text,
                    )
            else:
                process_df = execution_result.get("value")
                if process_df is None:
                    message = bt(
                        "脚本未写入 `process_df`。请重新生成代码，并确保脚本末尾为 `process_df` 赋值。",
                        "The script did not write `process_df`. Regenerate the code and make sure the script assigns `process_df` at the end.",
                    )
                    agent.save_error(message)
                    finish_code_execution(state, run_id, success=False)
                    record_execution_failure(state, message)
                    if agent.load_processed_df() is None:
                        record_stage_status(
                            st.session_state,
                            "preprocessing",
                            "failed",
                            input_fingerprint=current_dataset_fingerprint(st.session_state),
                            error=message,
                        )
                else:
                    if not isinstance(process_df, pd.DataFrame):
                        if isinstance(process_df, np.ndarray):
                            process_df = pd.DataFrame(process_df)
                        else:
                            message = bt(
                                f"期望 pandas.DataFrame 或 numpy.ndarray，收到 {type(process_df)}。请重新生成代码。",
                                f"Expected pandas.DataFrame or numpy.ndarray, but received {type(process_df)}. Please regenerate the code.",
                            )
                            agent.save_error(message)
                            finish_code_execution(state, run_id, success=False)
                            record_execution_failure(state, message)
                            if agent.load_processed_df() is None:
                                record_stage_status(
                                    st.session_state,
                                    "preprocessing",
                                    "failed",
                                    input_fingerprint=current_dataset_fingerprint(st.session_state),
                                    error=message,
                                )
                            return None

                    phase1_ctx = state.get("phase1_ctx")
                    if isinstance(phase1_ctx, dict):
                        from workflows.preprocessing import ensure_preprocessing_contract

                        phase1_ctx = ensure_preprocessing_contract(phase1_ctx)
                        state["phase1_ctx"] = phase1_ctx
                    preprocessing_contract = (
                        phase1_ctx.get("preprocessing_contract")
                        if isinstance(phase1_ctx, dict)
                        else {}
                    )
                    qc_summary = (
                        (execution_result.get("metadata") or {}).get("qc_summary")
                        if isinstance(execution_result.get("metadata"), dict)
                        else None
                    )
                    contract_issues = validate_preprocessing_result(
                        input_df=df,
                        output_df=process_df,
                        qc_summary=qc_summary,
                        contract=preprocessing_contract,
                    )
                    if contract_issues:
                        message = format_preprocessing_contract_violations(contract_issues)
                        agent.save_error(message)
                        finish_code_execution(state, run_id, success=False)
                        record_execution_failure(state, message)
                        if agent.load_processed_df() is None:
                            record_stage_status(
                                st.session_state,
                                "preprocessing",
                                "failed",
                                input_fingerprint=current_dataset_fingerprint(st.session_state),
                                error=message,
                            )
                        return None

                    invalidate_from(
                        st.session_state,
                        "preprocessing",
                        reason="preprocessing result replaced",
                    )
                    agent.save_processed_df(process_df)
                    st.session_state.preprocessing_qc_summary = qc_summary or {}
                    finish_code_execution(state, run_id, success=True)
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
                    return process_df

    error_text = str(state.get("last_execution_error") or "")
    if error_text:
        st.error(bt("执行失败", "Execution failed"))
        # This function is rendered inside the outer "Preprocessing Execution"
        # expander. Streamlit does not allow expanders to be nested.
        st.caption(bt("错误详情", "Error details"))
        st.code(error_text, language="text")
        attempts = int(state.get("auto_repair_attempts") or 0)
        if can_auto_repair(state):
            repair_slot = st.empty()
            with repair_slot.container():
                repair_requested = st.button(
                    bt("AutoSTAT 自动修复代码", "AutoSTAT Auto-fix Code"),
                    key="prep_auto_fix_code",
                )
                if attempts:
                    st.caption(bt(
                        f"本轮已自动修复 {attempts} 次，最多 5 次。",
                        f"This round has used {attempts} of 5 automatic repairs.",
                    ))
            if repair_requested:
                repair_slot.empty()
                st.session_state._prep_code_repair_requested = True
                st.rerun()
        else:
            st.caption(bt(
                "已达到自动修复上限，请手动修改代码或调整建议后重新生成。",
                "The automatic repair limit has been reached. Edit the code or revise the suggestion before generating again.",
            ))


def prep_code_gen(agent, auto=False, debug=False):
    suggest = agent.load_preprocessing_suggestions()
    control_slot = st.empty()

    summary_2 = st.session_state.get("summary_2")
    workflow_code = ""
    if isinstance(summary_2, dict):
        workflow_code = str(summary_2.get("code") or "").strip()

    if workflow_code:
        code_is_loaded = (
            str(agent.load_code() or "").strip() == workflow_code
            and bool(st.session_state.get("prep_code_visible"))
        )
        analyze_btn = False
        with control_slot.container():
            if not code_is_loaded:
                analyze_btn = st.button(
                    bt("🔡 生成预处理代码", "🔡 Generate Preprocessing Code"),
                    key="prep_code",
                )
        if analyze_btn or (auto and not code_is_loaded):
            control_slot.empty()
            agent.save_code(workflow_code)
            st.session_state.prep_code_visible = True
            workflow_loaded_message = bt(
                "预处理代码已从工作流加载，请在下方执行。",
                "Preprocessing code has been loaded from the workflow. Run it below.",
            )
            st.chat_message("assistant").write(workflow_loaded_message)
            agent.add_memory(
                {"role": "assistant", "content": workflow_loaded_message}
            )
            st.rerun()
        return

    if suggest is not None:
        code_is_loaded = bool(agent.load_code()) and bool(st.session_state.get("prep_code_visible"))
        analyze_btn = False
        with control_slot.container():
            if not code_is_loaded:
                analyze_btn = st.button(
                    bt("🔡 生成预处理代码", "🔡 Generate Preprocessing Code"),
                    key="prep_code",
                )
        if analyze_btn or (auto and not code_is_loaded):
            state = get_suggestion_state(st.session_state, "preprocessing")
            if isinstance(state.get("phase1_ctx"), dict):
                st.session_state._prep_phase2_requested = True
            else:
                st.error(
                    bt(
                        "当前预处理建议上下文已失效，请清除后重新生成建议。",
                        "The preprocessing recommendation context has expired. Clear it and generate a new recommendation.",
                    )
                )
                return
            control_slot.empty()
            st.rerun()
