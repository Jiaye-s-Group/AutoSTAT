import traceback

import numpy as np
import pandas as pd
import streamlit as st
from core.safe_code import safe_exec
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

from utils.i18n import bt
from utils.sanitize_code import sanitize_code
from utils.suggestion_state import (
    begin_code_execution,
    can_auto_repair,
    finish_code_execution,
    get_suggestion_state,
    mark_code_draft,
    record_execution_failure,
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
        execute_clicked = st.button(bt("▶️ 执行预处理", "▶️ Run Preprocessing")) or (auto and not_generated)
        if execute_clicked:
            code = sanitize_code(edited)
            run_id = begin_code_execution(state, code)
            agent.save_code(code)

            exec_ns = {
                "df": df,
                "np": np,
                "pd": pd,
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
                with st.spinner(bt("正在运行预处理脚本...", "Running the preprocessing script...")):
                    safe_exec(code, exec_ns)
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
                process_df = exec_ns.get("process_df")
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

                    invalidate_from(
                        st.session_state,
                        "preprocessing",
                        reason="preprocessing result replaced",
                    )
                    agent.save_processed_df(process_df)
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
        with st.expander(bt("查看错误详情", "View error details")):
            st.code(error_text, language="text")
        attempts = int(state.get("auto_repair_attempts") or 0)
        if can_auto_repair(state):
            repair_slot = st.empty()
            with repair_slot.container():
                repair_requested = st.button(
                    bt("自动修复代码", "Auto-fix Code"),
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
