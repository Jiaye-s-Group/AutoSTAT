import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from streamlit_ace import st_ace

from utils.page_paths import page_file
from utils.sanitize_code import sanitize_code
from workflow.modeling.model_training import (
    modeling_code_gen,
    train_download_model,
    train_execution,
)

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
        return "，".join(values)

    if isinstance(user_selection, str):
        return user_selection.strip()

    return ""


def _resolve_effective_target(target_value: str, user_prompt: str) -> str:
    target_value = (target_value or "").strip()
    if target_value:
        return target_value
    return user_prompt


def _build_modeling_inputs(
    source_data: Any,
    agent,
    user_input: str,
    target_value: str,
    history_train_code: str,
    modeling_auto: bool = True,
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
        st.info("暂无结果内容。")
        return

    normalized = text.replace("\r\n", "\n").strip()
    if "\n" in normalized:
        paragraphs = [segment.strip() for segment in normalized.split("\n\n") if segment.strip()]
        pretty_text = "\n\n".join(paragraphs) if paragraphs else normalized
        st.markdown(pretty_text)
        return

    st.markdown(f"> {normalized}")


def call_modeling_workflow(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Call the local modeling workflow."""
    from utils.local_workflow_bridge import call_modeling_bridge

    inputs = dict(inputs)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    inputs.setdefault("preference_selected", st.session_state.get("preference_selected") or "")

    result = call_modeling_bridge(inputs)
    if result is None:
        return None

    normalized = _normalize_modeling_workflow_result(result)
    if normalized is None:
        st.error("建模工作流返回结构异常，未解析到有效结果。")
        return None
    return normalized


def _call_modeling_phase1(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Run modeling phase 1 for recommendation preview."""
    from utils.local_workflow_bridge import call_modeling_phase1_bridge

    inputs = dict(inputs)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    inputs.setdefault("preference_selected", st.session_state.get("preference_selected") or "")
    return call_modeling_phase1_bridge(inputs)


def _call_modeling_phase2(inputs: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Run modeling phase 2 for code generation and result analysis."""
    from utils.local_workflow_bridge import call_modeling_phase2_bridge

    inputs = dict(inputs)
    inputs.setdefault("add_preference", st.session_state.get("add_preference") or "")
    inputs.setdefault("preference_selected", st.session_state.get("preference_selected") or "")

    result = call_modeling_phase2_bridge(inputs, ctx)
    if result is None:
        return None

    normalized = _normalize_modeling_workflow_result(result)
    if normalized is None:
        st.error("建模代码生成阶段返回结构异常。")
        return None
    return normalized


def _clear_modeling_workflow_state(agent) -> None:
    agent.clear_memory()
    agent.save_suggestion(None)
    agent.save_code(None)
    agent.save_modeling_result(None)
    _agent_save_value(agent, "save_user_input", "user_input", None)
    _agent_save_value(agent, "save_user_selection", "user_selection", None)
    _agent_save_value(agent, "save_target", "target", "")
    _agent_save_value(agent, "save_history_train_code", "history_train_code", "")
    st.session_state.history_train_code_reset_pending = True
    for key in (
        "modeling_workflow_result", "modeling_suggestion", "model_suggestion",
        "modeling_summary_4", "modeling_abstract_4", "summary_4", "abstract_4",
        "modeling_result_from_summary_4", "modeling_user_prompt",
        "_model_phase1_ctx", "_model_phase2_inputs", "_model_phase2_pending",
    ):
        st.session_state.pop(key, None)


def _reset_modeling_outputs(agent) -> None:
    agent.save_suggestion(None)
    agent.save_code(None)
    agent.save_modeling_result(None)
    for key in (
        "modeling_workflow_result", "modeling_suggestion", "model_suggestion",
        "modeling_summary_4", "modeling_abstract_4", "summary_4", "abstract_4",
        "modeling_result_from_summary_4",
        "_model_phase1_ctx", "_model_phase2_inputs", "_model_phase2_pending",
    ):
        st.session_state.pop(key, None)


def _request_modeling_recommendation(
    agent,
    source_data: Any,
    user_input: str,
    target_value: str,
    history_train_code: str,
) -> None:
    _reset_modeling_outputs(agent)

    inputs = _build_modeling_inputs(
        source_data=source_data,
        agent=agent,
        user_input=user_input,
        target_value=target_value,
        history_train_code=history_train_code,
        modeling_auto=True,
    )
    if inputs is None:
        st.error("无法从当前可用数据构造建模工作流输入，请检查预处理结果或原始上传数据是否可解析。")
        return

    # Phase 1 returns the recommendation text immediately.
    with st.spinner("正在生成建模推荐方案..."):
        phase1_result = _call_modeling_phase1(inputs)

    if not phase1_result:
        return

    suggestion = phase1_result.get("model_suggestion", "")
    phase1_ctx = phase1_result.get("_ctx", {})

    # Store the recommendation so the UI can render it before phase 2.
    st.session_state.modeling_suggestion = suggestion
    st.session_state.model_suggestion = suggestion
    agent.save_suggestion(suggestion)

    # Keep phase 1 context and inputs for phase 2.
    st.session_state._model_phase1_ctx = phase1_ctx
    st.session_state._model_phase2_inputs = inputs
    st.session_state._model_phase2_pending = True

    agent.add_memory({"role": "assistant", "content": suggestion})
    st.rerun()


def _continue_modeling_phase2(agent) -> None:
    """Continue modeling after the recommendation preview has rendered."""
    phase1_ctx = st.session_state.pop("_model_phase1_ctx", None)
    inputs = st.session_state.pop("_model_phase2_inputs", None)
    st.session_state.pop("_model_phase2_pending", None)

    if not phase1_ctx or not inputs:
        return

    with st.spinner("建模建议已生成，正在生成代码与训练分析..."):
        workflow_result = _call_modeling_phase2(inputs, phase1_ctx)

    if not workflow_result:
        return

    st.session_state.modeling_workflow_result = workflow_result
    st.session_state.modeling_summary_4 = workflow_result.get("summary_4")
    st.session_state.modeling_abstract_4 = workflow_result.get("abstract_4")
    st.session_state.summary_4 = workflow_result.get("summary_4")
    st.session_state.abstract_4 = workflow_result.get("abstract_4")

    summary_4 = workflow_result.get("summary_4")
    if isinstance(summary_4, dict):
        workflow_code = str(summary_4.get("code") or workflow_result.get("_final_code") or "").strip()
        if workflow_code:
            agent.save_code(workflow_code)
        result_text = summary_4.get("result")
        if result_text is not None:
            agent.save_modeling_result(result_text)

    agent.add_memory({"role": "assistant", "content": workflow_result})
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
    return has_code and has_result


def _has_report_prerequisites(agent) -> bool:
    return bool(
        st.session_state.get("summary_1")
        and st.session_state.get("summary_2")
        and st.session_state.get("summary_3")
        and _has_completed_modeling_result(agent)
    )


def modeling_quick_actions(agent):
    st.write("选择一个或多个 model:")
    selected_models = sac.chip(
        items=[
            sac.ChipItem(label="线性回归"),
            sac.ChipItem(label="XGBoost"),
            sac.ChipItem(label="随机森林"),
            sac.ChipItem(label="神经网络"),
        ],
        index=[0, 2],
        align="center",
        direction="horizontal",
        size="sm",
        radius="md",
        color="#44658C",
        multiple=True,
    )

    if st.button("🖋️ 快速建模", key="quick_modeling"):
        if not selected_models:
            st.error("请先选择训练 model。")
        else:
            agent.save_user_selection(selected_models)
            st.session_state.modeling_user_prompt = _format_user_prompt(selected_models)
            st.success("已保存快速建模选择，后续会作为 user_prompt 传入建模工作流。")
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

    not_executed = agent.load_modeling_result() is None

    if edited is not None:
        if st.button("▶️ 执行建模", key="modeling_run_code") or (auto and not_executed):
            code = sanitize_code(edited)
            agent.save_code(code)
            train_execution(agent)
            st.session_state.modeling_result_from_summary_4 = _extract_summary_4_result()
            agent.finish_auto()
            st.rerun()

        modeling_result = agent.load_modeling_result()
        summary_result = st.session_state.get("modeling_result_from_summary_4")
        if summary_result is not None or modeling_result is not None:
            train_download_model(agent)
            with st.container():
                st.subheader("训练结果")
                if summary_result is not None:
                    _render_modeling_result(summary_result)
                else:
                    _render_modeling_result(modeling_result)


def modeling_chat(agent, source_data: Any, auto: bool) -> None:
    current_target = _agent_load_value(agent, "load_target", "target", "") or ""
    target_value = st.text_input("建模目标", value=current_target, placeholder="请输入建模目标")
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
        "历史训练代码",
        key="history_train_code_input",
        placeholder="若有历史训练代码可在此输入，也可点击下方按钮同步当前执行区代码。",
        height=180,
    )
    _agent_save_value(agent, "save_history_train_code", "history_train_code", history_train_code)

    st.button(
        "获取当前执行区代码",
        key="sync_history_train_code",
        on_click=_sync_history_train_code_from_execution,
        args=(agent,),
    )

    with st.chat_message("assistant"):
        st.write(
            "我是 Autostat 数据分析助手。\n\n"
            "你可以在下方输入建模相关问题，或直接点击按钮获取建模推荐。"
        )

        columns = st.columns(2)
        with columns[0]:
            analyze_btn = st.button("🔍 建模推荐", key="modeling_suggest", use_container_width=True)
        with columns[1]:
            clear_modeling_suggest = st.button(
                "♻️ 清除建模分析",
                key="clear_modeling_suggest",
                use_container_width=True,
            )

        if clear_modeling_suggest:
            _clear_modeling_workflow_state(agent)
            st.rerun()

    chat_history = agent.load_memory()
    for entry in chat_history:
        role = entry.get("role")
        content = entry.get("content")
        if isinstance(content, str):
            st.chat_message(role).write(content)

    already_generated = bool(
        st.session_state.get("model_suggestion") or st.session_state.get("modeling_suggestion")
    )
    saved_user_input = _agent_load_value(agent, "load_user_input", "user_input", "") or ""

    # Continue phase 2 after the recommendation has rendered.
    if st.session_state.get("_model_phase2_pending"):
        _continue_modeling_phase2(agent)
        return

    if auto and _has_completed_modeling_result(agent) and not agent.finish_auto_task:
        agent.finish_auto()
        st.rerun()

    if analyze_btn or (auto and not already_generated):
        prompt_text = "请帮我获取建模建议"
        st.chat_message("user").write(prompt_text)
        agent.add_memory({"role": "user", "content": prompt_text})
        _request_modeling_recommendation(
            agent=agent,
            source_data=source_data,
            user_input=saved_user_input,
            target_value=target_value,
            history_train_code=history_train_code,
        )

    user_input = st.chat_input("请输入您的问题，例如“如何优化这个模型”")
    if user_input:
        agent.save_user_input(user_input)
        st.chat_message("user").write(user_input)
        agent.add_memory({"role": "user", "content": user_input})
        _request_modeling_recommendation(
            agent=agent,
            source_data=source_data,
            user_input=user_input,
            target_value=target_value,
            history_train_code=history_train_code,
        )


if __name__ == "__main__":
    st.title("数据建模")
    st.markdown("---")

    preproc_agent = st.session_state.data_preprocess_agent
    load_agent = st.session_state.data_loading_agent
    source_data, source_kind = _resolve_modeling_source(preproc_agent, load_agent)
    df = _source_to_dataframe(source_data)

    if df is None:
        st.warning("请先在数据导入页面加载数据。")
        st.stop()

    agent = st.session_state.modeling_coding_agent
    agent.add_df(df)
    planner = st.session_state.planner_agent
    auto = bool(st.session_state.auto_mode and planner.modeling_auto)

    if st.session_state.auto_mode is True:
        if planner.modeling_auto and _has_report_prerequisites(agent):
            planner.finish_modeling_auto()
            st.switch_page(page_file("report", "report_render.py"))

    code = agent.load_code()

    if source_kind == "raw":
        st.caption("当前未对原始数据进行预处理，后续将基于原始数据进行分析")

    columns = st.columns(2)
    with columns[0].expander("快速建模", True):
        modeling_quick_actions(agent)
    with columns[1].expander("建模建议", True):
        modeling_chat(agent, source_data, auto)
        modeling_code_gen(agent, auto=auto)
    with columns[0].expander("建模执行", code is not None):
        modeling_execution(agent, auto)
