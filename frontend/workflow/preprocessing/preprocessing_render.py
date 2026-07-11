import io
import json
from typing import Any

import pandas as pd
import streamlit as st

from utils.page_paths import page_file
from workflow.preprocessing.preprocessing_core import prep_code_gen, prep_meta_execution

# Local preprocessing workflow configuration.
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

    # The workflow accepts serialized data so it can run outside Streamlit.
    for column in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[column]):
            safe_df[column] = safe_df[column].astype(str)

    return safe_df.to_json(orient="records", force_ascii=False)


def call_preprocessing_workflow(
    df: pd.DataFrame,
    prep_auto: bool = True,
    user_input: str = "",
) -> dict[str, Any] | None:
    """Call the local preprocessing workflow."""
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
    }

    result = call_preprocessing_bridge(inputs)
    if result is None:
        return None

    normalized = _normalize_prep_workflow_result(result)
    if normalized is None:
        st.error("预处理工作流返回结构异常，未解析到有效结果。")
        return None
    return normalized


def prep_basic_info(agent) -> None:
    df = agent.load_df()

    row_count, col_count = df.shape
    missing_count = int(df.isnull().sum().sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("行数", row_count)
    col2.metric("列数", col_count)
    col3.metric("缺失值总数", missing_count)

    dtype_info = pd.DataFrame(
        {
            "列名": df.columns,
            "类型": df.dtypes.astype(str),
            "非空值数量": df.count().values,
            "缺失值比例(%)": (df.isnull().mean() * 100).round(2).values,
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

    if process_df is None and not workflow_processed_df:
        return

    st.write("处理前数据预览：", df.head(10))

    if process_df is not None:
        st.write("处理后数据预览：", process_df.head(10))

        csv_buffer = io.StringIO()
        process_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        st.download_button(
            label="⬇️ 下载处理后数据",
            data=csv_bytes,
            file_name="processed_data.csv",
            mime="text/csv",
        )
        return

    st.write("处理后数据预览：")
    st.write(workflow_processed_df)


def _clear_prep_workflow_state(agent) -> None:
    agent.preprocessing_suggestions = None
    agent.code = None
    agent.processed_df = None
    agent.error = None

    for key in ("suggestion", "abstract_2", "summary_2", "prep_result_from_summary_2", "prep_code_visible"):
        if key in st.session_state:
            del st.session_state[key]


def _has_prep_result(agent) -> bool:
    suggestion = st.session_state.get("suggestion") or agent.load_preprocessing_suggestions()
    return bool(suggestion)


def _request_prep_recommendation(agent, df: pd.DataFrame, user_input: str) -> None:
    with st.spinner("正在智能分析数据，预计需要 2-3 分钟，请耐心等待..."):
        workflow_result = call_preprocessing_workflow(
            df,
            prep_auto=True,
            user_input=user_input,
        )

    if not workflow_result:
        return

    abstract_2 = workflow_result.get("abstract_2")
    summary_2 = workflow_result.get("summary_2")
    suggestion = _extract_suggestion_text(workflow_result)

    if abstract_2 is not None:
        st.session_state.abstract_2 = abstract_2

    if summary_2 is not None:
        st.session_state.summary_2 = summary_2

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
        agent.save_code(None)
        st.session_state.prep_code_visible = False

    agent.finish_auto()
    st.rerun()


def prep_chat(agent, auto: bool = False) -> None:
    with st.chat_message("assistant"):
        st.write(
            "我是 Autostat 数据分析助手。\n\n"
            "你可以在下方输入预处理需求，或者直接点击按钮获取预处理推荐。"
        )

        columns = st.columns(2)
        with columns[0]:
            analyze_btn = st.button("🔍 预处理推荐", key="prep_suggest", use_container_width=True)
        with columns[1]:
            clear_prep_suggest = st.button(
                "♻️ 清除预处理分析",
                key="clear_prep_suggest",
                use_container_width=True,
            )

        if clear_prep_suggest:
            _clear_prep_workflow_state(agent)
            st.rerun()

    chat_history = agent.load_memory()
    for entry in chat_history:
        role = entry.get("role")
        content = entry.get("content")

        with st.chat_message(role):
            if isinstance(content, dict):
                suggestion = _extract_suggestion_text(content)
                if suggestion:
                    st.write(suggestion)
            elif isinstance(content, str):
                st.write(content)

    already_generated = bool(
        st.session_state.get("suggestion")
        or agent.load_preprocessing_suggestions()
        or st.session_state.get("summary_2")
    )

    if auto and _has_prep_result(agent) and not agent.finish_auto_task:
        agent.finish_auto()
        st.rerun()

    if analyze_btn or (auto and not already_generated):
        df = agent.load_df()
        if df is None:
            st.warning("请先在数据导入页面加载数据。")
            return

        prompt_text = "请给我预处理建议"
        st.chat_message("user").write(prompt_text)
        agent.add_memory({"role": "user", "content": prompt_text})

        _request_prep_recommendation(agent, df, prompt_text)
        return

    user_input = st.chat_input("请输入您的问题")
    if user_input:
        st.chat_message("user").write(user_input)
        agent.add_memory({"role": "user", "content": user_input})
        agent.save_user_input(user_input)

        df = agent.load_df()
        if df is None:
            return

        _request_prep_recommendation(agent, df, user_input)
        return


if __name__ == "__main__":
    st.title("数据预处理与标准化")
    st.markdown("---")

    data_loading_agent = st.session_state.data_loading_agent
    df = data_loading_agent.load_df()
    planner = st.session_state.planner_agent
    auto = bool(st.session_state.auto_mode and planner.prep_auto)

    if df is None:
        st.warning("请先在数据导入页面加载数据。")
        st.stop()

    agent = st.session_state.data_preprocess_agent
    agent.add_df(df)

    if st.session_state.auto_mode:
        if planner.prep_auto and _has_prep_result(agent):
            planner.finish_prep_auto()
            st.switch_page(page_file("visualization", "viz_render.py"))

    code = agent.load_code()
    code_expand = bool(st.session_state.get("prep_code_visible") and code is not None)

    columns = st.columns(2)
    with columns[0].expander("预处理展示", True):
        prep_basic_info(agent)
    with columns[1].expander("预处理建议", True):
        prep_chat(agent, auto)
        prep_code_gen(agent, auto=False)
    with columns[0].expander("预处理执行", code_expand):
        prep_execution(agent, auto)
    with columns[0].expander("预处理结果", code_expand):
        prep_result(agent)
