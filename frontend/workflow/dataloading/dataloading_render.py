import os
import json
from typing import Any

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac

from utils.page_paths import page_file
from workflow.dataloading.dataloading_core import process_complex_data, load_concat_file, PathFileWrapper

# --- Coze 工作流配置 ---
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


def _stringify_content(value: Any) -> str:
    value = _maybe_json_loads(value)

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return json.dumps(value, ensure_ascii=False, indent=2)


def _normalize_loading_workflow_result(result: Any) -> dict[str, Any] | None:
    result = _maybe_json_loads(result)
    if not isinstance(result, dict):
        return None

    normalized = dict(result)
    normalized["summary_1"] = _maybe_json_loads(_find_nested_field(result, "summary_1"))
    normalized["abstract_1"] = _stringify_content(_find_nested_field(result, "abstract_1"))
    return normalized


def _extract_summary_1_fields(summary_1: Any) -> dict[str, Any]:
    parsed_summary = _maybe_json_loads(summary_1)
    if not isinstance(parsed_summary, dict):
        return {"title": "", "desc": "", "df": None}

    return {
        "title": _stringify_content(parsed_summary.get("title")),
        "desc": _stringify_content(parsed_summary.get("desc")),
        "df": parsed_summary.get("df"),
    }


def _save_loading_workflow_outputs(agent, workflow_result: dict[str, Any]) -> None:
    summary_1 = workflow_result.get("summary_1", {})
    abstract_1 = workflow_result.get("abstract_1", "")
    summary_fields = _extract_summary_1_fields(summary_1)

    st.session_state.loading_workflow_result = workflow_result
    st.session_state.summary_1 = summary_1
    st.session_state.abstract_1 = abstract_1
    st.session_state.summary_1_title = summary_fields["title"]
    st.session_state.summary_1_desc = summary_fields["desc"]
    st.session_state.summary_1_df = summary_fields["df"]

    save_method = getattr(agent, "save_loading_workflow_result", None)
    if callable(save_method):
        save_method(workflow_result)
    else:
        agent.loading_workflow_result = workflow_result


def call_coze_workflow(
    df: pd.DataFrame,
    user_input: str = "",
    loading_auto: bool = True,
):
    """
    本地化版本：调用本地的 loading workflow，不再走 Coze。
    返回结构与原版一致：{summary_1, abstract_1}
    """
    from workflows.loading import run_loading_workflow
    from workflows._plugins import df_to_meta

    try:
        meta = df_to_meta(df)
        result = run_loading_workflow(
            shape_0=meta["shape_0"],
            shape_1=meta["shape_1"],
            dtype_info_str=meta["dtype_info_str"],
            head_dict_str=meta["head_dict_str"],
            loading_auto=loading_auto,
            user_input=user_input or "",
            add_preference=st.session_state.get("add_preference") or "",
            preference_selected=st.session_state.get("preference_selected") or "",
        )
        normalized = _normalize_loading_workflow_result(result)
        if normalized is None:
            st.error("数据导入工作流返回结构异常，未解析到有效结果。")
            return None
        return normalized
    except Exception as e:
        st.error(f"本地 Loading workflow 执行异常：{e}")
        return None

def loading_reference_docs(agent):
    """
    专门处理参考资料的上传逻辑
    """
    st.info("💡 提示：在此处上传业务背景、算法说明或数据手册 (PDF/Docx)，AI 会学习这些内容。")
    
    uploaded_docs = st.file_uploader(
        "上传参考文档",
        type=['pdf', 'docx', 'txt', 'names'],
        accept_multiple_files=True,
        key="ref_doc_uploader"
    )

    if uploaded_docs:
        if 'learned_doc_names' not in st.session_state:
            st.session_state.learned_doc_names = set()

        new_files = [f for f in uploaded_docs if f.name not in st.session_state.learned_doc_names]
        
        if new_files:
            if st.button("🧠 学习选中的资料", use_container_width=True):
                with st.spinner("正在解析文档并提取知识点..."):
                    count = st.session_state.retriever.add_uploaded_files(new_files)
                    for f in new_files:
                        st.session_state.learned_doc_names.add(f.name)
                st.success(f"学习成功！新增 {len(new_files)} 份文档，提取了 {count} 条知识片段。")
        else:
            st.caption("✅ 当前上传的文件已全部在知识库中。")

    if 'learned_doc_names' in st.session_state and st.session_state.learned_doc_names:
        st.markdown("**已加载的外部资料：**")
        for name in st.session_state.learned_doc_names:
            st.write(f"- 📄 {name}")

def loading_data_file(agent):
    """ """
    st.info(
        "💡 提示：\n"
        "1. 支持一次上传多个数据文件\n"
        "2. 自动使用大模型分析并处理数据\n"
        "3. 支持多种格式的文件类型上传\n"
    )

    selected_index = sac.tabs([
        sac.TabsItem(label='本地上传'),
        sac.TabsItem(label='路径导入'),
    ], color='#5980AE',)

    if selected_index == "本地上传":
        uploaded_files = st.file_uploader(
            "选择新文件",
            accept_multiple_files=True,
            help="拖拽或点击上传多个文件",
        )

        if uploaded_files:
            current_memory_file_name = agent.load_file_name()
            new_files = [f for f in uploaded_files if f.name not in current_memory_file_name]
            if new_files:
                try:
                    with st.spinner("正在处理数据..."):
                        df, dfs = process_complex_data(new_files, agent)
                    if df is not None:
                        agent.add_df(df)
                        agent.save_dfs(dfs)
                        for f in new_files:
                            agent.save_file_name(f.name)
                        st.rerun()
                except Exception as err:
                    st.error(f"导入失败：{err}")

    elif selected_index == "路径导入":
        raw_paths = st.text_area(
            "从路径导入数据 (每行一个文件路径)",
            placeholder="C:\\data\\iris.names\nC:\\data\\iris.data",
            height=100
        )

        if st.button("从路径加载文件", use_container_width=True):
            if raw_paths:
                path_list = [p.strip().strip("'\"") for p in raw_paths.strip().split('\n') if p.strip()]
                valid_paths = [p for p in path_list if os.path.exists(p)]
                invalid_paths = [p for p in path_list if not os.path.exists(p)]

                if invalid_paths:
                    st.warning(f"路径不存在，已跳过：\n- " + "\n- ".join(invalid_paths))

                if not valid_paths:
                    st.error("未找到任何有效的本地文件路径。")
                else:
                    current_memory_file_name = agent.load_file_name()
                    new_paths = [p for p in valid_paths if p not in current_memory_file_name]

                    if not new_paths:
                        st.info("所有指定的路径文件均已加载。")
                    else:
                        files_to_process = [PathFileWrapper(p) for p in new_paths]
                        try:
                            with st.spinner("正在处理数据..."):
                                df, dfs = process_complex_data(files_to_process, agent)
                            if df is not None:
                                agent.add_df(df)
                                agent.save_dfs(dfs)
                                for p in new_paths:
                                    agent.save_file_name(p)
                                st.rerun()
                        except Exception as err:
                            st.error(f"本地文件读取失败：{err}")
    
    dfs = agent.load_dfs()
    if dfs is not None and len(dfs) >= 2:
        load_concat_file(dfs, agent)

def loading_basic_info(agent):
    """ """
    df = agent.load_df()
    if df is not None:
        r, c = df.shape
        missing = int(df.isnull().sum().sum())
        col1, col2, col3 = st.columns(3)
        col1.metric("行数", r)
        col2.metric("列数", c)
        col3.metric("缺失值总数", missing)

        dtype_info = pd.DataFrame({
            "列名": df.columns,
            "类型": df.dtypes.astype(str),
            "非空": df.count().values,
            "缺失%": (df.isnull().mean() * 100).round(2).values,
        }).reset_index(drop=True)

        selected_index = sac.tabs([
            sac.TabsItem(label='数据类型概览'),
            sac.TabsItem(label='数据预览'),
        ],color='#5980AE',)

        if selected_index == "数据类型概览":
            st.dataframe(dtype_info, use_container_width=True)
        elif selected_index == "数据预览":
            if st.button("🎲 随机抽样"):
                display_df = df.sample(10)
                st.dataframe(display_df, use_container_width=True)
            else:
                st.dataframe(df.head(10), use_container_width=True)

def _extract_loading_display_text(workflow_result: dict[str, Any]) -> str:
    summary_fields = _extract_summary_1_fields(workflow_result.get("summary_1"))
    desc = summary_fields["desc"]
    if desc:
        return desc

    return "工作流已运行，但 summary_1.desc 为空。"


def _has_loading_result(agent) -> bool:
    if st.session_state.get("loading_workflow_result"):
        return True

    summary_1 = st.session_state.get("summary_1")
    abstract_1 = st.session_state.get("abstract_1")
    if summary_1 or abstract_1:
        return True

    load_method = getattr(agent, "load_loading_workflow_result", None)
    if callable(load_method) and load_method():
        return True

    for entry in reversed(agent.load_memory()):
        content = entry.get("content") if isinstance(entry, dict) else None
        if isinstance(content, dict) and (content.get("summary_1") or content.get("abstract_1")):
            return True

    return False


def _request_loading_analysis(agent, df: pd.DataFrame, user_input: str) -> None:
    with st.spinner("正在解析数据，请耐心等待..."):
        workflow_result = call_coze_workflow(
            df,
            user_input=user_input,
            loading_auto=True,
        )

    if not workflow_result:
        return

    agent.finish_auto()
    _save_loading_workflow_outputs(agent, workflow_result)
    agent.add_memory({"role": "assistant", "content": workflow_result})
    st.rerun()


def loading_chat(agent, auto=False) -> None:
    df = agent.load_df()
    if df is None:
        return

    with st.chat_message("assistant"):
        st.write(
            "我是 Autostat 数据分析助手，很高兴为您服务；\n\n"
            "请先上传您的数据文件，上传完成后，您可以在下方和我对话，也可以直接点击按钮解析数据含义。"
        )
        analyze_btn = st.button("🔍 解析含义")

    chat_history = agent.load_memory()
    for entry in chat_history:
        role = entry["role"]
        content = entry["content"]

        with st.chat_message(role):
            if isinstance(content, dict) and "summary_1" in content:
                st.write(_extract_loading_display_text(content))
            else:
                st.write(str(content))

    already_generated = any(
        entry["role"] == "assistant" and isinstance(entry.get("content"), dict)
        for entry in chat_history
    )

    if auto and _has_loading_result(agent) and not agent.finish_auto_task:
        agent.finish_auto()
        st.rerun()

    if analyze_btn or (auto and not already_generated):
        prompt_text = "请帮我解析数据含义"
        st.chat_message("user").write(prompt_text)
        agent.add_memory({"role": "user", "content": prompt_text})

        _request_loading_analysis(agent, df, prompt_text)
        return

    user_input = st.chat_input("请输入需求，例如“帮我分析 x 列”")
    if user_input:
        st.chat_message("user").write(user_input)
        agent.add_memory({"role": "user", "content": user_input})
        _request_loading_analysis(agent, df, user_input)
        return


if __name__ == "__main__":
    agent = st.session_state.data_loading_agent
    planner = st.session_state.planner_agent
    auto = bool(st.session_state.auto_mode and planner.loading_auto)

    if st.session_state.auto_mode == True:
        if planner.loading_auto and _has_loading_result(agent):
            planner.finish_loading_auto()
            st.switch_page(page_file("preprocessing", "preprocessing_render.py"))

    c1,c2 = st.columns(2)
    with c1:
        st.title("数据导入")
    with c2:
        st.write("")  
        st.write("")  
        sac.buttons([
            sac.ButtonsItem(label='Github', icon='github', href='https://github.com/Jiaye-s-Group/AutoSTAT'),
            sac.ButtonsItem(label='Doc', icon=sac.BsIcon(name='bi bi-file-earmark-post-fill', size=16), href='https://autostat.cc/docs/'),
            sac.ButtonsItem(label='Web', icon=sac.BsIcon(name='bi bi-globe', size=16), href='https://autostat.cc/'),
        ], align='end', color='dark', variant='filled', index=None)
    st.markdown("---")

    c = st.columns(3)
    with c[0].expander('数据上传', True):
        loading_data_file(agent)
    with c[0].expander('数据展示', True):
        loading_basic_info(agent)
    with c[1].expander('参考资料pdf/docx上传', True):
        loading_reference_docs(agent)
    with c[2].expander('数据建议', True):
        loading_chat(agent, auto)
