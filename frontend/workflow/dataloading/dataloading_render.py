import html
import os
import json
from typing import Any

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac

from utils.i18n import bt, get_language
from utils.page_paths import page_file
from utils.suggestion_state import (
    add_requirement,
    base_requirements_text,
    get_suggestion_state,
    queue_initial_request,
    queue_revision_request,
    replace_active_suggestion,
    revision_fallback_text,
    take_pending_initial_request,
    take_pending_revision,
    visible_messages,
)
from utils.workflow_state import (
    commit_dataset_fingerprint,
    current_dataset_fingerprint,
    dataframe_fingerprint,
    invalidate_from,
    record_stage_status,
    stable_fingerprint,
)
from workflow.dataloading.dataloading_core import (
    PathFileWrapper,
    build_file_manifest,
    file_manifest_fingerprint,
    load_concat_file,
    process_complex_data,
)


def _render_import_file_list_styles() -> None:
    st.markdown(
        """
        <style>
        .autostat-import-file-list {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            max-width: 100%;
            min-width: 0;
        }
        .autostat-import-file-row {
            align-items: flex-start;
            background: rgba(248, 250, 252, 0.9);
            border: 1px solid rgba(203, 213, 225, 0.85);
            border-radius: 0.55rem;
            box-sizing: border-box;
            display: flex;
            gap: 0.45rem;
            line-height: 1.35;
            max-width: 100%;
            min-width: 0;
            padding: 0.45rem 0.55rem;
        }
        .autostat-import-file-icon {
            flex: 0 0 auto;
            line-height: 1.35;
        }
        .autostat-import-file-name {
            flex: 1 1 auto;
            max-width: 100%;
            min-width: 0;
            overflow-wrap: anywhere;
            white-space: normal;
            word-break: break-word;
        }
        .autostat-import-loaded-title {
            color: #111827;
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_import_file_list(names: list[str], *, icon: str = "📄") -> None:
    escaped_icon = html.escape(icon)
    rows = []
    for name in names:
        escaped_name = html.escape(str(name))
        rows.append(
            f'<div class="autostat-import-file-row" title="{escaped_name}">'
            f'<span class="autostat-import-file-icon">{escaped_icon}</span>'
            f'<span class="autostat-import-file-name">{escaped_name}</span>'
            "</div>"
        )
    if rows:
        st.markdown(
            '<div class="autostat-import-file-list">' + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )


# --- Local workflow ---
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
    invalidate_from(
        st.session_state,
        "loading",
        include_source=True,
        reason="loading summary replaced",
    )
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
    record_stage_status(
        st.session_state,
        "loading",
        "succeeded",
        input_fingerprint=current_dataset_fingerprint(st.session_state),
        output_fingerprint=stable_fingerprint(summary_1, abstract_1),
    )


def call_loading_workflow(
    df: pd.DataFrame,
    user_input: str = "",
    loading_auto: bool = True,
    ref_context: str = "",
):
    """
    Run the local loading workflow.
    返回结构与原版一致：{summary_1, abstract_1}
    """
    from workflows.loading import run_loading_workflow
    from workflows._plugins import df_to_meta

    try:
        meta = df_to_meta(df)
        loading_ref_context = ref_context or _retrieve_loading_reference_context(df, user_input)
        result = run_loading_workflow(
            shape_0=meta["shape_0"],
            shape_1=meta["shape_1"],
            dtype_info_str=meta["dtype_info_str"],
            head_dict_str=meta["head_dict_str"],
            data_profile_str=meta.get("data_profile_str", ""),
            loading_auto=loading_auto,
            user_input=user_input or "",
            add_preference=st.session_state.get("add_preference") or "",
            preference_selected=st.session_state.get("preference_selected") or "",
            ref_context=loading_ref_context,
            language=get_language(),
        )
        normalized = _normalize_loading_workflow_result(result)
        if normalized is None:
            st.error(bt(
                "数据导入工作流返回结构异常，未解析到有效结果。",
                "The data import workflow returned an invalid structure and no valid result could be parsed.",
            ))
            return None
        return normalized
    except Exception as e:
        st.error(bt(f"本地 Loading workflow 执行异常：{e}", f"Local loading workflow error: {e}"))
        return None

def _commit_reference_fingerprint_change(reason: str) -> None:
    fingerprints = st.session_state.get("learned_doc_fingerprints") or {}
    reference_fingerprint = stable_fingerprint(fingerprints)
    if reference_fingerprint == st.session_state.get("reference_fingerprint"):
        return
    invalidate_from(st.session_state, "references", reason=reason)
    st.session_state.reference_fingerprint = reference_fingerprint


def _learned_reference_names() -> list[str]:
    names = st.session_state.get("learned_doc_names") or []
    if isinstance(names, set):
        values = list(names)
    elif isinstance(names, (list, tuple)):
        values = list(names)
    else:
        values = []
    statuses = st.session_state.get("reference_doc_statuses") or {}
    if isinstance(statuses, dict):
        for name, status in statuses.items():
            if isinstance(status, dict) and status.get("status") == "success":
                values.append(str(name))
    return sorted({str(name) for name in values if str(name).strip()})


def _reference_retriever_from_state():
    retriever = st.session_state.get("ref_retriever")
    if retriever is not None:
        return retriever
    wrapper = st.session_state.get("retriever")
    return getattr(wrapper, "_ref_retriever", None)


def _retriever_is_empty(retriever) -> bool:
    if retriever is None:
        return True
    is_empty = getattr(retriever, "is_empty", False)
    if callable(is_empty):
        try:
            return bool(is_empty())
        except Exception:
            return False
    return bool(is_empty)


def _retrieve_loading_reference_context(df: pd.DataFrame, user_input: str = "") -> str:
    learned_names = _learned_reference_names()
    retriever = _reference_retriever_from_state()
    if _retriever_is_empty(retriever):
        if not learned_names:
            return ""
        return bt(
            "【参考资料检索状态】已学习参考资料："
            + "、".join(learned_names)
            + "；但当前参考资料检索索引不可用。本次数据解析不得写“当前没有参考资料”，应写“已学习参考资料但未检索到相关片段”。",
            "[Reference retrieval status] Learned reference materials: "
            + ", ".join(learned_names)
            + "; however, the reference retrieval index is unavailable. Do not state that there are no references; state that learned references exist but no relevant chunks were retrieved.",
        )

    columns = [str(column) for column in getattr(df, "columns", [])]
    query = bt(
        "数据字典 字段说明 变量含义 单位 编码 取值方向 缺失值 列名 "
        + " ".join(columns[:160])
        + " 用户需求 "
        + str(user_input or "")
        + " "
        + str(st.session_state.get("add_preference") or ""),
        "data dictionary field descriptions variable meanings units coding value direction missing values columns "
        + " ".join(columns[:160])
        + " user request "
        + str(user_input or "")
        + " "
        + str(st.session_state.get("add_preference") or ""),
    )
    try:
        results = retriever.retrieve(query, top_k=5, min_score=0.0)
    except TypeError:
        results = retriever.retrieve(query, top_k=5)
    except Exception as exc:
        if not learned_names:
            return ""
        return bt(
            f"【参考资料检索状态】已学习参考资料：{'、'.join(learned_names)}；但本次检索失败：{exc}。请明确写“未检索到相关参考资料”，不要写“当前没有参考资料”。",
            f"[Reference retrieval status] Learned reference materials: {', '.join(learned_names)}; retrieval failed for this run: {exc}. State that no relevant reference material was retrieved; do not state that there are no references.",
        )

    if not results:
        source_names = learned_names or sorted(
            {
                str(chunk.get("source") or "")
                for chunk in getattr(retriever, "chunks", [])
                if str(chunk.get("source") or "").strip()
            }
        )
        if not source_names:
            return ""
        return bt(
            "【参考资料检索状态】已学习参考资料："
            + "、".join(source_names)
            + "；但针对本次数据解析查询未检索到相关片段。请明确写“未检索到相关参考资料”，不要写“当前没有参考资料”。",
            "[Reference retrieval status] Learned reference materials: "
            + ", ".join(source_names)
            + "; however, this data-understanding query retrieved no relevant chunks. State that no relevant reference material was retrieved; do not state that there are no references.",
        )

    try:
        formatted = retriever.format_results(results)
    except Exception:
        formatted = "\n\n".join(str(result.get("text") or "") for result in results)
    source_names = learned_names or sorted(
        {
            str(result.get("source") or "")
            for result in results
            if str(result.get("source") or "").strip()
        }
    )
    prefix = bt(
        "【参考资料检索状态】已学习参考资料："
        + ("、".join(source_names) if source_names else "未记录文件名")
        + "；本次数据解析已召回以下数据字典/字段说明片段，应优先用于字段语义、单位、编码和方向性解释。\n\n",
        "[Reference retrieval status] Learned reference materials: "
        + (", ".join(source_names) if source_names else "file names not recorded")
        + "; this data-understanding run retrieved the following data dictionary / field description chunks. Prioritize them for field meanings, units, coding, and directionality.\n\n",
    )
    return prefix + formatted


def loading_reference_docs(agent):
    """
    专门处理参考资料的上传逻辑
    """
    _render_import_file_list_styles()
    st.info(bt(
        "💡 提示：在此处上传业务背景、算法说明或数据手册，AI 会学习这些内容。",
        "💡 Tip: Upload business context, algorithm notes, or data manuals here. The AI will learn from them.",
    ))
    if not isinstance(st.session_state.get("learned_doc_names"), set):
        st.session_state.learned_doc_names = set(st.session_state.get("learned_doc_names") or [])
    if not isinstance(st.session_state.get("learned_doc_fingerprints"), dict):
        st.session_state.learned_doc_fingerprints = {}
    if not isinstance(st.session_state.get("reference_doc_statuses"), dict):
        st.session_state.reference_doc_statuses = {}
    
    uploaded_docs = st.file_uploader(
        bt("上传参考文档", "Upload Reference Documents"),
        type=['pdf', 'docx', 'txt', 'names'],
        accept_multiple_files=True,
        key="ref_doc_uploader"
    )

    if uploaded_docs:
        doc_manifest = build_file_manifest(uploaded_docs)
        uploaded_by_name = {str(file_obj.name): file_obj for file_obj in uploaded_docs}
        fingerprints_by_name = {
            str(item.get("name") or ""): str(item.get("sha256") or "")
            for item in doc_manifest
        }
        known_hash_owners = {
            digest: name
            for name, digest in st.session_state.learned_doc_fingerprints.items()
            if digest
        }
        files_to_process = []
        for item in doc_manifest:
            name = str(item.get("name") or "")
            digest = str(item.get("sha256") or "")
            current_digest = st.session_state.learned_doc_fingerprints.get(name)
            duplicate_owner = known_hash_owners.get(digest)

            if current_digest == digest:
                st.session_state.reference_doc_statuses[name] = {
                    "name": name,
                    "status": "success",
                    "chunk_count": sum(
                        1
                        for chunk in st.session_state.retriever.ref_chunks
                        if str(chunk.get("source", "")) == name
                    ),
                    "error": "",
                    "sha256": digest,
                }
            elif duplicate_owner and duplicate_owner != name:
                st.session_state.reference_doc_statuses[name] = {
                    "name": name,
                    "status": "duplicate",
                    "chunk_count": 0,
                    "error": bt(
                        f"内容与 {duplicate_owner} 相同，已跳过去重。",
                        f"Same content as {duplicate_owner}; skipped as a duplicate.",
                    ),
                    "sha256": digest,
                }
            else:
                files_to_process.append(uploaded_by_name[name])
                known_hash_owners[digest] = name
                existing_status = st.session_state.reference_doc_statuses.get(name) or {}
                if existing_status.get("sha256") != digest:
                    st.session_state.reference_doc_statuses[name] = {
                        "name": name,
                        "status": "pending",
                        "chunk_count": 0,
                        "error": "",
                        "sha256": digest,
                    }
        
        if files_to_process:
            if st.button(bt("🧠 学习资料", "🧠 Learn Documents"), use_container_width=True):
                with st.spinner(bt("正在解析文档并提取知识点...", "Parsing documents and extracting knowledge...")):
                    results = st.session_state.retriever.add_uploaded_files_detailed(files_to_process)
                    successful_results = []
                    failed_results = []
                    for result in results:
                        name = str(result.get("name") or "")
                        result = dict(result)
                        result["sha256"] = fingerprints_by_name.get(name, "")
                        st.session_state.reference_doc_statuses[name] = result
                        if result.get("status") == "success":
                            successful_results.append(result)
                            st.session_state.learned_doc_names.add(name)
                            st.session_state.learned_doc_fingerprints[name] = fingerprints_by_name[name]
                        else:
                            failed_results.append(result)

                    if successful_results:
                        _commit_reference_fingerprint_change("reference documents changed")

                if successful_results:
                    chunk_count = sum(int(result.get("chunk_count") or 0) for result in successful_results)
                    st.success(bt(
                        f"成功学习 {len(successful_results)} 份文档，提取了 {chunk_count} 条知识片段。",
                        f"Learned {len(successful_results)} document(s) and extracted {chunk_count} knowledge chunk(s).",
                    ))
                for result in failed_results:
                    st.error(f"{result.get('name')}: {result.get('error') or bt('未提取到有效内容', 'No usable content was extracted')}")
        else:
            st.caption(bt("✅ 当前上传的文件已全部在知识库中。", "✅ All uploaded files are already in the knowledge base."))

        current_statuses = [
            st.session_state.reference_doc_statuses.get(str(item.get("name") or ""), {})
            for item in doc_manifest
        ]
        status_labels = {
            "success": bt("成功", "Success"),
            "failed": bt("失败", "Failed"),
            "empty": bt("空文档", "Empty"),
            "duplicate": bt("重复", "Duplicate"),
            "pending": bt("待处理", "Pending"),
        }
        status_rows = [
            {
                bt("文件", "File"): status.get("name", ""),
                bt("状态", "Status"): status_labels.get(status.get("status"), status.get("status", "")),
                bt("知识片段", "Chunks"): int(status.get("chunk_count") or 0),
                bt("说明", "Details"): status.get("error", ""),
            }
            for status in current_statuses
            if status
        ]
        if status_rows:
            st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    if 'learned_doc_names' in st.session_state and st.session_state.learned_doc_names:
        st.markdown(
            f'<div class="autostat-import-loaded-title">'
            f'{html.escape(bt("已加载的外部资料：", "Loaded external references:"))}'
            "</div>",
            unsafe_allow_html=True,
        )
        for name in sorted(st.session_state.learned_doc_names):
            label_col, delete_col = st.columns([4, 1.15], gap="small")
            with label_col:
                _render_import_file_list([name], icon="📄")
            if delete_col.button(
                bt("删除", "Delete"),
                key=f"delete_reference_{stable_fingerprint(name)[:12]}",
                use_container_width=True,
            ):
                removed, error = st.session_state.retriever.remove_document(name)
                if removed:
                    st.session_state.learned_doc_names.discard(name)
                    st.session_state.learned_doc_fingerprints.pop(name, None)
                    st.session_state.reference_doc_statuses.pop(name, None)
                    _commit_reference_fingerprint_change("reference document removed")
                    st.rerun()
                else:
                    st.error(f"{name}: {error}")

        if st.button(bt("重建参考资料索引", "Rebuild Reference Index"), use_container_width=True):
            rebuilt, error = st.session_state.retriever.rebuild_index()
            if rebuilt:
                st.success(bt("参考资料索引已重建。", "Reference index rebuilt."))
            else:
                st.error(bt(f"索引重建失败：{error}", f"Index rebuild failed: {error}"))


def _replace_agent_file_names(agent, names: list[str]) -> None:
    agent.file_names = list(names)


def _commit_loaded_dataset(
    agent,
    *,
    df: pd.DataFrame,
    dfs,
    manifest: list[dict[str, object]],
    source: str,
    combine_mode: str = "vertical",
) -> bool:
    manifest_fingerprint = file_manifest_fingerprint(manifest, combine_mode=combine_mode)
    dataset_fingerprint = stable_fingerprint(
        manifest_fingerprint,
        dataframe_fingerprint(df),
    )
    changed = commit_dataset_fingerprint(st.session_state, dataset_fingerprint)

    agent.add_df(df)
    agent.save_dfs(dfs)
    _replace_agent_file_names(
        agent,
        [str(item.get("source") or item.get("name") or "") for item in manifest],
    )
    st.session_state.data_file_manifest = manifest
    st.session_state.data_file_manifest_fingerprint = manifest_fingerprint
    st.session_state.data_file_snapshot_fingerprint = file_manifest_fingerprint(
        manifest,
        combine_mode="snapshot",
    )
    st.session_state.data_source_kind = source
    st.session_state.data_combine_mode = combine_mode
    return changed


def _clear_loaded_dataset(agent) -> None:
    commit_dataset_fingerprint(st.session_state, "")
    agent.add_df(None)
    agent.save_dfs(None)
    _replace_agent_file_names(agent, [])
    for key in (
        "data_file_manifest",
        "data_file_manifest_fingerprint",
        "data_file_snapshot_fingerprint",
        "data_source_kind",
        "data_combine_mode",
    ):
        st.session_state.pop(key, None)

def loading_data_file(agent):
    """ """
    _render_import_file_list_styles()
    st.info(
        bt(
            "💡 提示：\n"
            "1. 自动使用大模型分析并处理数据\n"
            "2. 支持多种格式的文件类型上传\n",
            "💡 Tip:\n"
            "1. Automatically analyze and process data with the LLM\n"
            "2. Supports multiple file formats\n",
        )
    )

    local_upload_tab = bt("本地上传", "Local Upload")
    path_import_tab = bt("路径导入", "Path Import")
    public_deployment = os.getenv("AUTOSTAT_PUBLIC_DEPLOYMENT", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    selected_index = (
        local_upload_tab
        if public_deployment
        else sac.tabs([
            sac.TabsItem(label=local_upload_tab),
            sac.TabsItem(label=path_import_tab),
        ], color='#5980AE',)
    )

    if selected_index == local_upload_tab:
        uploader_generation = int(st.session_state.get("data_file_uploader_generation", 0))
        uploaded_files = st.file_uploader(
            bt("选择新文件", "Select New Files"),
            type=["csv", "data", "txt", "xlsx", "xls", "mat", "arff", "tsv", "dat", "tst", "names"],
            accept_multiple_files=True,
            help=bt("拖拽或点击上传多个文件", "Drag or click to upload multiple files"),
            key=f"data_file_uploader_{uploader_generation}",
        )

        persisted_manifest = st.session_state.get("data_file_manifest") or []
        persisted_names = [
            str(item.get("source") or item.get("name") or "").strip()
            for item in persisted_manifest
            if isinstance(item, dict)
        ]
        persisted_names = [name for name in persisted_names if name]
        if (
            not uploaded_files
            and persisted_names
            and st.session_state.get("data_source_kind") == "upload"
            and agent.load_df() is not None
        ):
            loaded_col, clear_col = st.columns([4, 1.15], gap="small")
            with loaded_col:
                st.markdown(
                    f'<div class="autostat-import-loaded-title">'
                    f'{html.escape(bt("当前已加载数据文件：", "Currently loaded data files: "))}'
                    "</div>",
                    unsafe_allow_html=True,
                )
                _render_import_file_list(persisted_names, icon="📊")
            with clear_col:
                if st.button(
                    bt("删除", "Remove"),
                    key="clear_loaded_dataset",
                    use_container_width=True,
                ):
                    _clear_loaded_dataset(agent)
                    st.session_state.data_file_uploader_generation = uploader_generation + 1
                    st.rerun()

        if uploaded_files:
            manifest = build_file_manifest(uploaded_files)
            snapshot_fingerprint = file_manifest_fingerprint(manifest, combine_mode="snapshot")
            if snapshot_fingerprint != st.session_state.get("data_file_snapshot_fingerprint"):
                try:
                    with st.spinner(bt("正在处理数据...", "Processing data...")):
                        df, dfs = process_complex_data(uploaded_files, agent)
                    if df is not None:
                        _commit_loaded_dataset(
                            agent,
                            df=df,
                            dfs=dfs,
                            manifest=manifest,
                            source="upload",
                        )
                        st.rerun()
                except Exception as err:
                    st.error(bt(f"导入失败：{err}", f"Import failed: {err}"))
        # ``st.file_uploader`` is reset when this multipage app navigates away
        # from the import page. An empty widget value therefore does not mean
        # that the user requested deletion of the imported dataset.

    elif not public_deployment and selected_index == path_import_tab:
        raw_paths = st.text_area(
            bt("从路径导入数据 (每行一个文件路径)", "Import Data From Paths (one file path per line)"),
            placeholder="C:\\data\\iris.names\nC:\\data\\iris.data",
            height=100
        )

        if st.button(bt("从路径加载文件", "Load Files From Paths"), use_container_width=True):
            if raw_paths:
                path_list = [p.strip().strip("'\"") for p in raw_paths.strip().split('\n') if p.strip()]
                valid_paths = [p for p in path_list if os.path.exists(p)]
                invalid_paths = [p for p in path_list if not os.path.exists(p)]

                if invalid_paths:
                    st.warning(bt("路径不存在，已跳过：\n- ", "Paths not found and skipped:\n- ") + "\n- ".join(invalid_paths))

                if not valid_paths:
                    st.error(bt("未找到任何有效的本地文件路径。", "No valid local file paths were found."))
                else:
                    files_to_process = [PathFileWrapper(p) for p in valid_paths]
                    manifest = build_file_manifest(files_to_process)
                    snapshot_fingerprint = file_manifest_fingerprint(manifest, combine_mode="snapshot")
                    if snapshot_fingerprint == st.session_state.get("data_file_snapshot_fingerprint"):
                        st.info(bt("当前路径文件快照已加载。", "The current path-file snapshot is already loaded."))
                    else:
                        try:
                            with st.spinner(bt("正在处理数据...", "Processing data...")):
                                df, dfs = process_complex_data(files_to_process, agent)
                            if df is not None:
                                _commit_loaded_dataset(
                                    agent,
                                    df=df,
                                    dfs=dfs,
                                    manifest=manifest,
                                    source="path",
                                )
                                st.rerun()
                        except Exception as err:
                            st.error(bt(f"本地文件读取失败：{err}", f"Local file read failed: {err}"))
    
    dfs = agent.load_dfs()
    if dfs is not None and len(dfs) >= 2:
        combined_df, combine_mode = load_concat_file(dfs, agent)
        manifest = st.session_state.get("data_file_manifest") or []
        if combined_df is not None and manifest:
            _commit_loaded_dataset(
                agent,
                df=combined_df,
                dfs=dfs,
                manifest=manifest,
                source=str(st.session_state.get("data_source_kind") or "upload"),
                combine_mode=combine_mode,
            )

def loading_basic_info(agent):
    """ """
    df = agent.load_df()
    if df is not None:
        r, c = df.shape
        missing = int(df.isnull().sum().sum())
        col1, col2, col3 = st.columns(3)
        col1.metric(bt("行数", "Rows"), r)
        col2.metric(bt("列数", "Columns"), c)
        col3.metric(bt("缺失值总数", "Missing Values"), missing)

        dtype_info = pd.DataFrame({
            bt("列名", "Column"): df.columns,
            bt("类型", "Type"): df.dtypes.astype(str),
            bt("非空", "Non-null"): df.count().values,
            bt("缺失%", "Missing %"): (df.isnull().mean() * 100).round(2).values,
        }).reset_index(drop=True)

        dtype_tab = bt("数据类型概览", "Data Type Overview")
        preview_tab = bt("数据预览", "Data Preview")
        selected_index = sac.tabs([
            sac.TabsItem(label=dtype_tab),
            sac.TabsItem(label=preview_tab),
        ],color='#5980AE',)

        if selected_index == dtype_tab:
            st.dataframe(dtype_info, use_container_width=True)
        elif selected_index == preview_tab:
            if st.button(bt("🎲 随机抽样", "🎲 Random Sample")):
                display_df = df.sample(min(10, len(df)))
                st.dataframe(display_df, use_container_width=True)
            else:
                st.dataframe(df.head(10), use_container_width=True)

def _extract_loading_display_text(workflow_result: dict[str, Any]) -> str:
    summary_fields = _extract_summary_1_fields(workflow_result.get("summary_1"))
    desc = summary_fields["desc"]
    if desc:
        return desc

    return bt(
        "工作流已运行，但 summary_1.desc 为空。",
        "The workflow has run, but summary_1.desc is empty.",
    )


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


def _publish_loading_suggestion(agent, state: dict[str, Any]) -> None:
    workflow_result = state.get("pending_payload")
    if not isinstance(workflow_result, dict) or not state.get("active_suggestion"):
        return
    _save_loading_workflow_outputs(agent, workflow_result)
    agent.finish_auto()


def _request_loading_analysis(
    agent,
    df: pd.DataFrame,
    user_input: str,
    *,
    auto: bool,
) -> None:
    state = get_suggestion_state(st.session_state, "loading")
    with st.spinner(bt("正在解析数据，请耐心等待...", "Analyzing the data. Please wait...")):
        workflow_result = call_loading_workflow(
            df,
            user_input=user_input,
            loading_auto=True,
        )

    if not workflow_result:
        return

    state["pending_payload"] = workflow_result
    replace_active_suggestion(state, _extract_loading_display_text(workflow_result))
    _publish_loading_suggestion(agent, state)
    st.rerun()


def _revise_loading_suggestion(agent, revision_instruction: str) -> None:
    from core.report_language import app_language_instruction
    from workflows.loading import revise_loading_result

    state = get_suggestion_state(st.session_state, "loading")
    current_result = state.get("pending_payload")
    if not isinstance(current_result, dict):
        return
    with st.spinner(bt("正在根据修改意见更新数据理解...", "Revising the data interpretation...")):
        revised_result = revise_loading_result(
            current_result=current_result,
            original_requirements=base_requirements_text(state),
            revision_instruction=revision_instruction,
            language_instruction=app_language_instruction(get_language()),
        )
    if state.get("confirmed_version") is not None:
        invalidate_from(
            st.session_state,
            "loading",
            include_source=True,
            reason="confirmed data interpretation is being revised",
        )
    state["pending_payload"] = revised_result
    replace_active_suggestion(
        state,
        _extract_loading_display_text(revised_result),
        revision_instruction=revision_instruction,
    )
    _publish_loading_suggestion(agent, state)
    st.rerun()


def _render_auto_planning_status() -> None:
    if not st.session_state.get("auto_planning"):
        return

    status_text = bt("正在规划自动分析流程，请耐心等待。", "Planning the automated analysis flow. Please wait.")
    st.markdown(
        """
        <style>
        .auto-planning-status {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            margin: 0.1rem 0 0.2rem 0;
            padding: 0.5rem 0.75rem;
            color: #24527a;
            background: #f3f8fc;
            border: 1px solid #d9e8f4;
            border-radius: 8px;
            font-weight: 600;
            line-height: 1.35;
            scroll-margin-top: 1rem;
        }
        .auto-planning-dot {
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 999px;
            background: #5980ae;
            animation: autoPlanningPulse 1.1s ease-in-out infinite;
            flex: 0 0 auto;
        }
        @keyframes autoPlanningPulse {
            0%, 100% { opacity: 0.35; transform: scale(0.85); }
            50% { opacity: 1; transform: scale(1); }
        }
        </style>
        <div id="auto-planning-status" class="auto-planning-status">
            <span class="auto-planning-dot"></span>
        """
        f"            <span>{status_text}</span>\n"
        """
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.components.v1.html(
        """
        <script>
        (() => {
          const parentWindow = window.parent;
          const doc = parentWindow.document;
          const frame = window.frameElement;
          if (frame) {
            const host = frame.closest('[data-testid="stElementContainer"]');
            [frame, host].filter(Boolean).forEach((element) => {
              element.style.setProperty("height", "0", "important");
              element.style.setProperty("min-height", "0", "important");
              element.style.setProperty("max-height", "0", "important");
              element.style.setProperty("margin", "0", "important");
              element.style.setProperty("padding", "0", "important");
              element.style.setProperty("overflow", "hidden", "important");
            });
          }

          let attempts = 0;
          function scrollToStatus() {
            const status = doc.getElementById("auto-planning-status");
            if (status) {
              status.scrollIntoView({ behavior: "auto", block: "center", inline: "nearest" });
              return;
            }

            attempts += 1;
            if (attempts < 20) {
              parentWindow.setTimeout(scrollToStatus, 50);
            }
          }

          parentWindow.requestAnimationFrame(scrollToStatus);
        })();
        </script>
        """,
        height=0,
    )


def loading_chat(agent, auto=False) -> None:
    df = agent.load_df()
    if df is None:
        return

    state = get_suggestion_state(st.session_state, "loading")

    with st.chat_message("assistant"):
        st.write(
            bt(
                "我是 Autostat 数据分析助手，很高兴为您服务；\n\n"
                "请先上传您的数据文件，上传完成后，您可以在下方和我对话，也可以直接点击按钮解析数据含义。",
                "I am the Autostat data analysis assistant. Glad to help.\n\n"
                "Upload your data file first. After the upload finishes, you can chat below or click the button to analyze field meanings.",
            )
        )
        analyze_btn = st.button(
            bt("🔍 生成数据解析", "🔍 Generate Data Interpretation"),
            disabled=bool(state.get("active_suggestion")),
        )

    for entry in visible_messages(state):
        role = entry["role"]
        content = entry["content"]
        with st.chat_message(role):
            st.write(str(content))

    pending_initial_request = take_pending_initial_request(state)
    if pending_initial_request:
        request_text = base_requirements_text(state, pending_initial_request)
        _request_loading_analysis(agent, df, request_text, auto=False)
        return

    pending_revision = take_pending_revision(state)
    if pending_revision:
        if isinstance(state.get("pending_payload"), dict):
            _revise_loading_suggestion(agent, pending_revision)
        else:
            st.warning(bt(
                "上一轮数据解析上下文已失效，正在基于当前数据和这条消息重新生成解析。",
                "The previous data-interpretation context expired. Regenerating from the current data and this message.",
            ))
            request_text = revision_fallback_text(
                state,
                pending_revision,
                default=bt("请帮我解析数据含义", "Please analyze the meaning of this dataset"),
            )
            _request_loading_analysis(agent, df, request_text, auto=False)
        return

    already_generated = bool(state.get("active_suggestion"))

    if auto and _has_loading_result(agent) and not agent.finish_auto_task:
        agent.finish_auto()
        st.rerun()

    if analyze_btn or (auto and not already_generated):
        prompt_text = bt("请帮我解析数据含义", "Please analyze the meaning of this dataset")
        if not state.get("base_requirements"):
            add_requirement(state, prompt_text)
        request_text = base_requirements_text(state, prompt_text)
        _request_loading_analysis(agent, df, request_text, auto=auto)
        return

    user_input = st.chat_input(bt(
        "请输入要求；建议生成后可在这里继续提出修改意见",
        "Enter requirements; after generation, use this box to request revisions",
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
    agent = st.session_state.data_loading_agent
    planner = st.session_state.planner_agent
    auto = bool(st.session_state.auto_mode and planner.loading_auto)

    if st.session_state.auto_mode == True:
        if planner.loading_auto and _has_loading_result(agent):
            next_page = planner.finish_loading_auto()
            if next_page is not None:
                st.switch_page(next_page)
            st.session_state.auto_mode = False
            st.rerun()

    c1,c2 = st.columns(2)
    with c1:
        st.title(bt("数据导入", "Data Import"))
        _render_auto_planning_status()
    with c2:
        st.write("")  
        st.write("")  
        sac.buttons([
            # sac.ButtonsItem(label='Github', icon='github', href='https://github.com/Jiaye-s-Group/AutoSTAT'),
            # sac.ButtonsItem(label='Doc', icon=sac.BsIcon(name='bi bi-file-earmark-post-fill', size=16), href='https://autostat.cc/docs/'),
            sac.ButtonsItem(label='Web', icon=sac.BsIcon(name='bi bi-globe', size=16), href='https://autostat.cc/docs/examples.html'),
        ], align='end', color='dark', variant='filled', index=None)
    st.markdown("---")

    c = st.columns(3)
    with c[0].expander(bt("数据上传", "Data Upload"), True):
        loading_data_file(agent)
    with c[0].expander(bt("数据展示", "Data Display"), True):
        loading_basic_info(agent)
    with c[1].expander(bt("参考资料上传", "Reference Material Upload"), True):
        loading_reference_docs(agent)
    with c[2].expander(bt("数据解析", "Data Interpretation"), True):
        loading_chat(agent, auto)
