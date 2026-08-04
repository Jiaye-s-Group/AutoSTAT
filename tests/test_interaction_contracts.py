from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
if str(FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(FRONTEND_ROOT))

from core.llm_client import (
    DEFAULT_SUGGESTION_MAX_TOKENS,
    LLMClient,
    LLMOutputIncompleteError,
    LLMOutputTruncatedError,
)
from core.modeling_contract import (
    build_analysis_contract,
    contract_as_prompt,
    validate_result_against_contract,
    validate_result_schema,
)
from core.modeling_report_artifacts import build_modeling_report_artifacts
from core.ref_doc_parser import parse_and_chunk_results
from core.prompt_template import render_file
from core.suggestion_revision import normalize_suggestion_output
from utils.workflow_state import (
    commit_dataset_fingerprint,
    dataframe_fingerprint,
    invalidate_from,
    record_stage_status,
)
from utils.suggestion_state import (
    add_requirement,
    add_revision_request,
    begin_code_execution,
    can_auto_repair,
    code_matches_current_suggestion,
    confirm_active_suggestion,
    finish_code_execution,
    get_suggestion_state,
    mark_code_draft,
    queue_code_revision_request,
    queue_initial_request,
    queue_revision_request,
    record_auto_repair,
    record_execution_failure,
    record_successful_code,
    replace_active_suggestion,
    take_pending_code_revision,
    take_pending_initial_request,
    take_pending_revision,
    visible_messages,
)
from workflow.dataloading.dataloading_core import (
    build_file_manifest,
    file_manifest_fingerprint,
    parse_names_file,
)


class NamedBytesIO(io.BytesIO):
    def __init__(self, name: str, value: bytes):
        super().__init__(value)
        self.name = name


class FakeAgent:
    def __init__(self):
        self.df = pd.DataFrame({"x": [1]})
        self.memory = ["old"]
        self.loading_workflow_result = {"old": True}
        self.processed_df = "old"
        self.finish_auto_task = True


class FakeChatCompletions:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.content = content
        self.finish_reason = finish_reason
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            model=kwargs.get("model"),
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish_reason,
                    message=SimpleNamespace(content=self.content),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class SequencedFakeChatCompletions(FakeChatCompletions):
    def __init__(self, responses: list[tuple[str, str]]):
        super().__init__("", finish_reason="stop")
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)
        content, finish_reason = self.responses.pop(0)
        return SimpleNamespace(
            model=kwargs.get("model"),
            choices=[
                SimpleNamespace(
                    finish_reason=finish_reason,
                    message=SimpleNamespace(content=content),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


def _fake_llm_client(completions: FakeChatCompletions) -> LLMClient:
    client = object.__new__(LLMClient)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client.model = "fake-model"
    client.base_url = "https://fake.local/v1"
    return client


def test_llm_complete_contract_detects_truncated_outputs():
    llm = _fake_llm_client(FakeChatCompletions("partial", finish_reason="length"))

    with pytest.raises(LLMOutputTruncatedError):
        llm.chat("system", "user", name="unit.truncated", require_complete=True)


def test_chat_suggestion_requires_and_strips_completion_marker():
    completions = FakeChatCompletions("complete suggestion\nAUTOSTAT_SUGGESTION_COMPLETE")
    llm = _fake_llm_client(completions)

    text = llm.chat_suggestion("system", "user", name="unit.suggestion")

    assert text == "complete suggestion"
    assert "AUTOSTAT_SUGGESTION_COMPLETE" in completions.kwargs["messages"][-1]["content"]
    assert completions.kwargs["max_tokens"] == DEFAULT_SUGGESTION_MAX_TOKENS


def test_chat_suggestion_rejects_missing_completion_marker():
    llm = _fake_llm_client(FakeChatCompletions("complete-looking suggestion"))

    with pytest.raises(LLMOutputIncompleteError):
        llm.chat_suggestion("system", "user", name="unit.suggestion")


def test_chat_suggestion_retries_once_after_an_incomplete_output():
    completions = SequencedFakeChatCompletions(
        [
            ("partial suggestion", "length"),
            ("complete suggestion\nAUTOSTAT_SUGGESTION_COMPLETE", "stop"),
        ]
    )
    llm = _fake_llm_client(completions)

    text = llm.chat_suggestion("system", "user", name="unit.suggestion")

    assert text == "complete suggestion"
    assert len(completions.calls) == 2
    assert "previous attempt was incomplete" in completions.calls[1]["messages"][-1]["content"]


def test_parallel_dag_invalidation_only_invalidates_real_descendants():
    state = {
        "summary_1": {"loading": True},
        "summary_3": {"viz": True},
        "summary_4": {"model": True},
        "report_final_html": "old",
    }
    for stage in ("loading", "preprocessing", "visualization", "modeling", "report"):
        record_stage_status(state, stage, "succeeded", input_fingerprint="dataset-a")

    invalidated = invalidate_from(state, "preprocessing", reason="prep changed")

    assert invalidated == {"visualization", "modeling", "report"}
    assert state["summary_1"] == {"loading": True}
    assert "summary_3" not in state
    assert "summary_4" not in state
    assert "report_final_html" not in state
    assert state["workflow_stage_states"]["loading"]["status"] == "succeeded"
    assert state["workflow_stage_states"]["visualization"]["status"] == "stale"


def test_dataset_change_clears_real_phase_context_keys():
    state = {
        "dataset_fingerprint": "old",
        "_viz_phase1_ctx": {"dataset": "old"},
        "_viz_phase2_inputs": {"data": "old"},
        "_model_phase1_ctx": {"dataset": "old"},
        "_model_phase2_inputs": {"data": "old"},
        "history_train_code_input": "old code",
    }

    commit_dataset_fingerprint(state, "new")

    for key in (
        "_viz_phase1_ctx",
        "_viz_phase2_inputs",
        "_model_phase1_ctx",
        "_model_phase2_inputs",
        "history_train_code_input",
    ):
        assert key not in state


def test_dataset_commit_invalidates_all_semantic_artifacts_but_keeps_dataframe():
    loading_agent = FakeAgent()
    state = {
        "dataset_fingerprint": "old",
        "summary_1": {"old": True},
        "summary_2": {"old": True},
        "data_loading_agent": loading_agent,
    }

    assert commit_dataset_fingerprint(state, "new") is True
    assert state["dataset_fingerprint"] == "new"
    assert "summary_1" not in state
    assert "summary_2" not in state
    assert loading_agent.df.equals(pd.DataFrame({"x": [1]}))
    assert loading_agent.loading_workflow_result is None


def test_file_snapshot_uses_content_not_only_filename():
    first = build_file_manifest([NamedBytesIO("same.csv", b"a,b\n1,2\n")])
    second = build_file_manifest([NamedBytesIO("same.csv", b"a,b\n3,4\n")])

    assert first[0]["name"] == second[0]["name"]
    assert first[0]["sha256"] != second[0]["sha256"]
    assert file_manifest_fingerprint(first) != file_manifest_fingerprint(second)


def test_data_import_does_not_treat_a_missing_uploader_value_as_dataset_deletion():
    source = (FRONTEND_ROOT / "workflow/dataloading/dataloading_render.py").read_text()

    assert "elif st.session_state.get(\"data_source_kind\") == \"upload\"" not in source
    assert "当前已加载数据文件" in source


def test_preference_page_persists_stable_form_values_for_downstream_workflows():
    source = (FRONTEND_ROOT / "workflow/preference/pref_render.py").read_text()

    assert "PREFERENCE_FIELDS" in source
    assert "preference_form_values" in source
    assert 'key=f"preference_{field_id}"' in source
    assert "format_func=lambda value: _option_label(field, str(value))" in source
    assert "stable_fingerprint(modeling_requirements, form_values)" in source
    assert "st.session_state.add_preference = modeling_requirements" in source
    assert "st.session_state.preference_form_values = form_values" in source
    assert "st.session_state.preference_selected = preferences" in source
    assert "st.session_state.add_preference is not None" not in source


def test_preference_save_button_writes_the_clicked_values_to_session_state(monkeypatch):
    import utils.i18n as i18n
    import workflow.preference.pref_render as pref_render

    class FakeSessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    class FakeColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeStreamlit:
        def __init__(self):
            self.session_state = FakeSessionState(
                {
                    "modeling_requirements": "run complete ChIP-seq exploratory statistics",
                    "preference_report_style": "technical",
                    "preference_analysis_type": "academic",
                    "preference_model_pref": "predictive",
                    "preference_missing_pref": "advanced",
                    "preference_lang_style": "academic",
                    "preference_feature_pref": "many_candidates",
                }
            )
            self.success_messages: list[str] = []
            self.rerun_called = False

        def text_area(self, _label, *, key, **_kwargs):
            return self.session_state.get(key, "")

        def radio(self, _label, options, *, key, **_kwargs):
            assert self.session_state[key] in options
            return self.session_state[key]

        def columns(self, count):
            return [FakeColumn() for _ in range(count)]

        def button(self, *_args, **_kwargs):
            return True

        def success(self, message):
            self.success_messages.append(message)

        def rerun(self):
            self.rerun_called = True

    fake_st = FakeStreamlit()
    monkeypatch.setattr(pref_render, "st", fake_st)
    monkeypatch.setattr(i18n, "st", fake_st)

    pref_render.preferences_select()

    assert fake_st.session_state["add_preference"] == "run complete ChIP-seq exploratory statistics"
    assert fake_st.session_state["preference_form_values"] == {
        "report_style": "technical",
        "analysis_type": "academic",
        "model_pref": "predictive",
        "missing_pref": "advanced",
        "lang_style": "academic",
        "feature_pref": "many_candidates",
    }
    assert fake_st.session_state["preference_selected"]["报告风格"] == "深度技术型"
    assert fake_st.session_state["preference_selected"]["分析方向偏好"] == "学术分析"
    assert fake_st.session_state["preference_fingerprint"]
    assert fake_st.rerun_called is True
    assert fake_st.success_messages == ["偏好设置已保存！"]


def test_preference_invalidation_keeps_the_saved_preference_payload():
    state = {
        "add_preference": "run a complete statistical analysis",
        "preference_selected": {"报告风格": "适中平衡"},
        "preference_form_values": {"report_style": "balanced"},
        "preference_fingerprint": "old",
        "summary_1": {"old": True},
    }

    invalidate_from(state, "preferences", reason="analysis preferences changed")

    assert state["add_preference"] == "run a complete statistical analysis"
    assert state["preference_selected"] == {"报告风格": "适中平衡"}
    assert state["preference_form_values"] == {"report_style": "balanced"}
    assert state["preference_fingerprint"] == "old"
    assert "summary_1" not in state


def test_suggestion_revisions_do_not_require_confirmation_buttons():
    stage_files = [
        "workflow/dataloading/dataloading_render.py",
        "workflow/preprocessing/preprocessing_render.py",
        "workflow/visualization/viz_render.py",
        "workflow/modeling/modeling_render.py",
    ]
    combined = "\n".join((FRONTEND_ROOT / path).read_text() for path in stage_files)
    forbidden = (
        "确认数据理解",
        "确认建议并执行预处理",
        "确认建议并生成图表",
        "确认建议并执行建模",
        "Confirm Data Interpretation",
        "Confirm and Run Preprocessing",
        "Confirm and Generate Charts",
        "Confirm and Run Modeling",
    )

    assert all(label not in combined for label in forbidden)


def test_manual_execution_failure_preserves_the_last_successful_result_reference():
    state = get_suggestion_state({}, "visualization")
    record_successful_code(state, "fig_dict = {'old': fig}")
    previous_fingerprint = state["executed_code_fingerprint"]

    run_id = begin_code_execution(state, "fig_dict = broken")
    finish_code_execution(state, run_id, success=False)
    record_execution_failure(state, "NameError: broken")

    assert state["executed_code_fingerprint"] == previous_fingerprint
    assert state["last_execution_error"] == "NameError: broken"
    assert can_auto_repair(state)

    record_auto_repair(state, "fig_dict = {'fixed': fig}")
    assert state["executed_code_fingerprint"] == previous_fingerprint
    assert state["code_status"] == "ready"
    assert state["auto_repair_attempts"] == 1


def test_all_executable_suggestion_prompts_forbid_waiting_for_user_selection():
    prompt_paths = (
        "prompts/preprocessing/get_preprocessing_suggestions2_llm_sys.txt",
        "prompts/visualizing/sec3_get_visual_recommendation_llm_sys.txt",
        "prompts/modeling/sec4_get_model_suggestion_llm_sys.txt",
        "prompts/shared/revise_suggestion_llm_sys.txt",
    )

    for path in prompt_paths:
        source = (PROJECT_ROOT / path).read_text()
        assert "完整、可直接执行的建议" in source
        assert "不得向用户提问" in source


def test_interactive_invitation_is_removed_only_from_suggestion_tail():
    suggestion = """**推荐方案**
使用逻辑回归和支持向量机作为候选，并按宏平均 F1 比较。

**后续工作**：确定模型后，请告知我您的选择，我可以为您生成代码。

请问您希望从哪一个模型开始？"""

    assert normalize_suggestion_output(suggestion) == (
        "**推荐方案**\n使用逻辑回归和支持向量机作为候选，并按宏平均 F1 比较。"
    )


def test_noninteractive_tail_cleanup_preserves_preceding_complete_sentence():
    suggestion = (
        "两个方案均进入后续执行并分别报告结果。"
        "若您后续有更明确的目标，可随时补充，我将调整方案。"
    )

    assert normalize_suggestion_output(suggestion) == "两个方案均进入后续执行并分别报告结果。"


def test_modeling_prompt_matches_direct_noninteractive_stage_style():
    system_prompt = (PROJECT_ROOT / "prompts/modeling/sec4_get_model_suggestion_llm_sys.txt").read_text()
    user_prompt = (PROJECT_ROOT / "prompts/modeling/sec4_get_model_suggestion_llm_user.txt").read_text()

    assert "不得使用“我/我们/您/你”等对话人称" in system_prompt
    assert "只推荐当前字段结构、样本量和数据粒度确实支持的方法" in system_prompt
    assert "不得强行横向排名" in system_prompt
    assert "**建模任务**" in user_prompt
    assert "**统一执行与评估规则**" in user_prompt
    assert "不邀请补充信息" in user_prompt

    revision_prompt = (PROJECT_ROOT / "prompts/shared/revise_suggestion_llm_sys.txt").read_text()
    assert "不得使用“我/我们/您/你”等对话人称" in revision_prompt


def test_code_buttons_use_phase2_context_instead_of_legacy_agent_templates():
    source_contracts = {
        "workflow/preprocessing/preprocessing_core.py": "_prep_phase2_requested",
        "workflow/visualization/viz_coding.py": "_viz_phase2_requested",
        "workflow/modeling/model_training.py": "_model_phase2_requested",
    }

    for path, request_key in source_contracts.items():
        source = (FRONTEND_ROOT / path).read_text()
        assert request_key in source
        assert "agent.code_generation(" not in source


def test_code_generation_and_repair_controls_clear_before_long_running_requests():
    code_generation_sources = (
        "workflow/preprocessing/preprocessing_core.py",
        "workflow/visualization/viz_coding.py",
        "workflow/modeling/model_training.py",
    )
    repair_sources = (
        "workflow/preprocessing/preprocessing_core.py",
        "workflow/visualization/viz_coding.py",
        "workflow/modeling/modeling_render.py",
    )

    for path in code_generation_sources:
        source = (FRONTEND_ROOT / path).read_text()
        assert "control_slot = st.empty()" in source
        assert "control_slot.empty()" in source

    for path in repair_sources:
        source = (FRONTEND_ROOT / path).read_text()
        assert "repair_slot = st.empty()" in source
        assert "repair_slot.empty()" in source


def test_resizable_cards_observe_the_parent_document_with_the_parent_constructor():
    source = (FRONTEND_ROOT / "utils/resizable_cards.py").read_text()

    assert "new parentWindow.MutationObserver" in source
    assert "if (doc.body && parentWindow.MutationObserver)" in source


def test_resizable_expander_columns_scroll_independently():
    source = (FRONTEND_ROOT / "utils/resizable_cards.py").read_text()

    assert "autostat-column-scroll-body" in source
    assert "overflow-y: auto" in source
    assert "overscroll-behavior-y: contain" in source
    assert "max-height: var(--autostat-column-scroll-max-height" in source
    assert "--autostat-column-scroll-max-height" in source
    assert "function updateColumnScrollHeights()" in source
    assert "function containColumnWheel(event)" in source
    assert "event.stopPropagation()" in source


def test_import_loaded_file_and_reference_labels_share_the_same_style():
    source = (FRONTEND_ROOT / "workflow/dataloading/dataloading_render.py").read_text()

    assert "autostat-import-loaded-title" in source
    assert "当前已加载数据文件" in source
    assert "已加载的外部资料" in source
    assert 'st.markdown(bt("**已加载的外部资料：**"' not in source
    assert "color: #111827" in source


def test_report_outline_and_body_inputs_share_current_stage_snapshot():
    source = (FRONTEND_ROOT / "workflow/report/report_render.py").read_text()
    outline_block = source[
        source.index("def _build_report_inputs"):
        source.index("def _build_word_report_inputs")
    ]
    word_block = source[
        source.index("def _build_word_report_inputs"):
        source.index("def _report_repo_root")
    ]

    assert "_current_report_stage_snapshot(load_agent, allow_report_cache=False)" in outline_block
    assert "_current_report_stage_snapshot(allow_report_cache=True)" in word_block
    assert "stage_snapshot[\"load_abstract\"]" in word_block
    assert "stage_snapshot[\"preference_selected\"]" in word_block
    assert "st.session_state.get(\"report_load_abstract\")" not in word_block
    assert "st.session_state.get(\"report_preference_selected\")" not in word_block


def test_report_worker_uses_the_resolved_current_preference_selection():
    source = (FRONTEND_ROOT / "workflow/report/report_render.py").read_text()
    worker_block = source[
        source.index("def _build_report_worker_payload"):
        source.index("def _cleanup_report_job_files")
    ]

    assert 'inputs["preference_select"] = (' in worker_block
    assert 'inputs.get("preference_selected")' in worker_block
    assert 'inputs.setdefault("preference_select"' not in worker_block


def test_stage_chat_inputs_show_user_message_before_first_generation():
    expectations = {
        "workflow/dataloading/dataloading_render.py": "_request_loading_analysis(agent, df, request_text, auto=False)",
        "workflow/preprocessing/preprocessing_render.py": "_request_prep_recommendation(agent, df, request_text, auto=False)",
        "workflow/visualization/viz_render.py": "_request_visualization_recommendation(agent, source_data, request_text, auto=False)",
        "workflow/modeling/modeling_render.py": "_request_modeling_recommendation(",
    }

    for path, expected_call in expectations.items():
        source = (FRONTEND_ROOT / path).read_text()
        pending_index = source.index("pending_initial_request = take_pending_initial_request(state)")
        pending_block = source[pending_index:pending_index + 1200]
        assert "request_text = base_requirements_text(state, pending_initial_request)" in pending_block
        assert expected_call in pending_block

        chat_input_index = source.index("user_input = st.chat_input")
        user_input_index = source.index("if user_input:", chat_input_index)
        user_input_block = source[user_input_index:user_input_index + 1400]
        assert "queue_initial_request(state, user_input)" in user_input_block
        assert "st.rerun()" in user_input_block
        assert expected_call not in user_input_block


def test_stage_chat_revision_falls_back_when_hidden_context_expired():
    expectations = {
        "workflow/dataloading/dataloading_render.py": "上一轮数据解析上下文已失效",
        "workflow/preprocessing/preprocessing_render.py": "上一轮预处理建议上下文已失效",
        "workflow/visualization/viz_render.py": "上一轮可视化建议上下文已失效",
        "workflow/modeling/modeling_render.py": "上一轮建模建议上下文已失效",
    }

    helper_source = (FRONTEND_ROOT / "utils/suggestion_state.py").read_text()
    assert "def revision_fallback_text(" in helper_source
    assert "Currently displayed suggestion" in helper_source
    assert "Latest user revision" in helper_source

    for path, warning_text in expectations.items():
        source = (FRONTEND_ROOT / path).read_text()
        pending_block = source[
            source.index("pending_revision = take_pending_revision(state)"):
            source.index("already_generated = bool", source.index("pending_revision = take_pending_revision(state)"))
        ]
        assert "revision_fallback_text(" in pending_block
        assert warning_text in pending_block


def test_automatic_mode_does_not_force_a_navigation_back_to_the_import_page():
    source = (FRONTEND_ROOT / "app.py").read_text()

    start_auto_block = source[source.index("# 自动模式逻辑"):source.index("# 检查logo目录是否存在")]
    assert "st.switch_page(page_file(\"dataloading\", \"dataloading_render.py\"))" not in start_auto_block
    assert "if _queue_auto_mode_start():\n                    st.rerun()" in start_auto_block
    pending_index = source.rindex("auto_planning_pending")
    page_run_index = source.rindex("    pg.run()")
    assert pending_index < page_run_index


def test_auto_mode_control_switches_to_stop_while_planning():
    source = (FRONTEND_ROOT / "app.py").read_text()

    assert "auto_control_active = bool(" in source
    assert "st.session_state.auto_mode\n            or st.session_state.auto_planning" in source
    assert "or st.session_state.auto_planning_pending" in source
    assert "if not auto_control_active:" in source


def test_auto_planning_status_is_rendered_before_the_blocking_workflow():
    source = (FRONTEND_ROOT / "app.py").read_text()
    pending_block = source[
        source.rindex('if st.session_state.get("auto_planning_pending"):'):
        source.rindex("    pg.run()")
    ]

    assert "正在规划自动分析流程，请耐心等待。" in pending_block
    assert pending_block.index("with st.spinner(") < pending_block.index("_start_auto_mode()")


def test_names_file_is_actually_parsed():
    names = NamedBytesIO(
        "sample.names",
        b"yes,no.\nage: continuous.\nsex: male,female.\n",
    )
    assert parse_names_file(names, 3) == ["age", "sex", "target"]


def test_dataframe_fingerprint_changes_with_values():
    assert dataframe_fingerprint(pd.DataFrame({"x": [1]})) != dataframe_fingerprint(
        pd.DataFrame({"x": [2]})
    )


def test_automatic_preprocessing_payload_is_restored_as_dataframe():
    from workflow.preprocessing.preprocessing_render import _coerce_processed_dataframe

    restored = _coerce_processed_dataframe('[{"x":1,"group":"a"},{"x":2,"group":"b"}]')

    assert isinstance(restored, pd.DataFrame)
    assert restored.to_dict(orient="records") == [
        {"x": 1, "group": "a"},
        {"x": 2, "group": "b"},
    ]


def test_invalid_automatic_preprocessing_payload_is_rejected():
    from workflow.preprocessing.preprocessing_render import _coerce_processed_dataframe

    assert _coerce_processed_dataframe("not-json") is None


def test_automatic_preprocessing_split_payload_is_restored_as_dataframe():
    from workflow.preprocessing.preprocessing_render import _coerce_processed_dataframe

    restored = _coerce_processed_dataframe(
        '{"columns":["x","group"],"index":[10,11],"data":[[1,"a"],[2,"b"]]}'
    )

    assert restored.index.tolist() == [10, 11]
    assert restored.to_dict(orient="records") == [
        {"x": 1, "group": "a"},
        {"x": 2, "group": "b"},
    ]


@pytest.mark.parametrize(
    ("value", "total", "expected"),
    [(None, 5, 1), ("bad", 5, 1), (0, 5, 1), (9, 5, 5), (3, 5, 3)],
)
def test_visualization_page_selection_is_clamped(value, total, expected):
    from workflow.visualization.viz_render import _safe_visualization_page

    assert _safe_visualization_page(value, total) == expected


def test_visualization_download_cache_lookup_never_starts_image_export(monkeypatch):
    import plotly.graph_objects as go
    import workflow.visualization.viz_render as viz_render

    monkeypatch.setattr(viz_render.st, "session_state", {})
    figure = go.Figure(go.Bar(x=["a", "b"], y=[1, 2]))

    assert viz_render._cached_figure_download_bytes(figure) is None

    cache_key = viz_render._figure_download_cache_key(figure)
    viz_render.st.session_state["viz_download_image_cache"] = {cache_key: b"image"}
    assert viz_render._cached_figure_download_bytes(figure) == b"image"


def test_preprocessing_failure_does_not_return_raw_data(monkeypatch):
    import workflows.preprocessing as prep

    monkeypatch.setattr(prep, "chat_suggestion", lambda *args, **kwargs: "suggestion")
    monkeypatch.setattr(prep, "chat", lambda *args, **kwargs: "suggestion")
    monkeypatch.setattr(prep, "chat_code", lambda *args, **kwargs: "process_df = df.copy()")
    monkeypatch.setattr(prep, "retrieve", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        prep,
        "code_runner",
        lambda **kwargs: {
            "is_success": False,
            "error": "forced failure",
            "processed_df": "",
            "processed_df_head": "",
        },
    )

    result = prep.run_preprocessing_workflow(
        df='[{"x": 1}]',
        shape_0=1,
        shape_1=1,
        dtype_info_str='{"x":"int64"}',
        head_dict_str='[{"x":1}]',
        prep_auto=True,
    )

    assert result["_status"] == "failed"
    assert result["_code_success"] is False
    assert "processed_df" not in result["summary_2"]
    assert result["summary_2"]["status"] == "failed"


def test_loading_metadata_distinguishes_full_dataframe_from_preview_rows():
    from workflows._plugins import df_to_meta

    frame = pd.DataFrame(
        {
            "signal": list(range(10)),
            "group": ["A", "B", "A", "C", "A", "B", "C", "C", "A", None],
        }
    )

    meta = df_to_meta(frame)
    profile = json.loads(meta["data_profile_str"])

    assert profile["full_dataset_loaded"] is True
    assert profile["preview_rows_are_sample_only"] is True
    assert profile["row_count"] == 10
    assert len(json.loads(meta["head_dict_str"])) == 5
    assert profile["columns_with_missing"]["group"] == 1
    assert profile["column_profiles"]["signal"]["numeric_summary"]["max"] == 9


def test_loading_prompts_use_full_profile_and_do_not_treat_preview_as_complete_data(monkeypatch):
    import workflows.loading as loading

    captured_prompts: dict[str, str] = {}

    def fake_chat(system_prompt, user_prompt, **kwargs):
        captured_prompts[str(kwargs.get("name") or len(captured_prompts))] = (
            system_prompt + "\n" + user_prompt
        )
        return "loading summary"

    monkeypatch.setattr(loading, "chat", fake_chat)
    monkeypatch.setattr(
        loading,
        "summary1_composer",
        lambda **kwargs: {
            "summary_1": {
                "title": "",
                "desc": kwargs["desc"],
                "df": kwargs["head_dict_str"],
            }
        },
    )

    profile_text = json.dumps(
        {
            "full_dataset_loaded": True,
            "preview_rows_are_sample_only": True,
            "row_count": 100,
            "column_count": 2,
            "total_missing_values": 3,
        },
        ensure_ascii=False,
    )
    result = loading.run_loading_workflow(
        shape_0=100,
        shape_1=2,
        dtype_info_str='{"signal":"int64","group":"object"}',
        head_dict_str='[{"signal":0,"group":"A"}]',
        data_profile_str=profile_text,
        ref_context="【参考资料检索状态】已学习参考资料：dictionary.txt；本次数据解析已召回字段说明。",
    )

    combined = "\n".join(captured_prompts.values())
    assert "完整数据统计摘要" in combined
    assert "前 5 行样本（仅用于展示取值样例，不代表数据只包含这些行）" in combined
    assert "禁止写“现有输入仅提供前五行”" in combined
    assert "不得写“当前没有参考资料”" in combined
    assert result["_data_profile_str"] == profile_text
    assert "dictionary.txt" in result["_reference_context"]


def test_cpp_inference_contract_and_validation():
    target = "stand_fullscale_iq_7years"
    contract = build_analysis_contract(
        target=target,
        columns=[target, "birth_weight_grams", "age_mom"],
        user_input=(
            "关联性分析而非因果推断；三个嵌套线性回归，共同 complete-case 样本，"
            "HC3 稳健标准误，不做训练测试划分，不做逐步变量选择。"
        ),
    )

    assert contract["task_type"] == "association_inference"
    assert contract["outcome"] == target
    assert contract["split_strategy"] == "none"
    assert contract["covariance"] == "HC3"
    assert contract["sample_rule"] == "common_complete_case"

    result = {
        "analysis_manifest": {
            "task_type": "association_inference",
            "outcome": target,
            "sample_rule": "common_complete_case",
            "split_strategy": "none",
            "covariance": "HC3",
            "model_structure": "nested",
        },
        "models": [
            {
                "name": "Model 1",
                "n_obs": 1012,
                "metrics": {"r_squared": 0.31, "adjusted_r_squared": 0.3},
                "coefficients": [
                    {
                        "term": "birth_weight_grams",
                        "estimate": 0.01,
                        "std_error": 0.002,
                        "ci_lower": 0.006,
                        "ci_upper": 0.014,
                        "p_value": 0.001,
                    }
                ],
            }
        ],
    }
    assert validate_result_against_contract(
        code="fit = model.fit(cov_type='HC3')",
        result_json=result,
        contract=contract,
    ) == []
    assert validate_result_schema(result_json=result, contract=contract) == []

    violations = validate_result_against_contract(
        code="X_train, X_test = train_test_split(X, y)",
        result_json=result,
        contract=contract,
    )
    assert any("train_test_split" in item for item in violations)
    assert any("HC3" in item for item in violations)


def test_inference_result_schema_rejects_empty_or_incomplete_models():
    contract = build_analysis_contract(
        target="outcome",
        columns=["outcome", "x"],
        user_input="Association inference with HC3 robust standard errors.",
    )
    assert validate_result_schema(
        result_json={"models": []},
        contract=contract,
    ) == ["result_dict.models must be a non-empty list of executed analyses."]

    issues = validate_result_schema(
        result_json={
            "models": [
                {
                    "name": "Model 1",
                    "n_obs": 100,
                    "metrics": {"r_squared": 0.2},
                    "coefficients": [{"term": "x", "estimate": 1.0}],
                }
            ]
        },
        contract=contract,
    )
    assert any("std_error" in issue for issue in issues)
    assert any("ci_lower" in issue for issue in issues)
    assert any("p_value" in issue for issue in issues)


def test_candidate_result_contract_requires_candidate_outputs_only_when_requested():
    ordinary_contract = build_analysis_contract(
        target="outcome",
        columns=["outcome", "x"],
        user_input="Association inference with HC3 robust standard errors.",
    )
    assert ordinary_contract["primary_result_type"] == "models"

    candidate_contract = build_analysis_contract(
        target="chip_count",
        columns=["chip_count", "input_count", "gc_content"],
        user_input="主结果只生成候选区段表，列出 top candidate segments。",
        task_type="association_inference",
    )
    assert candidate_contract["primary_result_type"] == "candidate_results"

    base_result = {
        "analysis_manifest": {
            "task_type": "association_inference",
            "outcome": "chip_count",
            "sample_rule": "model_specific",
            "split_strategy": "none",
            "covariance": "model_default",
            "model_structure": "as_requested",
        },
        "models": [
            {
                "name": "Poisson",
                "n_obs": 100,
                "metrics": {"aic": 123.4},
                "coefficients": [
                    {
                        "term": "input_count",
                        "estimate": 0.2,
                        "std_error": 0.05,
                        "ci_lower": 0.1,
                        "ci_upper": 0.3,
                        "p_value": 0.01,
                    }
                ],
            }
        ],
    }
    missing_issues = validate_result_schema(result_json=base_result, contract=candidate_contract)
    assert any("candidate" in issue for issue in missing_issues)

    with_candidates = dict(base_result)
    with_candidates["candidate_segments"] = [{"chromosome": "chr1", "start_bp": 100, "end_bp": 199}]
    assert not any(
        "candidate" in issue
        for issue in validate_result_schema(result_json=with_candidates, contract=candidate_contract)
    )


def test_modeling_report_artifacts_preserve_coefficients_and_candidates_separately():
    artifacts = build_modeling_report_artifacts(
        {
            "analysis_manifest": {"task_type": "association_inference"},
            "models": [
                {
                    "name": "NB2",
                    "n_obs": 100,
                    "metrics": {"aic": 10, "dispersion": 1.2},
                    "coefficients": [{"term": "input_count", "estimate": 0.4, "p_value": 0.01}],
                }
            ],
            "candidate_windows": [{"bin_id": "chr1:1-50", "score": 9.0}],
            "candidate_segments": [{"chromosome": "chr1", "start_bp": 1, "end_bp": 100}],
        },
        target="chip_count",
    )

    assert artifacts["primary_outputs"]["has_models"] is True
    assert artifacts["primary_outputs"]["has_coefficients"] is True
    assert artifacts["primary_outputs"]["has_candidate_windows"] is True
    assert artifacts["primary_outputs"]["has_candidate_segments"] is True
    assert artifacts["coefficients"]["rows"][0]["term"] == "input_count"
    assert artifacts["candidate_windows"]["rows"][0]["bin_id"] == "chr1:1-50"


def test_single_model_table_uses_summary_not_comparison_title():
    from core.modeling_table_utils import build_model_comparison_table_bundle

    zh_bundle = build_model_comparison_table_bundle(
        {
            "models": [
                {"name": "Poisson", "n_obs": 100, "metrics": {"aic": 123.4}},
            ]
        },
        target="chip_count",
        language="zh",
    )
    en_bundle = build_model_comparison_table_bundle(
        {
            "models": [
                {"name": "Poisson", "n_obs": 100, "metrics": {"aic": 123.4}},
            ]
        },
        target="chip_count",
        language="en",
    )

    assert "比较" not in zh_bundle["title"]
    assert "摘要" in zh_bundle["title"]
    assert "Comparison" not in en_bundle["title"]
    assert "Summary" in en_bundle["title"]


def test_supervised_contract_accepts_explicit_dataset_outcome_from_requirements():
    contract = build_analysis_contract(
        target="",
        columns=["stand_fullscale_iq_7years", "birth_weight_grams", "age_mom"],
        user_input="以 stand_fullscale_iq_7years 作为唯一主要结局，进行关联性分析。",
    )
    assert contract["outcome"] == "stand_fullscale_iq_7years"
    assert contract["valid"] is True
    assert contract["issues"] == []


def test_supervised_contract_still_rejects_an_ambiguous_missing_outcome():
    contract = build_analysis_contract(
        target="",
        columns=["outcome_a", "outcome_b", "feature"],
        user_input="请进行关联性分析，但尚未指定结局变量。",
    )
    assert contract["outcome"] == ""
    assert contract["valid"] is False
    assert "chosen from the dataset columns" in contract["issues"][0]


def test_contract_chooses_the_column_nearest_the_outcome_marker():
    contract = build_analysis_contract(
        target="",
        columns=["SepalLengthCm", "Species_encoded"],
        user_input=(
            "使用 Species_encoded 作为唯一目标列进行分类预测，"
            "SepalLengthCm 仅作为输入特征。"
        ),
    )

    assert contract["valid"] is True
    assert contract["outcome"] == "Species_encoded"


def test_contract_does_not_match_a_column_name_inside_another_word():
    contract = build_analysis_contract(
        target="",
        columns=["age", "feature"],
        user_input="Build an average prediction model without an identified outcome.",
    )

    assert contract["valid"] is False
    assert contract["outcome"] == ""


def test_modeling_phase1_can_resolve_an_explicit_iris_outcome(monkeypatch):
    import workflows.modeling as modeling

    responses = iter([
        "将 Species_encoded 明确作为目标列，使用四个测量特征进行分类预测。",
        "执行以 Species_encoded 为目标的监督分类并使用四个测量特征。",
    ])
    monkeypatch.setattr(modeling, "chat_suggestion", lambda *args, **kwargs: next(responses))

    result = modeling.run_modeling_phase1(
        data='[{"SepalLengthCm":5.1,"Species_encoded":0}]',
        df_head='{"SepalLengthCm":[5.1],"Species_encoded":[0]}',
        columns=[
            "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm",
            "Species_encoded",
        ],
        target="",
        user_input="请帮我获取建模建议",
    )

    assert result["analysis_contract"]["valid"] is True
    assert result["analysis_contract"]["task_type"] == "prediction"
    assert result["analysis_contract"]["outcome"] == "Species_encoded"


def test_promoted_modeling_outcome_is_written_to_phase2_context():
    from workflow.modeling.modeling_render import _promote_contract_outcome

    inputs = {"columns": ["x", "Species_encoded"], "target": ""}
    ctx = {"target": ""}
    contract = {
        "valid": True,
        "outcome": "Species_encoded",
        "task_type": "prediction",
    }

    assert _promote_contract_outcome(inputs, ctx, contract) == "Species_encoded"
    assert inputs["target"] == "Species_encoded"
    assert ctx["target"] == "Species_encoded"


def test_explicit_unsupervised_contract_does_not_require_outcome():
    contract = build_analysis_contract(
        target="",
        columns=["x", "y"],
        user_input="探索数据结构",
        task_type="unsupervised",
    )
    assert contract["task_type"] == "unsupervised"
    assert contract["outcome"] == ""
    assert contract["valid"] is True


def test_explicit_supervised_contract_accepts_selected_dataset_column():
    contract = build_analysis_contract(
        target="outcome",
        columns=["outcome", "x"],
        user_input="使用随机森林",
        task_type="prediction",
    )
    assert contract["task_type"] == "prediction"
    assert contract["outcome"] == "outcome"
    assert contract["valid"] is True


def test_modeling_prompt_compaction_keeps_full_contract_columns_and_target(monkeypatch):
    import workflows.modeling as modeling

    monkeypatch.setattr(modeling, "MODELING_PROMPT_MAX_COLUMNS", 5)
    monkeypatch.setattr(modeling, "MODELING_PROMPT_DATA_MAX_CHARS", 80)
    monkeypatch.setattr(modeling, "MODELING_PROMPT_HEAD_MAX_CHARS", 100)
    columns = [f"feature_{index}" for index in range(10)] + ["outcome"]
    head = json.dumps({column: "x" * 50 for column in columns})

    ctx = modeling._build_modeling_ctx(
        data="d" * 500,
        df_head=head,
        columns=columns,
        target="outcome",
        user_input="Run a prediction analysis.",
    )

    assert ctx["_all_columns"] == columns
    assert ctx["columns"][0] == "outcome"
    assert len(ctx["columns"]) == 6
    assert ctx["columns"][-1].startswith("...[")
    assert ctx["analysis_contract"]["outcome"] == "outcome"
    assert "truncated for modeling prompt" in ctx["data"]
    assert len(ctx["df_head"]) < len(head)


def test_modeling_runtime_profile_reports_removed_sklearn_apis(monkeypatch):
    import core.code_runtime_profile as runtime_profile

    monkeypatch.setattr(runtime_profile, "_installed_version", lambda _name: "1.9.0")
    profile = json.loads(
        runtime_profile.build_code_runtime_constraints(
            pd.DataFrame({"x": range(10), "target": [0, 1] * 5}),
            target="target",
            include_modeling_library_compatibility=True,
        )
    )

    rules = profile["libraries"]["scikit_learn"]["compatibility_rules"]
    assert profile["libraries"]["scikit_learn"]["version"] == "1.9.0"
    assert any("multi_class" in rule for rule in rules)
    assert any("sparse_output" in rule for rule in rules)
    assert profile["parameter_bounds"]["clustering"]["silhouette_cluster_range"] == {
        "min": 2,
        "max": 9,
    }


def test_modeling_runtime_profile_sets_large_data_execution_budget():
    import core.code_runtime_profile as runtime_profile

    profile = json.loads(
        runtime_profile.build_code_runtime_constraints(
            [{"x": 1}] * (runtime_profile.LARGE_DATASET_ROW_THRESHOLD + 1),
        )
    )

    budget = profile["execution_budget"]
    assert budget["is_large_dataset"] is True
    assert budget["diagnostic_sample_max_rows"] > 0
    assert any("leave-one-out" in rule for rule in budget["rules"])


def test_modeling_runtime_compatibility_rejects_removed_sklearn_keywords():
    from core.modeling_runtime_compat import validate_modeling_runtime_compatibility

    errors = validate_modeling_runtime_compatibility(
        """
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
classifier = LogisticRegression(multi_class='auto')
encoder = OneHotEncoder(sparse=False)
clusters = KMeans(algorithm='full')
""",
        sklearn_version="1.9.0",
    )

    assert any("multi_class" in error for error in errors)
    assert any("sparse_output" in error for error in errors)
    assert any("algorithm" in error for error in errors)


def test_modeling_runtime_compatibility_rejects_unestimated_glm_negative_binomial():
    from core.modeling_runtime_compat import validate_modeling_runtime_compatibility

    errors = validate_modeling_runtime_compatibility(
        "from statsmodels.genmod.families import NegativeBinomial\nfamily = NegativeBinomial()"
    )
    assert any("explicit alpha" in error for error in errors)

    allowed_errors = validate_modeling_runtime_compatibility(
        "import statsmodels.api as sm\nmodel = sm.NegativeBinomial(y, X)"
    )
    assert not any("NegativeBinomial" in error for error in allowed_errors)


def test_modeling_runtime_compatibility_rejects_full_sample_leave_one_out_diagnostics():
    from core.code_runtime_profile import LARGE_DATASET_ROW_THRESHOLD
    from core.modeling_runtime_compat import validate_modeling_runtime_compatibility

    full_sample_code = """
work_df = df.copy()
y = work_df['y']
X = work_df[['x']]
model = sm.OLS(y, X).fit()
influence = model.get_influence()
external_residual = influence.resid_studentized_external
"""
    errors = validate_modeling_runtime_compatibility(
        full_sample_code,
        n_rows=LARGE_DATASET_ROW_THRESHOLD + 1,
    )
    assert any("leave-one-out" in error for error in errors)

    bounded_diagnostic_code = """
diagnostic_df = df.sample(n=min(len(df), 1000), random_state=7)
y = diagnostic_df['y']
X = diagnostic_df[['x']]
model = sm.OLS(y, X).fit()
influence = model.get_influence()
external_residual = influence.resid_studentized_external
"""
    bounded_errors = validate_modeling_runtime_compatibility(
        bounded_diagnostic_code,
        n_rows=LARGE_DATASET_ROW_THRESHOLD + 1,
    )
    assert not any("leave-one-out" in error for error in bounded_errors)

    formula_code = """
model = smf.ols('y ~ x', data=df).fit()
influence = model.get_influence()
external_residual = influence.resid_studentized_external
"""
    formula_errors = validate_modeling_runtime_compatibility(
        formula_code,
        n_rows=LARGE_DATASET_ROW_THRESHOLD + 1,
    )
    assert any("leave-one-out" in error for error in formula_errors)


def test_modeling_validation_repairs_removed_sklearn_keyword_before_execution(monkeypatch):
    import workflows.modeling as modeling

    contract = build_analysis_contract(
        target="",
        columns=["x"],
        task_type="unsupervised",
    )
    ctx = {
        "analysis_contract": contract,
        "analysis_contract_json": contract_as_prompt(contract),
        "target": "",
        "user_input": "",
        "additional_preference": "",
        "language": "en",
        "runtime_constraints_json": "{}",
    }
    repaired_code = """
result_dict = {
    'analysis_manifest': {
        'task_type': 'unsupervised',
        'outcome': '',
        'sample_rule': 'model_specific',
        'split_strategy': 'random_80_20',
        'covariance': 'not_applicable',
        'model_structure': 'as_requested',
    },
    'models': [{'name': 'summary', 'n_obs': int(len(df)), 'metrics': {'n_obs': float(len(df))}}],
}
"""
    repaired = []

    monkeypatch.setattr(
        modeling,
        "validate_modeling_runtime_compatibility",
        lambda code, **_kwargs: ["remove multi_class"] if "multi_class" in code else [],
    )
    monkeypatch.setattr(
        modeling,
        "repair_modeling_code",
        lambda **_kwargs: repaired.append(True) or repaired_code,
    )

    result = modeling.validate_modeling_code(
        ctx=ctx,
        data='[{"x": 1}, {"x": 2}, {"x": 3}]',
        df_head='[{"x": 1}]',
        initial_code="from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(multi_class='auto')",
    )

    assert repaired == [True]
    assert result["success"] is True
    assert result["attempts"] == 2
    assert "multi_class" not in result["code"]


def test_modeling_validation_repairs_contract_prohibited_split_before_execution(monkeypatch):
    import workflows.modeling as modeling

    contract = build_analysis_contract(
        target="chip_count",
        columns=[
            "bin_id",
            "start_bp",
            "end_bp",
            "chip_count",
            "input_count",
            "gc_content",
            "mappability",
            "sequence_ambiguity_score",
        ],
        refined_suggestions=(
            "辅助模型使用 OLS linear regression 解释 chip_count 与 input_count、"
            "gc_content、mappability、sequence_ambiguity_score 的关系。"
        ),
        task_type="association_inference",
    )
    ctx = {
        "analysis_contract": contract,
        "analysis_contract_json": contract_as_prompt(contract),
        "target": "chip_count",
        "user_input": "association inference for high-signal ChIP-seq windows",
        "additional_preference": "",
        "language": "en",
        "runtime_constraints_json": "{}",
    }
    bad_code = """
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
result_dict = {}
"""
    repaired_code = """
model_name = 'OLS auxiliary association model'
result_dict = {}
"""
    valid_result = {
        "analysis_manifest": {
            "task_type": "association_inference",
            "outcome": "chip_count",
            "sample_rule": "model_specific",
            "split_strategy": "none",
            "covariance": "model_default",
            "model_structure": "as_requested",
        },
        "analysis_tables": {
            "candidate_windows": [{"bin_id": "chr21:0-49", "signal_score": 2.0}],
        },
        "models": [
            {
                "name": "OLS auxiliary association model",
                "n_obs": 100,
                "metrics": {"r_squared": 0.2, "aic": 10.0},
                "model_spec": {
                    "family": "linear_regression",
                    "outcome": "chip_count",
                    "features": [
                        "input_count",
                        "gc_content",
                        "mappability",
                        "sequence_ambiguity_score",
                    ],
                },
                "coefficients": [
                    {
                        "term": "input_count",
                        "estimate": 0.1,
                        "std_error": 0.01,
                        "ci_lower": 0.08,
                        "ci_upper": 0.12,
                        "p_value": 0.001,
                    }
                ],
            }
        ],
    }
    run_codes: list[str] = []
    repair_errors: list[str] = []

    monkeypatch.setattr(modeling, "validate_modeling_runtime_compatibility", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        modeling,
        "_run_modeling_code",
        lambda **kwargs: run_codes.append(kwargs["code"]) or {
            "is_success": True,
            "stdout": "",
            "result_json": valid_result,
        },
    )
    monkeypatch.setattr(
        modeling,
        "repair_modeling_code",
        lambda **kwargs: repair_errors.append(kwargs["error"]) or repaired_code,
    )

    result = modeling.validate_modeling_code(
        ctx=ctx,
        data='[{"chip_count": 10, "input_count": 5, "gc_content": 0.45, "mappability": 0.9, "sequence_ambiguity_score": 0.1}]',
        df_head='[{"chip_count": 10}]',
        initial_code=bad_code,
    )

    assert result["success"] is True
    assert result["attempts"] == 2
    assert any("train_test_split is prohibited" in error for error in repair_errors)
    assert run_codes == [repaired_code]


def test_modeling_retrieval_filters_contract_prohibited_code_templates():
    import workflows.modeling as modeling

    contract = build_analysis_contract(
        target="chip_count",
        columns=["chip_count", "input_count", "gc_content"],
        refined_suggestions="Use OLS linear regression as an auxiliary association model.",
        task_type="association_inference",
    )
    filtered = modeling._filter_recall_results_for_contract(
        [
            {
                "name": "Generic prediction template",
                "code": "from sklearn.model_selection import train_test_split\ntrain_test_split(X, y)",
            },
            {
                "name": "Association summary template",
                "code": "model_df = df[['chip_count', 'input_count']].dropna()",
            },
        ],
        contract,
    )

    assert [item["name"] for item in filtered] == ["Association summary template"]


def test_modeling_manual_execution_uses_shared_runtime_compatibility_gate():
    source = (FRONTEND_ROOT / "workflow/modeling/model_training.py").read_text()
    runner_source = (PROJECT_ROOT / "workflows/modeling.py").read_text()

    assert "validate_modeling_runtime_compatibility(code, n_rows=len(df))" in source
    assert "validate_code_against_contract(" in source
    assert "run_bounded_safe_exec(" in source
    assert "validate_modeling_result(" in source
    assert "safe_exec(code, exec_ns)" not in source
    assert "validate_code_against_contract(" in runner_source
    assert "validate_modeling_runtime_compatibility(" in runner_source
    assert "n_rows=n_rows" in runner_source
    assert "validate_modeling_result(" in runner_source


def test_quick_modeling_selection_does_not_persist_as_hidden普通_context():
    source = (FRONTEND_ROOT / "workflow/modeling/modeling_render.py").read_text()

    assert "modeling_quick_selection_active" in source
    assert "_clear_quick_modeling_selection(agent)" in source
    assert 'if st.session_state.get("modeling_quick_selection_active")' in source
    assert source.count("_clear_quick_modeling_selection(agent)") >= 3


def test_preprocessing_prompts_define_qc_summary_semantics():
    generation_prompt = (PROJECT_ROOT / "prompts/preprocessing/code_generation_llm_sys.txt").read_text()
    fixer_prompt = (PROJECT_ROOT / "prompts/preprocessing/code_fixer_llm_sys.txt").read_text()
    combined = generation_prompt + "\n" + fixer_prompt

    assert "missing_by_column" in combined
    assert "process_df" in combined
    assert "modified_row_count" in combined
    assert "唯一行数" in combined
    assert "imputed_cell_count" in combined


def test_modeling_fixer_prompt_allows_association_primary_outputs_without_models():
    fixer_prompt = (PROJECT_ROOT / "prompts/modeling/sec4_code_fixed_llm_sys.txt").read_text()
    generation_prompt = (PROJECT_ROOT / "prompts/modeling/sec4_code_generation_llm_sys.txt").read_text()
    combined = fixer_prompt + "\n" + generation_prompt

    assert "所有任务必须返回非空 `models` 列表" not in fixer_prompt
    assert "prediction/unsupervised 任务必须返回非空 `models` 列表" in fixer_prompt
    assert "association_inference 若已有非空主分析输出" in fixer_prompt
    assert "不得为了修复格式而强行补模型" in fixer_prompt
    assert "allowed_model_families" in combined
    assert "不是待执行模型清单" in combined
    assert "只有 `required_model_specs`" in combined
    assert "不要把系统 contract 的 `required_model_specs`" in combined
    assert "这些由系统 validator 持有，不是模型结果" in combined


def test_manual_generated_code_execution_is_bounded_in_every_page_path():
    source_paths = (
        FRONTEND_ROOT / "workflow/preprocessing/preprocessing_core.py",
        FRONTEND_ROOT / "workflow/visualization/viz_coding.py",
        FRONTEND_ROOT / "workflow/modeling/model_training.py",
        FRONTEND_ROOT / "workflow/modeling/model_inference.py",
    )

    for source_path in source_paths:
        source = source_path.read_text()
        assert "run_bounded_safe_exec(" in source
        assert "safe_exec(" not in source.replace("run_bounded_safe_exec(", "")


def test_bounded_executor_preserves_dataframe_dtypes_and_terminates_timeouts():
    from core.bounded_code_execution import run_bounded_safe_exec

    source_df = pd.DataFrame(
        {
            "id": pd.Series([1, 2], dtype="Int64"),
            "group": pd.Series(pd.Categorical(["a", "b"])),
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        },
        index=pd.Index([10, 20], name="row_id"),
    )
    completed = run_bounded_safe_exec(
        kind="preprocessing",
        code="process_df = df.copy()\nprocess_df['id_plus_one'] = process_df['id'] + 1",
        dataframe=source_df,
        timeout_seconds=20,
    )
    assert completed["is_success"] is True
    processed_df = completed["value"]
    assert processed_df.index.equals(source_df.index)
    assert str(processed_df["id"].dtype) == "Int64"
    assert str(processed_df["group"].dtype) == "category"
    assert str(processed_df["timestamp"].dtype) == "datetime64[ns, UTC]"

    np.random.seed(20260717)
    expected_random = np.random.random(2)
    np.random.seed(20260717)
    randomized = run_bounded_safe_exec(
        kind="preprocessing",
        code="process_df = df.copy()\nprocess_df['random_value'] = np.random.random(len(df))",
        dataframe=source_df,
        timeout_seconds=20,
    )
    assert randomized["is_success"] is True
    assert np.allclose(randomized["value"]["random_value"].to_numpy(), expected_random)
    assert np.isclose(np.random.random(), np.random.RandomState(20260717).random_sample(3)[2])

    timed_out = run_bounded_safe_exec(
        kind="preprocessing",
        code="for index in range(10**10):\n    value = index\nprocess_df = df",
        dataframe=source_df,
        timeout_seconds=1,
    )
    assert timed_out["is_success"] is False
    assert "超时" in timed_out["error"]


def test_modeling_bounded_executor_terminates_timeout():
    from core.bounded_code_execution import run_bounded_safe_exec

    timed_out = run_bounded_safe_exec(
        kind="modeling",
        code="for index in range(10**10):\n    value = index\nresult_dict = {'models': []}",
        dataframe=pd.DataFrame({"y": [1, 2], "x": [3, 4]}),
        timeout_seconds=1,
    )

    assert timed_out["is_success"] is False
    assert "超时" in timed_out["error"]


def test_plotly_interval_categories_are_safe_for_transport_without_changing_labels():
    import plotly.express as px
    import plotly.io as pio

    from core.plotly_serialization import figure_to_json, json_safe_figure

    frame = pd.DataFrame(
        {
            "bin": pd.IntervalIndex.from_breaks([0, 1, 2], closed="right"),
            "count": [3, 4],
        }
    )
    figure = px.bar(frame, x="bin", y="count")

    serialized = figure_to_json(figure)
    round_tripped = pio.from_json(serialized)

    assert list(json_safe_figure(figure).data[0].x) == ["(0, 1]", "(1, 2]"]
    assert list(round_tripped.data[0].x) == ["(0, 1]", "(1, 2]"]
    assert serialized


def test_visualization_transport_paths_use_interval_safe_serialization():
    paths = (
        FRONTEND_ROOT / "workflow/visualization/viz_coding.py",
        FRONTEND_ROOT / "workflow/visualization/viz_render.py",
        FRONTEND_ROOT / "workflow/report/report_render.py",
        PROJECT_ROOT / "workflows/_plugins.py",
    )
    for path in paths:
        assert "figure_to_json" in path.read_text()


def test_visualization_partial_execution_keeps_figures_created_before_a_later_error():
    from core.bounded_code_execution import run_bounded_safe_exec
    from workflows._plugins import execute_and_extract

    code = """
fig_dict = {}
fig_dict['good_scatter'] = px.scatter(df, x='x', y='y')
raise ValueError('broken second chart')
"""
    dataframe = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    manual_result = run_bounded_safe_exec(
        kind="visualization",
        code=code,
        dataframe=dataframe,
        timeout_seconds=20,
    )
    assert manual_result["is_success"] is True
    assert list(manual_result["value"]) == ["good_scatter"]
    assert manual_result["warnings"]

    workflow_result = execute_and_extract(
        code=code,
        df_data=dataframe.to_json(orient="records"),
        timeout_seconds=20,
    )
    assert workflow_result["error"] == ""
    assert [item["title"] for item in workflow_result["fig_task_list"]] == ["good_scatter"]
    assert workflow_result["warnings"]


def test_manual_visualization_figures_are_synchronized_for_report(monkeypatch):
    from utils import i18n
    from workflow.visualization import viz_coding

    class FakeVisualizationAgent:
        def __init__(self):
            self.figures = [
                {"title": "变量分布", "desc": "展示主要变量的分布形态。"},
                {"title": "变量关系", "desc": "展示两个变量之间的关系。"},
            ]

        def load_fig(self):
            return self.figures

        def load_code(self):
            return "fig_dict = {'chart_01': fig}"

        def load_suggestion(self):
            return "生成变量分布和变量关系图。"

    state = {
        "analysis_dataset_fingerprint": "dataset-a",
        "tu_title": [],
        "summary_3": {"title": "数据可视化", "fig_analysis": []},
        "abstract_3": "可视化代码生成失败：timeout",
    }
    monkeypatch.setattr(viz_coding.st, "session_state", state)
    monkeypatch.setattr(i18n.st, "session_state", state)

    assert viz_coding.sync_visualization_report_state_from_figures(FakeVisualizationAgent()) is True

    assert len(state["summary_3"]["fig_analysis"]) == 2
    assert state["summary_3"]["fig_analysis"][0]["title"] == "变量分布"
    assert "[FIG:0]" in state["full"]
    assert "失败" not in state["abstract_3"]
    assert state["viz_workflow_result"]["_synchronized_from_figures"] is True
    assert state["workflow_stage_states"]["visualization"]["status"] == "succeeded"


def test_visualization_sync_replaces_stale_report_material(monkeypatch):
    from utils import i18n
    from workflow.visualization import viz_coding

    class FakeVisualizationAgent:
        def __init__(self):
            self.figures = [
                {"title": "图1 | 新变量分布", "desc": "新的分布说明。", "chart_id": "new_distribution"},
                {"title": "Figure 2 | New Relationship", "desc": "New relationship description.", "chart_id": "new_relationship"},
            ]

        def load_fig(self):
            return self.figures

        def save_fig(self, figures):
            self.figures = figures

        def load_code(self):
            return "fig_dict = {'new_distribution': fig}"

        def load_suggestion(self):
            return "生成新的图表。"

    state = {
        "analysis_dataset_fingerprint": "dataset-a",
        "tu_title": ["旧变量分布"],
        "summary_3": {
            "title": "数据可视化",
            "fig_analysis": [{"fig": "old", "title": "旧变量分布", "analysis": "旧说明。"}],
        },
        "full": "[FIG:0] 图题：旧变量分布\n旧说明。",
        "abstract_3": "已生成旧图。",
    }
    monkeypatch.setattr(viz_coding.st, "session_state", state)
    monkeypatch.setattr(i18n.st, "session_state", state)

    assert viz_coding.sync_visualization_report_state_from_figures(FakeVisualizationAgent()) is True

    titles = state["tu_title"]
    assert titles == ["新变量分布", "New Relationship"]
    assert state["summary_3"]["fig_analysis"][0]["title"] == "新变量分布"
    assert state["summary_3"]["fig_analysis"][1]["title"] == "New Relationship"
    assert "旧变量分布" not in state["full"]
    assert "新变量分布" in state["full"]
    assert "figure_result_fingerprint" in state["viz_workflow_result"]


def test_visualization_success_clears_stale_render_and_report_image_caches(monkeypatch):
    from workflow.visualization import viz_coding

    state = {
        "viz_title_input_0_old": "Old title",
        "viz_download_image_cache": {"old": b"image"},
        "report_figure_data_uri_cache": {"0:old": "data:image/png;base64,old"},
        "report_figure_ledger": {"figures": []},
        "report_final_html": "<main>old</main>",
        "report_selected_full_conten": "[FIG:0] old",
    }
    monkeypatch.setattr(viz_coding.st, "session_state", state)

    viz_coding._clear_visualization_render_state_after_success()

    for key in (
        "viz_title_input_0_old",
        "viz_download_image_cache",
        "report_figure_data_uri_cache",
        "report_figure_ledger",
        "report_final_html",
        "report_selected_full_conten",
    ):
        assert key not in state


def test_visualization_result_keys_are_bound_to_figure_fingerprint():
    source = (FRONTEND_ROOT / "workflow/visualization/viz_render.py").read_text()

    assert 'input_key = f"viz_title_input_{idx}_{figure_key_suffix}"' in source
    assert 'key=f"fig_{idx}_{figure_key_suffix}"' in source
    assert 'key=f"viz_prepare_download_{idx}_{figure_key_suffix}"' in source
    assert 'key=f"viz_download_{idx}_{figure_key_suffix}"' in source


def test_report_figure_preparation_uses_real_title_and_white_background():
    import plotly.graph_objects as go

    from workflow.report import report_render

    figure = go.Figure(data=[go.Bar(x=["A"], y=[1])])
    figure.update_layout(title_text="真实图题")
    fig_item = {
        "fig": figure,
        "title": "图表 3",
        "chart_id": "chart_07_ecdf",
        "fig_dict_key": "chart_07__ECDF",
    }

    prepared, title = report_render._prepare_report_figure_for_output(fig_item, 0, [], "zh")
    caption = report_render._build_figure_caption(1, 0, [], "zh", title_text=title)

    assert title == "真实图题"
    assert prepared.layout.paper_bgcolor == "white"
    assert prepared.layout.plot_bgcolor == "white"
    assert caption == "图1 真实图题"


def test_report_figure_preparation_preserves_explicit_background_colors():
    import plotly.graph_objects as go

    from workflow.report import report_render

    figure = go.Figure(data=[go.Bar(x=["A"], y=[1])])
    figure.update_layout(
        title_text="Custom Background",
        paper_bgcolor="red",
        plot_bgcolor="yellow",
    )

    prepared, title = report_render._prepare_report_figure_for_output(
        {"fig": figure, "title": "Custom Background"},
        0,
        [],
        "zh",
    )

    assert title == "Custom Background"
    assert prepared.layout.paper_bgcolor == "red"
    assert prepared.layout.plot_bgcolor == "yellow"


def test_report_figure_preparation_turns_default_plotly_bluegray_to_white():
    import plotly.graph_objects as go

    from workflow.report import report_render

    figure = go.Figure(data=[go.Bar(x=["A"], y=[1])])
    assert figure.layout.plot_bgcolor is None
    assert report_render._is_default_plotly_template_background("#E5ECF6") is True
    assert report_render._is_default_plotly_template_background("rgb(229, 236, 246)") is True

    prepared, _title = report_render._prepare_report_figure_for_output(
        {"fig": figure, "title": "Default Background"},
        0,
        [],
        "zh",
    )

    assert prepared.layout.paper_bgcolor == "white"
    assert prepared.layout.plot_bgcolor == "white"


def test_figure_titles_strip_number_pipe_prefixes():
    import plotly.graph_objects as go

    from workflow.report import report_render
    from workflow.visualization import viz_coding

    assert viz_coding._normalize_visual_report_titles(
        ["图1 | 主要变量分布", "Figure 2 | Input Count Distribution", "图x|窗口局部形态"]
    ) == ["主要变量分布", "Input Count Distribution", "窗口局部形态"]
    assert report_render._normalize_figure_title_text("图3 | 主要变量分布") == "主要变量分布"
    assert report_render._normalize_figure_title_text("Figure 4 | Input Count Distribution") == "Input Count Distribution"

    figure = go.Figure(data=[go.Bar(x=["A"], y=[1])])
    fig_item = {"fig": figure, "title": "图1 | 主要变量分布"}
    prepared, title = report_render._prepare_report_figure_for_output(fig_item, 0, [], "zh")
    caption = report_render._build_figure_caption(1, 0, [], "zh", title_text=title)

    assert prepared is not None
    assert title == "主要变量分布"
    assert caption == "图1 主要变量分布"
    assert "|" not in prepared.layout.title.text


def test_report_figure_ledger_records_stable_metadata(monkeypatch):
    from workflow.report import report_render

    state = {}
    monkeypatch.setattr(report_render.st, "session_state", state)
    html = """
    <div class="report-figure-block" data-fig-index="0" data-report-figure-number="1">
      <img src="data:image/png;base64,abc" />
      <div class="report-figure-caption">图1 真实图题</div>
    </div>
    """
    fig_items = [
        {
            "title": "真实图题",
            "chart_id": "chip_count_distribution",
            "fig_dict_key": "chart_01__chip_count_distribution",
        }
    ]

    report_render._store_report_figure_ledger(
        html,
        fig_items,
        [],
        raw_refs=[0],
        valid_refs=[0],
        inserted_count=1,
        report_language="zh",
    )

    ledger = state["report_figure_ledger"]
    assert ledger["loaded_figure_count"] == 1
    assert ledger["figures"][0]["chart_id"] == "chip_count_distribution"
    assert ledger["figures"][0]["fig_dict_key"] == "chart_01__chip_count_distribution"
    assert ledger["figures"][0]["caption"] == "图1 真实图题"
    assert ledger["missing_valid_refs"] == []


def test_existing_report_figure_blocks_are_rerendered_for_target_language(monkeypatch):
    import plotly.graph_objects as go

    from workflow.report import report_render

    captured_titles = []

    def fake_cached_image(fig_index, fig, image_uri_cache):
        captured_titles.append(fig.layout.title.text)
        return f"data:image/png;base64,english-{fig_index}"

    figure = go.Figure(data=[go.Bar(x=["A"], y=[1])])
    figure.update_layout(title_text="主要变量分布")

    class FakeVisualizationAgent:
        def load_fig(self):
            return [
                {
                    "fig": figure,
                    "title": "主要变量分布",
                    "chart_id": "main_variables",
                    "fig_dict_key": "chart_01__main_variables",
                }
            ]

    state = {
        "visualization_agent": FakeVisualizationAgent(),
        "report_figure_data_uri_cache": {},
    }
    monkeypatch.setattr(report_render.st, "session_state", state)
    monkeypatch.setattr(report_render, "_get_cached_figure_data_uri", fake_cached_image)

    html = """
    <main>
      <div class="report-figure-block" data-fig-index="0" data-report-figure-number="1">
        <img src="data:image/png;base64,old-zh" />
        <div class="report-figure-caption">Figure 1 Distribution of Main Variables</div>
      </div>
    </main>
    """

    refreshed = report_render._inject_visualizations_into_html(html, report_language="en")

    assert "old-zh" not in refreshed
    assert "english-0" in refreshed
    assert captured_titles == ["Distribution of Main Variables"]
    assert state["report_figure_ledger"]["figures"][0]["caption"] == "Figure 1 Distribution of Main Variables"


def test_report_figure_localization_translates_colorbar_titles(monkeypatch):
    import plotly.graph_objects as go

    from workflow.report import report_render

    monkeypatch.setattr(
        report_render,
        "_translate_report_inline_text",
        lambda text, _language, _hint: {"信号强度": "Signal intensity", "局部密度": "Local density"}.get(str(text), str(text)),
    )
    figure = go.Figure(data=[go.Heatmap(z=[[1, 2], [3, 4]], colorbar={"title": {"text": "局部密度"}})])
    figure.update_layout(coloraxis={"colorbar": {"title": {"text": "信号强度"}}})

    localized = report_render._localize_plotly_figure_for_report(figure, "en")

    assert localized.layout.coloraxis.colorbar.title.text == "Signal intensity"
    assert localized.data[0].colorbar.title.text == "Local density"


def test_modeling_table_injection_does_not_use_plain_chapter4_fallback(monkeypatch):
    from workflow.report import report_render

    monkeypatch.setattr(
        report_render.st,
        "session_state",
        {
            "summary_4": {
                "table_title": "Model Summary",
                "table_html": "<table><tr><th>Model</th></tr><tr><td>M1</td></tr></table>",
            }
        },
    )

    html = "<main><h1>4 Discussion</h1><p>No modeling section here.</p></main>"
    injected = report_render._inject_modeling_table_into_html(html, report_language="en")

    assert "Modeling Results" in injected
    assert injected.index("Modeling Results") > injected.index("Discussion")
    assert "Table 1 Model Summary" in injected


def test_existing_report_figure_blocks_refresh_caption_from_current_figure(monkeypatch):
    import plotly.graph_objects as go

    from workflow.report import report_render

    captured_titles = []

    def fake_cached_image(fig_index, fig, image_uri_cache):
        captured_titles.append(fig.layout.title.text)
        return f"data:image/png;base64,current-{fig_index}"

    figure = go.Figure(data=[go.Bar(x=["A"], y=[1])])
    figure.update_layout(title_text="Current Figure Title")

    class FakeVisualizationAgent:
        def load_fig(self):
            return [
                {
                    "fig": figure,
                    "title": "Current Figure Title",
                    "chart_id": "current_chart",
                    "fig_dict_key": "chart_01__current_chart",
                }
            ]

    state = {"visualization_agent": FakeVisualizationAgent(), "report_figure_data_uri_cache": {}}
    monkeypatch.setattr(report_render.st, "session_state", state)
    monkeypatch.setattr(report_render, "_get_cached_figure_data_uri", fake_cached_image)

    html = """
    <main>
      <div class="report-figure-block" data-fig-index="0" data-report-figure-number="1">
        <img src="data:image/png;base64,old-image" />
        <div class="report-figure-caption">Figure 1 Old Figure Title</div>
      </div>
    </main>
    """

    refreshed = report_render._inject_visualizations_into_html(html, report_language="en")

    assert "old-image" not in refreshed
    assert "Current Figure Title" in refreshed
    assert "Old Figure Title" not in refreshed
    assert captured_titles == ["Current Figure Title"]
    assert state["report_figure_ledger"]["figures"][0]["caption"] == "Figure 1 Current Figure Title"


def test_report_textual_figure_refs_follow_actual_inserted_order(monkeypatch):
    import plotly.graph_objects as go

    from workflow.report import report_render

    def fake_cached_image(fig_index, fig, image_uri_cache):
        return f"data:image/png;base64,fig-{fig_index}"

    figure_a = go.Figure(data=[go.Bar(x=["A"], y=[1])])
    figure_b = go.Figure(data=[go.Bar(x=["B"], y=[2])])
    figure_a.update_layout(title_text="First Chart")
    figure_b.update_layout(title_text="Second Chart")

    class FakeVisualizationAgent:
        def load_fig(self):
            return [
                {"fig": figure_a, "title": "First Chart", "chart_id": "first"},
                {"fig": figure_b, "title": "Second Chart", "chart_id": "second"},
            ]

    state = {"visualization_agent": FakeVisualizationAgent(), "report_figure_data_uri_cache": {}}
    monkeypatch.setattr(report_render.st, "session_state", state)
    monkeypatch.setattr(report_render, "_get_cached_figure_data_uri", fake_cached_image)

    html = """
    <main>
      <p>As shown in Figure 2, the second chart appears first. [FIG:1]</p>
      <p>As shown in Figure 1, the first chart appears second. [FIG:0]</p>
    </main>
    """

    injected = report_render._inject_visualizations_into_html(html, report_language="en")

    assert "As shown in Figure 1, the second chart appears first." in injected
    assert "As shown in Figure 2, the first chart appears second." in injected
    assert "Figure 1 Second Chart" in injected
    assert "Figure 2 First Chart" in injected


def test_report_language_conversion_paths_refinalize_figures_for_target_language():
    source = (FRONTEND_ROOT / "workflow/report/report_render.py").read_text()
    restore_block = source[
        source.index("def _restore_report_language_version"):
        source.index("def _normalize_visualization_titles")
    ]
    convert_block = source[
        source.index("def _convert_report_content_language"):
        source.index("def _render_report_language_conversion_controls")
    ]

    assert "_finalize_report_html(html_content, \"\", report_language=language)" in restore_block
    assert "_finalize_report_html(translated_html, \"\", report_language=target_language)" in convert_block
    assert "st.session_state.pop(REPORT_FIGURE_DATA_URI_CACHE_KEY, None)" in restore_block
    assert "st.session_state.pop(REPORT_FIGURE_DATA_URI_CACHE_KEY, None)" in convert_block


def test_generated_code_error_details_do_not_nest_expanders():
    source_paths = (
        FRONTEND_ROOT / "workflow/modeling/modeling_render.py",
        FRONTEND_ROOT / "workflow/preprocessing/preprocessing_core.py",
        FRONTEND_ROOT / "workflow/preprocessing/preprocessing_render.py",
        FRONTEND_ROOT / "workflow/visualization/viz_coding.py",
        FRONTEND_ROOT / "workflow/visualization/viz_render.py",
    )

    for source_path in source_paths:
        source = source_path.read_text()
        assert 'with st.expander(bt("查看错误详情", "View error details"))' not in source


def test_validation_runners_use_the_same_safe_executor_as_the_frontend():
    plugin_source = (PROJECT_ROOT / "workflows/_plugins.py").read_text()
    modeling_source = (PROJECT_ROOT / "workflows/modeling.py").read_text()

    assert "_CODE_RUNNER_TEMPLATE" not in plugin_source
    assert "_VIZ_RUNNER_TEMPLATE" not in plugin_source
    assert "run_bounded_safe_exec(" in plugin_source
    assert "safe_exec(__USER_CODE_JSON__, _exec_ns)" not in plugin_source
    assert "run_bounded_safe_exec(" in modeling_source
    assert "safe_exec(__USER_CODE_JSON__, _exec_ns)" not in modeling_source


def test_visualization_sanitizer_removes_df_reassignment_and_invalid_plotly_options():
    from core.visualization_code_sanitizer import sanitize_visualization_code

    code = """
df = pd.DataFrame({'x': [1], 'y': [2]})
fig = px.scatter(
    df,
    x='x',
    y='y',
    trendline='ols',
    trendline_options={'ci': 0.95, 'add_constant': True},
    unsupported_option='drop-me',
)
"""
    sanitized = sanitize_visualization_code(code)

    assert "df =" not in sanitized
    assert "unsupported_option" not in sanitized
    assert "'ci'" not in sanitized
    assert "add_constant" in sanitized


def test_visualization_runner_executes_sanitized_code_on_full_runtime_dataframe():
    from workflows._plugins import execute_and_extract

    result = execute_and_extract(
        code="""
df = pd.DataFrame({'x': [999], 'y': [999]})
fig = px.scatter(df, x='x', y='y', unsupported_option='drop-me')
fig_dict = {'scatter': fig}
""",
        df_data='[{"x":1,"y":2},{"x":2,"y":3}]',
        timeout_seconds=30,
    )

    assert not result.get("error")
    assert len(result["fig_task_list"]) == 1
    assert result["fig_task_list"][0]["title"] == "scatter"


def test_plotly_image_renderer_terminates_timed_out_subprocess(monkeypatch):
    import subprocess

    import workflows.visualizing as visualizing

    class TimedOutProcess:
        pid = 12345
        returncode = None

        def communicate(self, _raw_text, timeout):
            raise subprocess.TimeoutExpired(cmd="plotly-render", timeout=timeout)

    terminated: list[object] = []
    monkeypatch.setattr(visualizing.subprocess, "Popen", lambda *args, **kwargs: TimedOutProcess())
    monkeypatch.setattr(
        visualizing,
        "_terminate_process_group",
        lambda process: terminated.append(process),
    )

    status, encoded = visualizing._render_plotly_image_base64('{"data": [], "layout": {}}')

    assert status == "timeout"
    assert encoded == ""
    assert len(terminated) == 1


def test_plotly_image_timeout_is_isolated_between_session_contexts(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import Context

    import workflows.visualizing as visualizing
    from core.llm_client import submit_with_context

    monkeypatch.setattr(visualizing, "FIG_IMAGE_TIMEOUT_SECONDS", 20)
    monkeypatch.setattr(visualizing, "DISABLE_FIGURE_IMAGES_AFTER_TIMEOUT", True)

    session_a = Context()
    session_b = Context()
    state_a = {"disabled": False}
    state_b = {"disabled": False}
    session_a.run(visualizing.bind_figure_image_render_state, state_a)
    session_b.run(visualizing.bind_figure_image_render_state, state_b)

    render_calls: list[str] = []

    def timeout_renderer(raw_text):
        render_calls.append(raw_text)
        return "timeout", ""

    monkeypatch.setattr(visualizing, "_render_plotly_image_base64", timeout_renderer)
    figure = {"data": [], "layout": {}}

    assert session_a.run(visualizing.render_plotly_image_data_url, figure) == ""
    assert state_a["disabled"] is True
    assert state_b["disabled"] is False

    def successful_renderer(raw_text):
        render_calls.append(raw_text)
        return "ok", "session-b-image"

    monkeypatch.setattr(visualizing, "_render_plotly_image_base64", successful_renderer)
    assert session_b.run(visualizing.render_plotly_image_data_url, figure) == (
        "data:image/jpeg;base64,session-b-image"
    )

    def state_reaches_render_worker():
        with ThreadPoolExecutor(max_workers=1) as pool:
            return submit_with_context(
                pool,
                lambda: visualizing._get_figure_image_render_state() is state_b,
            ).result()

    assert session_b.run(state_reaches_render_worker) is True

    calls_before_disabled_retry = len(render_calls)
    assert session_a.run(visualizing.render_plotly_image_data_url, figure) == ""
    assert len(render_calls) == calls_before_disabled_retry


def test_preprocessing_runner_supports_legacy_one_hot_encoder_sparse_argument():
    from workflows._plugins import code_runner

    result = code_runner(
        code="""
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['category']])
process_df = pd.DataFrame(encoded)
""",
        df='[{"category":"a"},{"category":"b"}]',
        timeout_seconds=30,
    )

    assert result["is_success"] is True
    processed = json.loads(result["processed_df"])
    assert len(processed) == 2
    assert len(processed[0]) == 2


def test_report_figure_limits_are_unlimited_by_default_and_zero(monkeypatch):
    import workflows.reporting_partly as reporting

    monkeypatch.delenv("AUTOSTAT_REPORT_MAX_FIGURES", raising=False)
    monkeypatch.delenv("AUTOSTAT_REPORT_MAX_FIGURES_PER_SECTION", raising=False)
    figures = list(range(25))
    contexts = {
        figure: f"[FIG:{figure}] Activity distribution chart with valid statistical context."
        for figure in figures
    }
    toc = [{"num": "1", "title": "Visualization Analysis", "figures": figures}]

    selected = reporting._select_report_figures(
        candidate_figures=figures,
        figure_contexts=contexts,
        toc_list=toc,
    )
    assert selected == figures

    monkeypatch.setenv("AUTOSTAT_REPORT_MAX_FIGURES", "0")
    monkeypatch.setenv("AUTOSTAT_REPORT_MAX_FIGURES_PER_SECTION", "0")
    assert reporting._select_report_figures(
        candidate_figures=figures,
        figure_contexts=contexts,
        toc_list=toc,
    ) == figures

    monkeypatch.setenv("AUTOSTAT_USE_LLM_FIGURE_MATCHER", "0")
    planned_toc, planned_figures, insert_plan = reporting._build_figure_insert_plan(
        toc_list=[{"num": "3", "title": "Visualization Analysis", "figures": []}],
        candidate_figures=figures,
        figure_contexts=contexts,
    )
    assert planned_figures == figures
    assert len(insert_plan) == 25
    assert planned_toc[0]["figures"] == figures

    conclusion_toc, conclusion_figures, _ = reporting._build_figure_insert_plan(
        toc_list=[{"num": "5", "title": "Conclusions and Recommendations", "figures": []}],
        candidate_figures=[0],
        figure_contexts={0: contexts[0]},
    )
    assert conclusion_figures == []
    assert conclusion_toc[0]["figures"] == []


def test_visualization_artifacts_are_scoped_to_visualization_sections(monkeypatch):
    import workflows.reporting_partly as reporting

    monkeypatch.setenv("AUTOSTAT_USE_LLM_FIGURE_MATCHER", "0")
    toc = [
        {"num": "1.4", "title": "Count Variable and Covariate Overview", "figures": []},
        {"num": "3", "title": "Visualization Analysis", "figures": []},
        {"num": "4.4", "title": "High-signal Region Screening", "figures": []},
    ]
    contexts = {
        0: "[FIG:0] Title: Count Variable and Covariate Overview\nDistribution of count variables.",
        1: "[FIG:1] Title: High-signal Region Screening\nLocal plot for high signal regions.",
    }
    artifacts = [
        {"chart_id": "viz_0001", "section_scope": "visualization", "title": "Count Variable and Covariate Overview"},
        {"chart_id": "viz_0002", "section_scope": "visualization", "title": "High-signal Region Screening"},
    ]

    planned_toc, planned_figures, insert_plan = reporting._build_figure_insert_plan(
        toc_list=toc,
        candidate_figures=[0, 1],
        figure_contexts=contexts,
        figure_artifacts=artifacts,
    )

    assert planned_figures == [0, 1]
    assert planned_toc[0]["figures"] == []
    assert planned_toc[1]["figures"] == [0, 1]
    assert planned_toc[2]["figures"] == []
    assert {item["section_num"] for item in insert_plan} == {"3"}


def test_report_figure_limit_can_still_be_enabled_explicitly(monkeypatch):
    import workflows.reporting_partly as reporting

    monkeypatch.setenv("AUTOSTAT_REPORT_MAX_FIGURES", "3")
    monkeypatch.setenv("AUTOSTAT_REPORT_MAX_FIGURES_PER_SECTION", "3")
    figures = list(range(8))
    contexts = {
        figure: f"[FIG:{figure}] Distinct chart {figure} with complete statistical context."
        for figure in figures
    }
    selected = reporting._select_report_figures(
        candidate_figures=figures,
        figure_contexts=contexts,
        toc_list=[{"num": "1", "title": "Visualization Analysis"}],
    )
    assert len(selected) == 3


def test_modeling_runner_strips_transport_heavy_values():
    from workflows.modeling import _run_modeling_code

    result = _run_modeling_code(
        code="""
result_dict = {
    'analysis_manifest': {'task_type': 'prediction'},
    'models': [{'name': 'LinearRegression', 'n_obs': 2, 'metrics': {'r2': 0.5}}],
    'artifacts': {'best_model_b64': 'x' * 5000},
    'predictions': list(range(100)),
}
""",
        data='[{"x": 1, "y": 2}, {"x": 2, "y": 3}]',
        timeout_seconds=30,
    )

    assert result["is_success"] is True
    assert result["result_json"]["artifacts"]["stripped"] is True
    assert result["result_json"]["predictions"]["count"] == 100
    assert len(result["result_json"]["predictions"]["sample"]) == 10


def test_stage_reference_context_uses_executed_evidence_and_stable_figure_ids():
    from workflows.reporting_reference_context import build_stage_reference_contexts

    contexts = build_stage_reference_contexts(
        plan={
            "shape_0": 1012,
            "shape_1": 3,
            "dtype_info_str": {"outcome": "float64", "age": "int64", "sex": "object"},
            "head_dict_str": '[{"outcome": 100.0, "age": 7, "sex": "F"}]',
            "data_profile_str": '{"row_count": 1012, "total_missing_values": 4}',
        },
        loading={"summary_1": {"desc": "Loaded selected sample."}, "abstract_1": "Loaded."},
        prep={
            "summary_2": {
                "code": "work_df = df.dropna()",
                "processed_df": '[{"outcome": 100.0, "age": 7}]',
            },
            "abstract_2": "Complete cases retained.",
        },
        viz={
            "summary_3": {
                "fig_analysis": [
                    {"title": "Outcome distribution", "analysis": "Figure 1 shows the spread."}
                ]
            },
            "abstract_3": "Distribution reviewed.",
            "full": "Figure 1 shows the spread.",
        },
        model={
            "summary_4": {
                "table_title": "Nested models",
                "table_markdown": "| Model | R2 |\n|---|---|\n| M1 | 0.31 |",
                "result": "M1 used HC3.",
                "report_artifacts": {
                    "coefficients": {
                        "count": 1,
                        "rows": [{"term": "age", "estimate": 0.2}],
                    }
                },
            },
            "_modeling_report_artifacts": {
                "candidate_segments": {
                    "count": 1,
                    "rows": [{"chromosome": "chr1", "start_bp": 1}],
                }
            },
            "abstract_4": "Association analysis.",
        },
        next_cols=["outcome", "age"],
    )

    assert '"rows": 1012' in contexts["loading"]
    assert "total_missing_values" in contexts["loading"]
    assert "work_df = df.dropna()" in contexts["preprocessing"]
    assert "[FIG:0]" in contexts["visualization"]
    assert "Figure 1" not in contexts["visualization"]
    assert "0.31" in contexts["modeling"]
    assert "candidate_segments" in contexts["modeling"]
    assert "chr1" in contexts["modeling"]


def test_autostat_records_parallel_stage_runtime_and_passes_report_evidence(monkeypatch):
    import workflows.autostat as autostat

    data = '[{"outcome": 1.0, "x": 2.0}]'
    captured: dict[str, object] = {}
    loading_kwargs: dict[str, object] = {}
    monkeypatch.setattr(
        autostat,
        "run_planning_workflow",
        lambda **kwargs: {
            "shape_0": 1,
            "shape_1": 2,
            "dtype_info_str": '{"outcome":"float64","x":"float64"}',
            "head_dict_str": data,
            "data_profile_str": '{"row_count": 1, "column_count": 2}',
            "df": data,
            "loading_auto": True,
            "prep_auto": True,
            "vis_auto": True,
            "modeling_auto": True,
            "report_auto": True,
        },
    )
    monkeypatch.setattr(
        autostat,
        "run_loading_workflow",
        lambda **kwargs: (
            loading_kwargs.update(kwargs)
            or {"summary_1": {"desc": "loaded"}, "abstract_1": "loaded"}
        ),
    )
    monkeypatch.setattr(
        autostat,
        "run_preprocessing_workflow",
        lambda **kwargs: {
            "summary_2": {"processed_df": data, "code": "work_df = df.copy()"},
            "abstract_2": "prepared",
            "_status": "succeeded",
        },
    )
    monkeypatch.setattr(
        autostat,
        "run_visualizing_workflow",
        lambda **kwargs: {
            "summary_3": {"fig_analysis": []},
            "abstract_3": "visualized",
            "full": "",
            "_status": "succeeded",
        },
    )
    monkeypatch.setattr(
        autostat,
        "run_modeling_workflow",
        lambda **kwargs: {
            "summary_4": {
                "table_title": "Model table",
                "table_markdown": "| Model | R2 |\n|---|---|\n| M1 | 0.5 |",
            },
            "abstract_4": "modeled",
            "_status": "succeeded",
        },
    )
    monkeypatch.setattr(
        autostat,
        "run_reporting_toc_workflow",
        lambda **kwargs: {
            "toc_text": "1. Results",
            "selected_full_conten": "",
            "load_abstract": "loaded",
            "preproc_abstract": "prepared",
            "visual_abstract": "visualized",
            "coding_abstract": "modeled",
        },
    )

    def report_partly(**kwargs):
        captured.update(kwargs)
        return {"final_html": "<p>done</p>", "final_html_parts": ["<p>done</p>"], "title": "Done"}

    monkeypatch.setattr(autostat, "run_reporting_partly_workflow", report_partly)

    result = autostat.run_autostat(pd.DataFrame({"outcome": [1.0], "x": [2.0]}))

    steps = [event["step"] for event in result["runtime_events"]]
    assert steps[0] == "planning"
    assert set(steps) == {
        "planning", "loading", "preprocessing", "visualizing", "modeling",
        "reporting_toc", "reporting_partly",
    }
    assert max(steps.index("loading"), steps.index("preprocessing")) < min(
        steps.index("visualizing"), steps.index("modeling")
    )
    assert steps[-2:] == ["reporting_toc", "reporting_partly"]
    assert all(event["runtime_seconds"] >= 0 for event in result["runtime_events"])
    stage_contexts = captured["stage_reference_contexts"]
    assert isinstance(stage_contexts, dict)
    assert loading_kwargs["data_profile_str"] == '{"row_count": 1, "column_count": 2}'
    assert "work_df = df.copy()" in stage_contexts["preprocessing"]
    assert "0.5" in stage_contexts["modeling"]


def test_reference_parser_reports_each_file_independently():
    files = [
        NamedBytesIO("valid.txt", "第一段参考资料。\n\n第二段。".encode("utf-8")),
        NamedBytesIO("empty.txt", b""),
    ]
    results = parse_and_chunk_results(files)

    assert [result["name"] for result in results] == ["valid.txt", "empty.txt"]
    assert results[0]["status"] == "success"
    assert results[0]["chunk_count"] > 0
    assert results[1]["status"] == "empty"
    assert results[1]["chunk_count"] == 0


def test_planner_missing_fields_use_complete_true_defaults(monkeypatch):
    import workflows.planning as planning

    monkeypatch.setattr(planning, "chat_json", lambda *args, **kwargs: {"loading_auto": False})
    monkeypatch.setattr(planning, "chat", lambda *args, **kwargs: "plan")

    result = planning.run_planning_workflow(df=pd.DataFrame({"x": [1, 2]}))

    assert result["loading_auto"] is False
    assert result["prep_auto"] is True
    assert result["vis_auto"] is True
    assert result["modeling_auto"] is True
    assert result["report_auto"] is True


def test_planning_uses_the_original_full_metadata_and_output_limits(monkeypatch):
    import workflows.planning as planning

    calls: dict[str, dict] = {}

    def fake_chat_json(*args, **kwargs):
        calls["planner"] = kwargs
        return {
            "loading_auto": True,
            "prep_auto": True,
            "vis_auto": True,
            "modeling_auto": True,
            "report_auto": True,
        }

    def fake_chat(*args, **kwargs):
        calls["path"] = kwargs
        return "full plan"

    monkeypatch.setattr(planning, "chat_json", fake_chat_json)
    monkeypatch.setattr(planning, "chat", fake_chat)

    result = planning.run_planning_workflow(df=pd.DataFrame({"x": [1, 2]}))

    assert result["df"]
    assert calls["planner"]["temperature"] == 1.0
    assert "max_tokens" not in calls["planner"]
    assert calls["path"]["temperature"] == 0.5
    assert calls["path"]["max_tokens"] == 8192


def test_repair_prompts_keep_code_requirements_and_contract():
    ctx = {
        "df_head": "HEAD_SENTINEL",
        "refined_suggestions": "SUGGESTION_SENTINEL",
        "model_suggestion": "FULL_SUGGESTION_SENTINEL",
        "analysis_contract_json": "CONTRACT_SENTINEL",
        "error": "ERROR_SENTINEL",
        "code": "CODE_SENTINEL",
        "preference_select": "PREF_SENTINEL",
        "additional_preference": "EXTRA_SENTINEL",
        "language_instruction": "",
    }
    rendered = render_file("modeling/sec4_code_fixed_llm_user.txt", ctx, strict=True)
    for sentinel in (
        "SUGGESTION_SENTINEL",
        "FULL_SUGGESTION_SENTINEL",
        "CONTRACT_SENTINEL",
        "ERROR_SENTINEL",
        "CODE_SENTINEL",
    ):
        assert sentinel in rendered
    assert rendered.count("HEAD_SENTINEL") == 1
    assert "此处填入" not in rendered

    with pytest.raises(KeyError):
        render_file("modeling/sec4_code_fixed_llm_user.txt", {"df_head": "x"}, strict=True)


def test_visualization_repair_prompt_uses_canonical_context_keys():
    ctx = {
        "error": "ERROR_SENTINEL",
        "code": "CODE_SENTINEL",
        "def_head": "HEAD_SENTINEL",
        "color": "COLOR_SENTINEL",
        "refined_suggestions": "SUGGESTION_SENTINEL",
        "visual_recommendation": "FULL_VISUAL_SENTINEL",
        "visualization_contract_json": "VISUAL_CONTRACT_SENTINEL",
        "preference_selected": "PREF_SENTINEL",
        "add_preference": "EXTRA_SENTINEL",
        "language_name": "English",
        "language_instruction": "",
    }
    rendered = render_file("visualizing/sec3_fixed_code_llm_user.txt", ctx, strict=True)
    for sentinel in (
        "ERROR_SENTINEL",
        "CODE_SENTINEL",
        "SUGGESTION_SENTINEL",
        "FULL_VISUAL_SENTINEL",
        "VISUAL_CONTRACT_SENTINEL",
        "PREF_SENTINEL",
        "EXTRA_SENTINEL",
    ):
        assert sentinel in rendered
    assert rendered.count("HEAD_SENTINEL") == 1

    system_prompt = render_file(
        "visualizing/sec3_fixed_code_llm_sys.txt",
        {
            "language_instruction": "",
            "runtime_constraints_json": "CONSTRAINTS_SENTINEL",
        },
        strict=True,
    )
    assert "可独立运行" not in system_prompt
    assert "fig_dict" in system_prompt
    assert "CONSTRAINTS_SENTINEL" in system_prompt


def test_visualization_full_data_failure_enters_same_repair_loop(monkeypatch):
    import workflows.visualizing as viz

    generated_codes = iter(["bad_code", "good_code"])
    full_calls: list[str] = []

    monkeypatch.setattr(viz, "render_file", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(viz, "chat_code", lambda *args, **kwargs: next(generated_codes))
    monkeypatch.setattr(
        viz,
        "validate_viz_code",
        lambda **kwargs: {
            "is_success": True,
            "final_code": kwargs["code"],
            "error_msg": "",
            "fig_task_list": [{"fig": "preview"}],
        },
    )

    def execute_full(**kwargs):
        full_calls.append(kwargs["code"])
        if kwargs["code"] == "bad_code":
            return {"fig_task_list": [], "error": "full-only failure"}
        return {"fig_task_list": [{"title": "plot", "fig": "{}"}]}

    monkeypatch.setattr(viz, "execute_and_extract", execute_full)
    monkeypatch.setattr(viz, "_attach_figure_llm_artifacts", lambda items: items)
    monkeypatch.setattr(
        viz,
        "_batch_run",
        lambda *args, **kwargs: [{"title": "Plot", "summary": {"analysis": "ok"}}],
    )
    monkeypatch.setattr(viz, "sec3_check_full", lambda **kwargs: {"full": "analysis"})
    monkeypatch.setattr(viz, "chat", lambda *args, **kwargs: "abstract")
    monkeypatch.setattr(viz, "sec3_composer", lambda **kwargs: {"summary_3": {"fig_analysis": kwargs["fig_analysis"]}})

    result = viz.run_visualizing_phase2(
        ctx={
            "visual_recommendatio": "recommendation",
            "refined_suggestions": "suggestion",
            "language": "en",
        },
        data='[{"x":1},{"x":2}]',
        cols=["x"],
        def_head='[{"x":1}]',
    )

    assert full_calls == ["bad_code", "good_code"]
    assert result["_status"] == "succeeded"
    assert result["final_code"] == "good_code"
    assert result["_fix_attempts"] == 2


def test_suggestion_revision_loop_keeps_complete_conversation_history():
    session: dict = {}
    state = get_suggestion_state(session, "modeling")

    add_requirement(state, "Use three nested HC3 models.")
    replace_active_suggestion(state, "version one")
    add_revision_request(state, "Use one common complete-case sample.")
    replace_active_suggestion(
        state,
        "version two",
        revision_instruction="Use one common complete-case sample.",
    )
    add_revision_request(state, "Keep continuous variables in natural units.")
    replace_active_suggestion(
        state,
        "version three",
        revision_instruction="Keep continuous variables in natural units.",
    )

    assert state["active_suggestion"] == "version three"
    assert [item["suggestion"] for item in state["versions"]] == [
        "version one",
        "version two",
        "version three",
    ]
    assert confirm_active_suggestion(state) is True
    assert state["confirmed_version"] == 3
    visible_suggestions = [
        item["content"] for item in visible_messages(state) if item.get("kind") == "suggestion"
    ]
    assert visible_suggestions == ["version one", "version two", "version three"]


def test_revision_is_queued_for_the_next_render_after_user_message_is_added():
    state = get_suggestion_state({}, "loading")
    replace_active_suggestion(state, "version one")

    assert queue_revision_request(state, "Please revise this.") == "Please revise this."
    assert [item["content"] for item in visible_messages(state)] == [
        "version one",
        "Please revise this.",
    ]
    assert take_pending_revision(state) == "Please revise this."
    assert take_pending_revision(state) == ""


def test_code_source_suggestion_fingerprint_marks_stale_code_after_revision():
    state = get_suggestion_state({}, "visualization")
    replace_active_suggestion(state, "plot distributions")
    assert confirm_active_suggestion(state) is True
    record_successful_code(state, "fig_dict = {'a': fig}")
    assert code_matches_current_suggestion(state) is True

    replace_active_suggestion(state, "plot distributions and correlations")
    mark_code_draft(state, "fig_dict = {'a': fig}")

    assert code_matches_current_suggestion(state) is False


def test_code_revision_request_uses_a_separate_queue_from_suggestion_revision():
    state = get_suggestion_state({}, "modeling")
    replace_active_suggestion(state, "fit a Poisson model")

    assert queue_code_revision_request(state, "Add robust covariance.") == "Add robust covariance."
    assert take_pending_revision(state) == ""
    assert take_pending_code_revision(state) == "Add robust covariance."
    assert take_pending_code_revision(state) == ""
    assert [item["kind"] for item in visible_messages(state)] == ["suggestion", "code_revision"]


def test_code_revision_buttons_run_from_their_execution_panels():
    sources = {
        "preprocessing": (
            FRONTEND_ROOT / "workflow/preprocessing/preprocessing_core.py",
            "_revise_preprocessing_code_from_execution",
            "prep_code_revision_flash",
        ),
        "visualization": (
            FRONTEND_ROOT / "workflow/visualization/viz_coding.py",
            "_revise_visualization_code_draft_from_execution",
            "viz_code_revision_flash",
        ),
        "modeling": (
            FRONTEND_ROOT / "workflow/modeling/modeling_render.py",
            "_revise_modeling_code_draft",
            "modeling_code_revision_flash",
        ),
    }

    for _stage, (path, helper_name, flash_key) in sources.items():
        source = path.read_text()
        assert helper_name in source
        assert flash_key in source
        assert "current_code_override=sanitize" in source
        assert "queue_code_revision_request(state, revision_text)" not in source


def test_initial_request_is_queued_after_user_message_is_added():
    state = get_suggestion_state({}, "preprocessing")

    assert queue_initial_request(state, "Please preprocess this.") == "Please preprocess this."
    assert [item["content"] for item in visible_messages(state)] == ["Please preprocess this."]
    assert state["base_requirements"] == ["Please preprocess this."]
    assert take_pending_initial_request(state) == "Please preprocess this."
    assert take_pending_initial_request(state) == ""


def test_data_interpretation_labels_replace_data_suggestion_wording():
    loading_source = (FRONTEND_ROOT / "workflow/dataloading/dataloading_render.py").read_text()
    app_source = (FRONTEND_ROOT / "app.py").read_text()

    assert 'bt("数据解析", "Data Interpretation")' in loading_source
    assert 'bt("🔍 生成数据解析", "🔍 Generate Data Interpretation")' in loading_source
    assert 'bt("数据建议", "Data Suggestions")' not in loading_source + app_source


def test_import_page_file_lists_wrap_long_names_without_squeezing_action_buttons():
    loading_source = (FRONTEND_ROOT / "workflow/dataloading/dataloading_render.py").read_text()

    assert "def _render_import_file_list(" in loading_source
    assert "overflow-wrap: anywhere" in loading_source
    assert "word-break: break-word" in loading_source
    assert "st.columns([4, 1.15], gap=\"small\")" in loading_source
    assert 'label_col.write(f"- 📄 {name}")' not in loading_source
    assert '"+ \", \".join(persisted_names)' not in loading_source


def test_import_page_loading_analysis_passes_full_profile_and_reference_context():
    loading_source = (FRONTEND_ROOT / "workflow/dataloading/dataloading_render.py").read_text()
    bridge_source = (FRONTEND_ROOT / "utils/local_workflow_bridge.py").read_text()

    assert "_retrieve_loading_reference_context(df, user_input)" in loading_source
    assert "data_profile_str=meta.get(\"data_profile_str\", \"\")" in loading_source
    assert "ref_context=loading_ref_context" in loading_source
    assert "数据字典 字段说明 变量含义" in loading_source
    assert "data_profile_str=str(inputs.get(\"data_profile_str\", \"\"))" in bridge_source
    assert "data dictionary field descriptions variable meanings" in bridge_source


def test_ui_and_report_language_widgets_stay_synchronized_across_reruns():
    app_source = (FRONTEND_ROOT / "app.py").read_text()
    report_source = (FRONTEND_ROOT / "workflow/report/report_render.py").read_text()
    report_controls = report_source[
        report_source.index("def report_basic_info("):
        report_source.index("def report_outline(")
    ]

    assert 'ui_language_widget_sync_key = "ui_language_widget_synced"' in app_source
    assert "st.session_state[ui_language_widget_key] = current_language" in app_source
    assert 'sync_report_language(st.session_state.get("report_agent"), selected_language)' in app_source
    assert "saved_language = sync_report_language(report_agent)" in report_controls
    assert "set_language(selected_language)" in report_controls
    assert "sync_report_language(report_agent, selected_language)" in report_controls
    assert "st.session_state.pop(REPORT_OUTLINE_LENGTH_SELECTOR_KEY, None)" in report_controls
    assert "st.rerun()" in report_controls


def test_sidebar_llm_config_panel_uses_i18n_for_visible_copy():
    llm_source = (FRONTEND_ROOT / "settings/llm_config.py").read_text()
    i18n_source = (FRONTEND_ROOT / "utils/i18n.py").read_text()

    forbidden_hardcoded_widgets = [
        'st.caption("选择预设服务商',
        '"模型服务商",',
        '"保存到本机用户配置",',
        'st.button("保存配置"',
        'st.success("配置已保存。")',
        'st.error(f"配置无效：{exc}")',
        'st.warning("请填写 API Key、Base URL 和 Model。")',
        "已就绪 ·",
        "未连接，请填写完整配置后保存。",
    ]
    for snippet in forbidden_hardcoded_widgets:
        assert snippet not in llm_source

    required_translation_keys = [
        "sidebar.llm_caption",
        "sidebar.llm_provider",
        "sidebar.remember_config",
        "sidebar.remember_config_help",
        "sidebar.save_config",
        "sidebar.config_saved",
        "sidebar.config_invalid",
        "sidebar.fill_llm_fields",
        "sidebar.status_ready",
        "sidebar.status_not_connected",
    ]
    for key in required_translation_keys:
        assert f't("{key}"' in llm_source
        assert f'"{key}"' in i18n_source


def test_pdf_parser_is_a_required_runtime_dependency():
    requirements = (PROJECT_ROOT / "requirements.txt").read_text().splitlines()

    assert any(line.startswith("PyMuPDF>=") for line in requirements)


def test_dependency_and_ci_guardrails_are_enabled():
    requirements = (PROJECT_ROOT / "requirements.txt").read_text()
    workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text()

    assert "chardet>=5.0.0,<6.0.0" in requirements
    assert "PyMuPDF>=" in requirements
    assert "python -m pip check" in workflow
    assert "Record Docker image size" in workflow
    assert "GITHUB_STEP_SUMMARY" in workflow


def test_revision_prompt_contains_original_current_feedback_and_constraints(monkeypatch):
    import core.suggestion_revision as revision

    captured: dict[str, str] = {}

    def fake_chat(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return "REVISED_SENTINEL"

    monkeypatch.setattr(revision, "chat_suggestion", fake_chat)
    output = revision.revise_suggestion(
        stage_label="modeling",
        original_requirements="ORIGINAL_SENTINEL",
        current_suggestion="CURRENT_SENTINEL",
        revision_instruction="FEEDBACK_SENTINEL",
        hard_constraints="CONSTRAINT_SENTINEL",
        language_instruction="LANGUAGE_SENTINEL",
    )

    assert output == "REVISED_SENTINEL"
    combined = captured["system"] + captured["user"]
    for sentinel in (
        "ORIGINAL_SENTINEL",
        "CURRENT_SENTINEL",
        "FEEDBACK_SENTINEL",
        "CONSTRAINT_SENTINEL",
        "LANGUAGE_SENTINEL",
    ):
        assert sentinel in combined


def test_edited_code_makes_old_result_stale_until_exact_draft_succeeds():
    state = get_suggestion_state({}, "visualization")
    first_run_id = begin_code_execution(state, "result = 1")
    assert finish_code_execution(state, first_run_id, success=True) is True

    _, is_current = mark_code_draft(state, "result = 2")
    assert is_current is False
    assert state["code_status"] == "stale"

    second_run_id = begin_code_execution(state, "result = 2")
    assert second_run_id != first_run_id
    assert finish_code_execution(state, second_run_id, success=True) is True
    _, is_current = mark_code_draft(state, "result = 2")
    assert is_current is True
    assert state["code_status"] == "succeeded"
