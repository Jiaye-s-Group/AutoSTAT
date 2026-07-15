from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
if str(FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(FRONTEND_ROOT))

from core.modeling_contract import (
    build_analysis_contract,
    contract_as_prompt,
    validate_result_against_contract,
    validate_result_schema,
)
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
    confirm_active_suggestion,
    finish_code_execution,
    get_suggestion_state,
    mark_code_draft,
    queue_revision_request,
    record_auto_repair,
    record_execution_failure,
    record_successful_code,
    replace_active_suggestion,
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
    monkeypatch.setattr(modeling, "chat", lambda *args, **kwargs: next(responses))

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
        lambda code: ["remove multi_class"] if "multi_class" in code else [],
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


def test_modeling_manual_execution_uses_shared_runtime_compatibility_gate():
    source = (FRONTEND_ROOT / "workflow/modeling/model_training.py").read_text()
    runner_source = (PROJECT_ROOT / "workflows/modeling.py").read_text()

    assert "validate_modeling_runtime_compatibility(code)" in source
    assert "validate_modeling_runtime_compatibility(current_code)" in runner_source


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
            },
            "abstract_4": "Association analysis.",
        },
        next_cols=["outcome", "age"],
    )

    assert '"rows": 1012' in contexts["loading"]
    assert "work_df = df.dropna()" in contexts["preprocessing"]
    assert "[FIG:0]" in contexts["visualization"]
    assert "Figure 1" not in contexts["visualization"]
    assert "0.31" in contexts["modeling"]


def test_autostat_records_parallel_stage_runtime_and_passes_report_evidence(monkeypatch):
    import workflows.autostat as autostat

    data = '[{"outcome": 1.0, "x": 2.0}]'
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        autostat,
        "run_planning_workflow",
        lambda **kwargs: {
            "shape_0": 1,
            "shape_1": 2,
            "dtype_info_str": '{"outcome":"float64","x":"float64"}',
            "head_dict_str": data,
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
        lambda **kwargs: {"summary_1": {"desc": "loaded"}, "abstract_1": "loaded"},
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


def test_data_interpretation_labels_replace_data_suggestion_wording():
    loading_source = (FRONTEND_ROOT / "workflow/dataloading/dataloading_render.py").read_text()
    app_source = (FRONTEND_ROOT / "app.py").read_text()

    assert 'bt("数据解析", "Data Interpretation")' in loading_source
    assert 'bt("🔍 生成数据解析", "🔍 Generate Data Interpretation")' in loading_source
    assert 'bt("数据建议", "Data Suggestions")' not in loading_source + app_source


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

    monkeypatch.setattr(revision, "chat", fake_chat)
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
