from __future__ import annotations

import json

import pandas as pd

from core.modeling_contract import (
    build_analysis_contract,
    validate_code_against_contract,
    validate_modeling_result,
    validate_result_against_contract,
    validate_result_schema,
)
from core.bounded_code_execution import run_bounded_safe_exec
from core.modeling_runtime_compat import validate_modeling_runtime_compatibility
from core.code_runtime_profile import LARGE_DATASET_ROW_THRESHOLD
from core.preprocessing_contract import build_preprocessing_contract, validate_preprocessing_result
from core.visualization_contract import build_visualization_contract, validate_visualization_result
from workflows._plugins import code_runner


_PREPROCESSING_PLAN = """
保留全部 3 行，不删除记录。
不创建任何富集、峰值或下游模型变量。
仅对实际缺失的 gc_content 使用全局中位数填补。
其余字段保持原样，不进行标准化、log1p 变换或类别编码。
坐标与 bin_id 仅做 QC 记录，输出可复核 QC 摘要；end_bp = start_bp + 49，相邻合格窗口 start_bp 相差 50。
"""


def _source_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bin_id": ["chr1:0-49", "chr1:50-99", "chr1:100-149"],
            "chromosome": ["chr1", "chr1", "chr1"],
            "start_bp": [0, 50, 100],
            "end_bp": [49, 99, 149],
            "gc_content": [0.4, None, 0.6],
            "sequence_ambiguity_score": [0.1, 0.2, 0.3],
        }
    )


def _valid_qc(output_df: pd.DataFrame) -> dict:
    return {
        "rows_before": 3,
        "columns_before": 6,
        "rows_after": 3,
        "columns_after": 6,
        "missing_by_column": {str(column): int(output_df[column].isna().sum()) for column in output_df.columns},
        "modified_fields": ["gc_content"],
        "modified_row_count": 1,
        "duplicate_bin_id_count": 0,
        "coordinate_or_id_mismatch_count": 0,
        "end_bp_rule_violation_count": 0,
        "adjacent_window_gap_violation_count": 0,
    }


def test_preprocessing_contract_rejects_row_deletion_and_nonmissing_mutation():
    source = _source_df()
    contract = build_preprocessing_contract(
        columns=list(source.columns),
        suggestion=_PREPROCESSING_PLAN,
    )
    assert contract["retain_all_rows"] is True
    assert contract["actual_missing_fill_columns"] == ["gc_content"]

    dropped = source.iloc[1:].copy()
    issues = validate_preprocessing_result(
        input_df=source,
        output_df=dropped,
        qc_summary=_valid_qc(dropped),
        contract=contract,
    )
    assert any("retains all rows" in issue for issue in issues)

    changed = source.copy()
    changed.loc[0, "gc_content"] = 0.9
    changed.loc[1, "gc_content"] = 0.5
    issues = validate_preprocessing_result(
        input_df=source,
        output_df=changed,
        qc_summary=_valid_qc(changed),
        contract=contract,
    )
    assert any("not genuinely missing" in issue for issue in issues)


def test_preprocessing_contract_accepts_minimal_actual_missing_fill_with_auditable_qc():
    source = _source_df()
    output = source.copy()
    output.loc[output["gc_content"].isna(), "gc_content"] = 0.5
    contract = build_preprocessing_contract(columns=list(source.columns), suggestion=_PREPROCESSING_PLAN)

    assert validate_preprocessing_result(
        input_df=source,
        output_df=output,
        qc_summary=_valid_qc(output),
        contract=contract,
    ) == []


def test_preprocessing_runner_rejects_executable_code_that_violates_the_contract():
    source = _source_df()
    contract = build_preprocessing_contract(columns=list(source.columns), suggestion=_PREPROCESSING_PLAN)
    result = code_runner(
        code="process_df = df.iloc[1:].copy()\nqc_summary = {}",
        df=source.to_json(orient="records"),
        preprocessing_contract=contract,
    )
    assert result["is_success"] is False
    assert "retains all rows" in result["error"]


def test_preprocessing_runner_accepts_compliant_code_and_returns_qc_summary():
    source = _source_df()
    contract = build_preprocessing_contract(columns=list(source.columns), suggestion=_PREPROCESSING_PLAN)
    code = """
process_df = df.copy()
_missing = process_df['gc_content'].isna()
process_df.loc[_missing, 'gc_content'] = process_df['gc_content'].median()
qc_summary = {
    'rows_before': len(df),
    'columns_before': len(df.columns),
    'rows_after': len(process_df),
    'columns_after': len(process_df.columns),
    'missing_by_column': {str(column): int(process_df[column].isna().sum()) for column in process_df.columns},
    'modified_fields': ['gc_content'],
    'modified_row_count': int(_missing.sum()),
    'duplicate_bin_id_count': 0,
    'coordinate_or_id_mismatch_count': 0,
    'end_bp_rule_violation_count': 0,
    'adjacent_window_gap_violation_count': 0,
}
process_df = process_df
"""
    result = code_runner(
        code=code,
        df=source.to_json(orient="records"),
        preprocessing_contract=contract,
    )
    assert result["is_success"] is True
    assert result["qc_summary"]["modified_fields"] == ["gc_content"]


def test_preprocessing_qc_counts_unique_modified_rows_not_imputed_cells():
    source = pd.DataFrame(
        {
            "bin_id": ["a", "b", "c"],
            "gc_content": [None, None, 0.5],
            "sequence_ambiguity_score": [None, None, 0.2],
            "chip_count": [10, 20, 30],
        }
    )
    output = source.copy()
    output["gc_content"] = output["gc_content"].fillna(output["gc_content"].median())
    output["sequence_ambiguity_score"] = output["sequence_ambiguity_score"].fillna(
        output["sequence_ambiguity_score"].median()
    )
    contract = build_preprocessing_contract(
        columns=list(source.columns),
        suggestion=(
            "保留全部记录；只对 gc_content 和 sequence_ambiguity_score 的实际缺失做中位数填补；"
            "其他字段不处理；输出可复核 QC summary。"
        ),
    )
    wrong_qc = {
        "rows_before": 3,
        "columns_before": 4,
        "rows_after": 3,
        "columns_after": 4,
        "missing_by_column": {str(column): int(source[column].isna().sum()) for column in source.columns},
        "modified_fields": ["gc_content", "sequence_ambiguity_score"],
        "modified_row_count": 4,
    }
    issues = validate_preprocessing_result(
        input_df=source,
        output_df=output,
        qc_summary=wrong_qc,
        contract=contract,
    )
    assert any("missing_by_column" in issue for issue in issues)
    assert any("modified_row_count must be 2, got 4" in issue for issue in issues)

    modified_row_mask = source["gc_content"].isna() | source["sequence_ambiguity_score"].isna()
    right_qc = {
        **wrong_qc,
        "missing_by_column": {str(column): int(output[column].isna().sum()) for column in output.columns},
        "missing_before_by_column": {str(column): int(source[column].isna().sum()) for column in source.columns},
        "missing_after_by_column": {str(column): int(output[column].isna().sum()) for column in output.columns},
        "modified_row_count": int(modified_row_mask.sum()),
        "imputed_cell_count": 4,
    }
    assert validate_preprocessing_result(
        input_df=source,
        output_df=output,
        qc_summary=right_qc,
        contract=contract,
    ) == []


def test_visualization_contract_reports_partial_results_instead_of_discarding_valid_figures():
    contract = build_visualization_contract(
        refined_suggestions="gc_content: histogram, box plot\n总体: correlation heatmap"
    )
    assert [chart["id"] for chart in contract["charts"]] == ["chart_01", "chart_02", "chart_03"]

    result = validate_visualization_result(
        figure_keys=["chart_01__histogram"],
        contract=contract,
    )
    assert result["status"] == "partial"
    assert [item["id"] for item in result["missing_charts"]] == ["chart_02", "chart_03"]


def test_visualization_contract_accepts_numbered_chart_lines_without_colons():
    contract = build_visualization_contract(
        refined_suggestions="1. Histogram of gc_content\n2) Scatter plot of gc_content vs signal"
    )

    assert [chart["id"] for chart in contract["charts"]] == ["chart_01", "chart_02"]
    assert contract["charts"][0]["spec"] == "Histogram of gc_content"
    assert contract["charts"][1]["spec"] == "Scatter plot of gc_content vs signal"


def test_modeling_contract_requires_reported_offset_for_explicit_offset_model():
    contract = build_analysis_contract(
        target="chip_count",
        columns=["chip_count", "input_count", "mappability"],
        refined_suggestions="训练 NegativeBinomial 模型，使用 input_count 作为 offset。",
        task_type="prediction",
    )
    assert contract["required_model_specs"][0]["requires_offset"] is True
    result = {
        "analysis_manifest": {
            "task_type": "prediction",
            "outcome": "chip_count",
            "sample_rule": "model_specific",
            "split_strategy": "random_80_20",
            "covariance": "not_applicable",
            "model_structure": "as_requested",
        },
        "models": [
            {
                "name": "NegativeBinomial chip_count",
                "model_spec": {"family": "negative_binomial", "outcome": "chip_count", "features": ["mappability"]},
                "metrics": {"aic": 1.0},
            }
        ],
    }
    issues = validate_result_against_contract(code="", result_json=result, contract=contract)
    assert any("offset/exposure" in issue for issue in issues)


def test_modeling_result_validator_combines_contract_and_schema_checks():
    contract = build_analysis_contract(
        target="outcome",
        columns=["outcome", "x"],
        user_input="Prediction for outcome.",
    )
    result = {
        "analysis_manifest": {
            "task_type": "prediction",
            "outcome": "outcome",
            "sample_rule": "model_specific",
            "split_strategy": "random_80_20",
            "covariance": "not_applicable",
            "model_structure": "as_requested",
        },
        "models": [{"name": "Model 1", "metrics": {"r2": 0.2}}],
    }

    issues = validate_modeling_result(code="", result_json=result, contract=contract)

    assert any("n_obs" in issue for issue in issues)


def test_modeling_contract_keeps_model_family_from_original_suggestion_when_refined_text_omits_it():
    contract = build_analysis_contract(
        target="chip_count",
        columns=["chip_count", "input_count", "mappability"],
        model_suggestion="Use a Poisson regression for chip_count.",
        refined_suggestions="Report n_obs, metrics, and analysis_manifest.",
        task_type="prediction",
    )

    assert [spec["family"] for spec in contract["required_model_specs"]] == ["poisson"]


def test_modeling_contract_does_not_turn_allowed_optional_or_prohibited_families_into_required_specs():
    contract = build_analysis_contract(
        target="chip_count",
        columns=["chip_count", "input_count", "gc_content"],
        refined_suggestions="""
        allowed_model_families includes linear_regression, poisson, negative_binomial, random_forest.
        不要生成 random forest 或其他预测模型。
        可选扩展或未来敏感性分析可以讨论 random forest，但本轮不执行。
        如果采用 A/B/C 中任一形式，需要解释模型假设。
        """,
        task_type="association_inference",
    )

    assert contract["required_model_specs"] == []
    assert contract["required_model_specs_source"] == "none"
    assert contract["hard_constraints"]["prohibited_model_families"] == ["random_forest"]


def test_modeling_contract_hard_user_constraints_block_stale_model_suggestion_fallback():
    contract = build_analysis_contract(
        target="chip_count",
        columns=["chip_count", "input_count", "gc_content"],
        user_input="主结果只生成候选区段表；不执行随机森林或其他额外模型。",
        model_suggestion=(
            "Fit a Poisson model and a random forest model "
            "as sensitivity models."
        ),
        refined_suggestions=(
            "最终方案只生成候选区段表；不要生成 random forest，"
            "不执行其他额外模型。"
        ),
        task_type="association_inference",
    )

    assert contract["required_model_specs"] == []
    assert contract["primary_result_type"] == "candidate_results"
    assert contract["hard_constraints"]["prohibited_model_families"] == ["random_forest"]


def test_modeling_contract_still_extracts_explicit_zero_inflated_requirement():
    contract = build_analysis_contract(
        target="chip_count",
        columns=["chip_count", "input_count", "gc_content"],
        refined_suggestions="明确拟合 zero-inflated negative binomial 模型解释 chip_count。",
        task_type="association_inference",
    )

    assert [spec["family"] for spec in contract["required_model_specs"]] == [
        "zero_inflated_negative_binomial"
    ]
    assert contract["required_model_specs"][0]["requires_zero_inflation"] is True
    assert contract["required_model_specs_source"] == "natural_language_explicit"


def test_modeling_contract_merges_repeated_mentions_of_the_same_auxiliary_model():
    refined = """
    执行构造一个简单、可解释的辅助 Poisson 回归模型。
    执行在 Poisson 回归中以 chip_count 为因变量。
    执行在 Poisson 回归中保留 matched background 解释项 input_count。
    执行在 Poisson 回归中纳入 gc_content、mappability 和 sequence_ambiguity_score。
    执行使用 model_default 作为 Poisson 回归的标准误。
    执行评估 Poisson 回归的对数似然、AIC 和残差诊断。
    辅助模型最多只能有 1 个，result_dict.models 最多包含 1 个实际拟合的辅助 Poisson 模型。
    """
    contract = build_analysis_contract(
        target="chip_count",
        columns=[
            "chip_count",
            "input_count",
            "gc_content",
            "mappability",
            "sequence_ambiguity_score",
        ],
        refined_suggestions=refined,
        task_type="association_inference",
    )

    assert contract["max_models"] == 1
    assert [spec["family"] for spec in contract["required_model_specs"]] == ["poisson"]
    assert contract["required_model_specs"][0]["id"] == "model_01"
    assert "model_02" not in json.dumps(contract["required_model_specs"], ensure_ascii=False)
    assert "AIC" in contract["required_model_specs"][0]["source"]


def test_modeling_contract_max_one_model_compresses_multiple_families_and_schema_enforces_it():
    contract = build_analysis_contract(
        target="chip_count",
        columns=["chip_count", "input_count", "gc_content"],
        refined_suggestions=(
            "最多一个辅助模型。明确拟合 Poisson 回归。\n"
            "明确拟合 Negative Binomial 回归。"
        ),
        task_type="association_inference",
    )

    assert contract["max_models"] == 1
    assert len(contract["required_model_specs"]) == 1
    assert contract["contract_warnings"]

    result = {
        "analysis_manifest": {
            "task_type": "association_inference",
            "outcome": "chip_count",
            "sample_rule": "model_specific",
            "split_strategy": "none",
            "covariance": "model_default",
            "model_structure": "as_requested",
        },
        "analysis_tables": {
            "candidate_segments": {
                "row_count": 1,
                "columns": ["start_bp", "end_bp", "score"],
                "preview": [{"start_bp": 1, "end_bp": 50, "score": 2.0}],
            }
        },
        "models": [
            {
                "name": "auxiliary_poisson",
                "n_obs": 100,
                "metrics": {"aic": 10.0},
                "model_spec": {"family": "poisson", "outcome": "chip_count", "features": ["input_count"]},
                "coefficients": [{"term": "input_count", "estimate": 1.0, "std_error": 0.1, "ci_lower": 0.8, "ci_upper": 1.2, "p_value": 0.01}],
            },
            {
                "name": "auxiliary_negative_binomial",
                "n_obs": 100,
                "metrics": {"aic": 9.0},
                "model_spec": {"family": "negative_binomial", "outcome": "chip_count", "features": ["input_count"]},
                "coefficients": [{"term": "input_count", "estimate": 1.0, "std_error": 0.1, "ci_lower": 0.8, "ci_upper": 1.2, "p_value": 0.01}],
            },
        ],
    }
    issues = validate_result_schema(result_json=result, contract=contract)

    assert any("at most 1 model" in issue for issue in issues)


def test_association_inference_contract_statically_rejects_train_test_split():
    contract = build_analysis_contract(
        target="chip_count",
        columns=[
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

    issues = validate_code_against_contract(
        code=(
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
            "best_model_b64 = 'abc'\n"
        ),
        contract=contract,
    )

    assert "train_test_split is prohibited by the analysis contract." in issues
    assert any("Mandatory model export" in issue for issue in issues)


def test_association_inference_contract_ignores_prohibited_names_in_strings_and_comments():
    contract = build_analysis_contract(
        target="chip_count",
        columns=["chip_count", "input_count", "gc_content"],
        refined_suggestions="Use OLS linear regression as an auxiliary association model.",
        task_type="association_inference",
    )
    code = """
# The manifest records that train_test_split is prohibited.
result_dict = {
    'analysis_manifest': {
        'task_type': 'association_inference',
        'outcome': 'chip_count',
        'sample_rule': 'model_specific',
        'split_strategy': 'none',
        'covariance': 'model_default',
        'model_structure': 'as_requested',
        'prohibited_operations': ['train_test_split'],
    },
    'models': [],
}
"""
    assert validate_code_against_contract(code=code, contract=contract) == []


def test_association_inference_model_spec_must_describe_real_auxiliary_model_features():
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
    result = {
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
                "metrics": {"r_squared": 0.2},
                "model_spec": {
                    "family": "linear_regression",
                    "outcome": "candidate_region",
                    "features": ["bin_id", "start_bp"],
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

    issues = validate_modeling_result(
        code="model_name = 'OLS auxiliary association model'",
        result_json=result,
        contract=contract,
    )

    assert any("model_spec.outcome must match" in issue for issue in issues)
    assert any("genomic window identifier/location columns" in issue for issue in issues)

    result["models"][0]["model_spec"] = {
        "family": "linear_regression",
        "outcome": "chip_count",
        "features": [
            "input_count",
            "gc_content",
            "mappability",
            "sequence_ambiguity_score",
        ],
    }

    assert validate_modeling_result(
        code="model_name = 'OLS auxiliary association model'",
        result_json=result,
        contract=contract,
    ) == []


def test_association_inference_rejects_placeholder_models_and_status_metrics():
    contract = build_analysis_contract(
        target="chip_count",
        columns=["chip_count", "input_count", "gc_content"],
        refined_suggestions="Use random forest for this analysis.",
        task_type="association_inference",
    )
    result = {
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
                "name": "model_02_placeholder_random_forest",
                "n_obs": 1,
                "metrics": {"status_code": 0.0},
                "model_spec": {"family": "random_forest", "outcome": "chip_count", "features": ["input_count"]},
                "coefficients": [
                    {
                        "term": "placeholder",
                        "estimate": 0.0,
                        "std_error": 0.0,
                        "ci_lower": 0.0,
                        "ci_upper": 0.0,
                        "p_value": 1.0,
                    }
                ],
            }
        ],
    }
    issues = validate_modeling_result(
        code="from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor()",
        result_json=result,
        contract=contract,
    )
    assert any("placeholder model" in issue for issue in issues)
    assert any("status values" in issue for issue in issues)
    assert any("too small" in issue for issue in issues)
    assert any("not a coefficient-based auxiliary inference model" in issue for issue in issues)


def test_association_inference_can_use_primary_candidate_tables_without_fake_models():
    contract = build_analysis_contract(
        target="chip_count",
        columns=["chip_count", "input_count", "gc_content"],
        task_type="association_inference",
    )
    result = {
        "analysis_manifest": {
            "task_type": "association_inference",
            "outcome": "chip_count",
            "sample_rule": "model_specific",
            "split_strategy": "none",
            "covariance": "model_default",
            "model_structure": "as_requested",
        },
        "analysis_tables": {
            "candidate_segments": {
                "row_count": 100,
                "columns": ["chromosome", "start_bp", "end_bp", "max_enrichment_score"],
                "preview": [{"chromosome": "chr21", "start_bp": 100, "end_bp": 250, "max_enrichment_score": 2.5}],
            }
        },
        "models": [],
    }
    assert validate_result_schema(result_json=result, contract=contract) == []


def test_modeling_result_validator_rejects_large_embedded_record_tables():
    contract = build_analysis_contract(
        target="chip_count",
        columns=["chip_count", "input_count", "gc_content"],
        task_type="association_inference",
    )
    too_many_records = [
        {"bin_id": f"chr21:{index}-{index + 49}", "score": float(index)}
        for index in range(contract["max_embedded_rows"] + 1)
    ]
    result = {
        "analysis_manifest": {
            "task_type": "association_inference",
            "outcome": "chip_count",
            "sample_rule": "model_specific",
            "split_strategy": "none",
            "covariance": "model_default",
            "model_structure": "as_requested",
        },
        "analysis_tables": {"candidate_windows": too_many_records},
        "models": [],
    }
    issues = validate_modeling_result(code="", result_json=result, contract=contract)
    assert any("embeds" in issue and "row records" in issue for issue in issues)


def test_large_modeling_code_rejects_full_dataframe_records_export_before_execution():
    code = """
work_df = df.copy()
window_level_results = work_df.to_dict(orient='records')
result_dict = {'models': [], 'analysis_tables': {'window_level_results': window_level_results}}
"""
    issues = validate_modeling_runtime_compatibility(
        code,
        n_rows=LARGE_DATASET_ROW_THRESHOLD + 1,
    )
    assert any("to_dict(orient='records')" in issue for issue in issues)


def test_modeling_worker_summarizes_large_record_lists_before_transport():
    source = pd.DataFrame({"x": [1, 2, 3]})
    code = """
records = [{'row_id': int(i), 'score': float(i)} for i in range(1005)]
result_dict = {
    'analysis_manifest': {'task_type': 'association_inference'},
    'analysis_tables': {'candidate_windows': records},
    'models': [],
}
"""
    result = run_bounded_safe_exec(
        kind="modeling",
        code=code,
        dataframe=source,
        timeout_seconds=10,
    )
    assert result["is_success"] is True
    candidate_windows = result["value"]["analysis_tables"]["candidate_windows"]
    assert candidate_windows["row_count"] == 1005
    assert len(candidate_windows["preview"]) == 20
    assert candidate_windows["omitted_count"] == 985
