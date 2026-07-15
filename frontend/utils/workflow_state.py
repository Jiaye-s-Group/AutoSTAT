from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, MutableMapping
from typing import Any

import pandas as pd


STAGE_DESCENDANTS = {
    "planning": {"loading", "preprocessing", "visualization", "modeling", "report"},
    "loading": {"report"},
    "preprocessing": {"visualization", "modeling", "report"},
    "visualization": {"report"},
    "modeling": {"report"},
    "report": set(),
}

CHANGE_ROOTS = {
    "dataset": {"planning", "loading", "preprocessing", "visualization", "modeling", "report"},
    "references": {"planning", "loading", "preprocessing", "visualization", "modeling", "report"},
    "preferences": {"planning", "loading", "preprocessing", "visualization", "modeling", "report"},
}

STAGE_ARTIFACT_KEYS = {
    "planning": ("planning_workflow_result",),
    "loading": (
        "loading_workflow_result",
        "summary_1",
        "abstract_1",
        "summary_1_title",
        "summary_1_desc",
        "summary_1_df",
    ),
    "preprocessing": (
        "suggestion",
        "abstract_2",
        "summary_2",
        "prep_result_from_summary_2",
        "prep_code_visible",
        "preprocessing_failure",
        "_prep_phase2_requested",
        "_prep_code_repair_requested",
    ),
    "visualization": (
        "abstract_3",
        "summary_3",
        "visual_recommendatio",
        "viz_suggestion",
        "viz_workflow_result",
        "final_code",
        "full",
        "tu_title",
        "visualization_failure",
        "_viz_phase2_pending",
        "_viz_phase2_requested",
        "_viz_code_repair_requested",
        "_viz_phase1_ctx",
        "_viz_phase2_inputs",
        "viz_current_page",
        "viz_pagination",
        "viz_download_image_cache",
    ),
    "modeling": (
        "abstract_4",
        "summary_4",
        "model_suggestion",
        "modeling_suggestion",
        "modeling_summary_4",
        "modeling_abstract_4",
        "modeling_result_from_summary_4",
        "modeling_workflow_result",
        "modeling_analysis_contract",
        "modeling_failure",
        "_model_phase2_pending",
        "_model_phase2_requested",
        "_model_code_repair_requested",
        "_model_phase1_ctx",
        "_model_phase2_inputs",
        "history_train_code_input",
        "history_train_code_reset_pending",
    ),
    "report": (
        "report_title",
        "report_workflow_outputs",
        "report_toc_text",
        "report_display_outline",
        "report_display_to_internal_toc_map",
        "report_load_abstract",
        "report_preproc_abstract",
        "report_visual_abstract",
        "report_coding_abstract",
        "report_selected_full_conten",
        "report_final_html",
        "report_generation_job",
        "report_generation_running",
    ),
}

STAGE_AGENT_RESETS = {
    "loading": ("data_loading_agent", ("loading_workflow_result", "memory", "finish_auto_task")),
    "preprocessing": (
        "data_preprocess_agent",
        (
            "preprocessing_suggestions",
            "code",
            "processed_df",
            "user_input",
            "error",
            "memory",
            "finish_auto_task",
        ),
    ),
    "visualization": (
        "visualization_agent",
        ("code", "suggestion", "user_input", "error", "memory", "fig", "fig_desc_list", "finish_auto_task"),
    ),
    "modeling": (
        "modeling_coding_agent",
        (
            "code",
            "suggestion",
            "modeling_result",
            "error",
            "memory",
            "finish_auto_task",
        ),
    ),
    "report": (
        "report_agent",
        ("report_content", "report", "report_workflow_result", "outline", "word", "html", "markdown", "pdf", "finish_auto_task"),
    ),
}


def stable_fingerprint(*values: Any) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(stable_fingerprint(list(map(str, df.columns)), list(map(str, df.dtypes)), df.shape).encode())
    try:
        digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    except Exception:
        digest.update(df.to_json(orient="split", date_format="iso", default_handler=str).encode("utf-8"))
    return digest.hexdigest()


def current_dataset_fingerprint(state: Mapping[str, Any]) -> str:
    return str(state.get("dataset_fingerprint") or "")


def stage_input_fingerprint(state: Mapping[str, Any], *extra: Any) -> str:
    return stable_fingerprint(current_dataset_fingerprint(state), *extra)


def _stage_states(state: MutableMapping[str, Any]) -> dict[str, dict[str, Any]]:
    stages = state.get("workflow_stage_states")
    if not isinstance(stages, dict):
        stages = {}
        state["workflow_stage_states"] = stages
    return stages


def record_stage_status(
    state: MutableMapping[str, Any],
    stage: str,
    status: str,
    *,
    input_fingerprint: str = "",
    output_fingerprint: str = "",
    error: str = "",
) -> None:
    _stage_states(state)[stage] = {
        "status": status,
        "input_fingerprint": input_fingerprint,
        "output_fingerprint": output_fingerprint,
        "error": error,
    }


def stage_is_current(
    state: Mapping[str, Any],
    stage: str,
    *,
    input_fingerprint: str | None = None,
) -> bool:
    stages = state.get("workflow_stage_states")
    if not isinstance(stages, dict):
        return False
    item = stages.get(stage)
    if not isinstance(item, dict) or item.get("status") != "succeeded":
        return False
    if input_fingerprint is not None and item.get("input_fingerprint") != input_fingerprint:
        return False
    return True


def _reset_agent(agent: Any, attrs: tuple[str, ...]) -> None:
    if agent is None:
        return
    for attr in attrs:
        if not hasattr(agent, attr):
            continue
        if attr in {"memory", "fig_desc_list"}:
            value: Any = []
        elif attr == "finish_auto_task":
            value = False
        else:
            value = None
        setattr(agent, attr, value)


def _collect_descendants(stage: str) -> set[str]:
    pending = list(STAGE_DESCENDANTS.get(stage, ()))
    collected: set[str] = set()
    while pending:
        item = pending.pop()
        if item in collected:
            continue
        collected.add(item)
        pending.extend(STAGE_DESCENDANTS.get(item, ()))
    return collected


def invalidate_stages(state: MutableMapping[str, Any], stages: set[str], *, reason: str) -> set[str]:
    stage_states = _stage_states(state)
    for stage in stages:
        previous = stage_states.get(stage)
        stage_states[stage] = {
            "status": "stale",
            "input_fingerprint": previous.get("input_fingerprint", "") if isinstance(previous, dict) else "",
            "output_fingerprint": previous.get("output_fingerprint", "") if isinstance(previous, dict) else "",
            "error": reason,
        }
        for key in STAGE_ARTIFACT_KEYS.get(stage, ()):
            state.pop(key, None)
        agent_key, attrs = STAGE_AGENT_RESETS.get(stage, (None, ()))
        if agent_key:
            _reset_agent(state.get(agent_key), attrs)
    return stages


def invalidate_from(
    state: MutableMapping[str, Any],
    source: str,
    *,
    include_source: bool = False,
    reason: str = "upstream input changed",
) -> set[str]:
    if source in CHANGE_ROOTS:
        state.pop("_suggestion_interactions", None)
        stages = set(CHANGE_ROOTS[source])
        state["analysis_dataset_fingerprint"] = current_dataset_fingerprint(state)
        if state.get("auto_mode"):
            state["auto_mode"] = False
            planner = state.get("planner_agent")
            stop_auto = getattr(planner, "stop_auto", None)
            if callable(stop_auto):
                stop_auto()
    else:
        stages = _collect_descendants(source)
        if include_source:
            stages.add(source)
            if source == "preprocessing":
                state["analysis_dataset_fingerprint"] = current_dataset_fingerprint(state)
    return invalidate_stages(state, stages, reason=reason)


def commit_dataset_fingerprint(state: MutableMapping[str, Any], fingerprint: str) -> bool:
    previous = current_dataset_fingerprint(state)
    changed = previous != fingerprint
    if changed:
        invalidate_from(state, "dataset", reason="dataset snapshot changed")
        state["dataset_fingerprint"] = fingerprint
        state["analysis_dataset_fingerprint"] = fingerprint
    return changed
