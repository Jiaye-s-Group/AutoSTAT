"""Build section-writing reference context from existing stage outputs."""
from __future__ import annotations

import json
import re
from typing import Any


NATURAL_FIGURE_REFERENCE_RE = re.compile(
    r"(?:(?:另|并)?(?:如|见|参见|详见|参考|根据|结合|从)\s*)?"
    r"(?:图表|图片|插图|图|Figure|Fig\.)\s*"
    r"(?:第\s*)?(?:\d+|[一二三四五六七八九十]+)"
    r"(?:\s*(?:所示|可见|可以看出|显示|展示|中|里))?\s*[，,、:：；;]?",
    flags=re.IGNORECASE,
)


def _compact(value: Any, max_chars: int = 1600) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and not value:
        return ""
    if isinstance(value, (list, tuple, set, dict)) and not value:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n...[truncated {len(text) - max_chars} chars]"


def _format_block(title: str, fields: list[tuple[str, Any, int]]) -> str:
    lines = [f"[{title}]"]
    for label, value, max_chars in fields:
        text = _compact(value, max_chars)
        if text:
            lines.append(f"{label}:\n{text}")
    return "\n\n".join(lines).strip()


def _strip_natural_figure_references(value: Any) -> str:
    text = _compact(value, 100000)
    if not text:
        return ""
    text = NATURAL_FIGURE_REFERENCE_RE.sub("", text)
    text = re.sub(r"\s+([，,。.!！?？；;：:、])", r"\1", text)
    text = re.sub(r"[，,、；;：:]\s*([。.!！?？])", r"\1", text)
    return text.strip()


def _dtype_counts_from_info(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        counts: dict[str, int] = {}
        for dtype in value.values():
            key = str(dtype)
            counts[key] = counts.get(key, 0) + 1
        return counts

    text = _compact(value, 100000)
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        return _dtype_counts_from_info(parsed)

    counts: dict[str, int] = {}
    dtype_pattern = re.compile(
        r"(?:^|[\s:,\"])(bool|object|category|string|int\d*|float\d*|datetime64[^\s,\"]*|timedelta64[^\s,\"]*)\s*$",
        flags=re.IGNORECASE,
    )
    for line in text.splitlines():
        match = dtype_pattern.search(line.strip())
        if not match:
            continue
        key = match.group(1)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _structured_data_facts(plan: dict[str, Any]) -> str:
    facts: dict[str, Any] = {}
    if plan.get("shape_0") is not None:
        facts["rows"] = plan.get("shape_0")
    if plan.get("shape_1") is not None:
        facts["columns"] = plan.get("shape_1")
    dtype_counts = plan.get("dtype_counts")
    if not isinstance(dtype_counts, dict):
        dtype_counts = _dtype_counts_from_info(plan.get("dtype_info_str"))
    if dtype_counts:
        facts["dtype_counts"] = dtype_counts
    if not facts:
        return ""
    return json.dumps(facts, ensure_ascii=False, default=str)


def _records_shape(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and not value:
        return ""
    if isinstance(value, (list, tuple, set, dict)) and not value:
        return ""
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except Exception:
        return ""
    if isinstance(parsed, list):
        rows = len(parsed)
        columns: set[str] = set()
        for row in parsed:
            if isinstance(row, dict):
                columns.update(str(key) for key in row)
        if columns:
            return f"{rows} preview records x {len(columns)} columns"
    if isinstance(parsed, dict):
        return f"{len(parsed)} fields in preview object"
    return ""


def _figure_contexts(summary_3: dict[str, Any], full: Any, abstract: str) -> str:
    fig_analysis = summary_3.get("fig_analysis") if isinstance(summary_3, dict) else None
    figure_lines: list[str] = []
    if isinstance(fig_analysis, list):
        for index, item in enumerate(fig_analysis):
            if not isinstance(item, dict):
                continue
            title = _compact(_strip_natural_figure_references(item.get("title")), 180)
            desc = _compact(_strip_natural_figure_references(item.get("desc")), 400)
            analysis = _compact(_strip_natural_figure_references(item.get("analysis")), 900)
            parts = [f"Figure index: [FIG:{index}]"]
            if title:
                parts.append(f"Title: {title}")
            if desc:
                parts.append(f"Description/statistical basis: {desc}")
            if analysis:
                parts.append(f"Figure analysis: {analysis}")
            figure_lines.append("\n".join(parts))

    return _format_block(
        "Visualization Reference Context",
        [
            ("Figure reference rule", "Use only [FIG:index] placeholders in report text; final display numbers are assigned during rendering.", 220),
            ("Figure-level records", "\n\n".join(figure_lines), 5000),
            ("Visualization full context", _strip_natural_figure_references(full), 1800),
            ("Stage abstract", _strip_natural_figure_references(abstract), 900),
        ],
    )


def build_stage_reference_contexts(
    *,
    plan: dict[str, Any],
    loading: dict[str, Any],
    prep: dict[str, Any],
    viz: dict[str, Any],
    model: dict[str, Any],
    next_cols: list[str] | None = None,
    next_head: Any = "",
) -> dict[str, str]:
    """Return report-facing context for section writing.

    The context is derived from existing stage outputs. It is not a new artifact
    family; it is a compact view used by the Report Agent when writing a section.
    """
    summary_1 = loading.get("summary_1") if isinstance(loading, dict) else {}
    summary_2 = prep.get("summary_2") if isinstance(prep, dict) else {}
    summary_3 = viz.get("summary_3") if isinstance(viz, dict) else {}
    summary_4 = model.get("summary_4") if isinstance(model, dict) else {}
    summary_1 = summary_1 if isinstance(summary_1, dict) else {}
    summary_2 = summary_2 if isinstance(summary_2, dict) else {}
    summary_3 = summary_3 if isinstance(summary_3, dict) else {}
    summary_4 = summary_4 if isinstance(summary_4, dict) else {}

    processed_shape = _records_shape(summary_2.get("processed_df"))
    loading_context = _format_block(
        "Data Profiling Reference Context",
        [
            ("Structured data facts", _structured_data_facts(plan), 600),
            ("Raw data shape", f"{plan.get('shape_0')} rows x {plan.get('shape_1')} columns", 120),
            ("Full-table metadata", plan.get("data_profile_str") or loading.get("_data_profile_str"), 2600),
            ("Column names and dtypes", plan.get("dtype_info_str"), 2200),
            ("Raw data preview", plan.get("head_dict_str") or summary_1.get("df"), 1800),
            ("Stage summary", summary_1.get("desc"), 1200),
            ("Stage abstract", loading.get("abstract_1", "") if isinstance(loading, dict) else "", 900),
        ],
    )

    preprocessing_context = _format_block(
        "Preprocessing Reference Context",
        [
            ("Executed preprocessing source", summary_2.get("code"), 1800),
            ("Processed data shape", processed_shape, 120),
            ("Processed data preview", summary_2.get("processed_df") or next_head, 1800),
            ("Current columns after preprocessing", next_cols or "", 1600),
            ("Stage summary", summary_2.get("desc"), 1200),
            ("Stage abstract", prep.get("abstract_2", "") if isinstance(prep, dict) else "", 900),
        ],
    )

    visualization_context = _figure_contexts(
        summary_3,
        viz.get("full", "") if isinstance(viz, dict) else "",
        viz.get("abstract_3", "") if isinstance(viz, dict) else "",
    )

    table_context = "\n\n".join(
        part
        for part in [
            _compact(summary_4.get("table_title"), 200),
            _compact(summary_4.get("table_markdown"), 2400),
            _compact(summary_4.get("table_html"), 1800),
        ]
        if part
    )
    modeling_report_artifacts = {}
    if isinstance(model, dict):
        raw_report_artifacts = model.get("_modeling_report_artifacts")
        modeling_report_artifacts = raw_report_artifacts if isinstance(raw_report_artifacts, dict) else {}
    if not modeling_report_artifacts:
        raw_report_artifacts = summary_4.get("report_artifacts")
        modeling_report_artifacts = raw_report_artifacts if isinstance(raw_report_artifacts, dict) else {}
    modeling_evidence = {}
    if isinstance(model, dict):
        raw_evidence = model.get("_modeling_result_evidence")
        modeling_evidence = raw_evidence if isinstance(raw_evidence, dict) else {}
    modeling_context = _format_block(
        "Modeling Reference Context",
        [
            ("Structured modeling report artifacts", modeling_report_artifacts, 5000),
            ("Compact modeling execution evidence", modeling_evidence, 2600),
            ("Modeling metric copy source", table_context, 3600),
            ("Metric copy rule", "Model names, best-model claims, rankings, and metric values in prose must be copied from the modeling table. If a value is not present in the table or execution output, omit it.", 500),
            ("Modeling execution results", summary_4.get("result"), 2200),
            ("Modeling task and interpretation", summary_4.get("desc"), 1500),
            ("Stage abstract", model.get("abstract_4", "") if isinstance(model, dict) else "", 1000),
        ],
    )

    return {
        "loading": loading_context,
        "preprocessing": preprocessing_context,
        "visualization": visualization_context,
        "modeling": modeling_context,
    }
