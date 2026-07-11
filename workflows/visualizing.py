"""
Visualizing workflow local implementation.
"""

from __future__ import annotations

import ast
import base64
import json
import math
import os
import re
import signal
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from core.llm_client import chat, chat_multimodal
from core.prompt_template import render_file
from core.workflow_runner import to_str
from workflows._plugins import (
    desc_fig_prompt,
    execute_and_extract,
    sec3_check_full,
    sec3_composer,
    summary_fig_list_prompt,
    validate_viz_code,
)

MAX_FIX_ATTEMPTS = 5
BATCH_CONCURRENCY = 10
MAX_VIS_TITLE_CHARS = 20
MAX_PROMPT_DATA_CHARS = int(os.getenv("AUTOSTAT_VIZ_PROMPT_DATA_CHARS", "20000"))
MAX_PROMPT_HEAD_CHARS = int(os.getenv("AUTOSTAT_VIZ_PROMPT_HEAD_CHARS", "12000"))
MAX_PROMPT_CONTEXT_CHARS = int(os.getenv("AUTOSTAT_VIZ_PROMPT_CONTEXT_CHARS", "12000"))
MAX_PROMPT_USER_TEXT_CHARS = int(os.getenv("AUTOSTAT_VIZ_PROMPT_USER_TEXT_CHARS", "6000"))
FIG_ARTIFACT_MAX_TRACES = 8
FIG_ARTIFACT_MAX_SAMPLE_VALUES = 8
FIG_ARTIFACT_MAX_CATEGORIES = 12
FIG_IMAGE_WIDTH = int(os.getenv("AUTOSTAT_FIG_IMAGE_WIDTH", "1100"))
FIG_IMAGE_HEIGHT = int(os.getenv("AUTOSTAT_FIG_IMAGE_HEIGHT", "720"))
FIG_IMAGE_TIMEOUT_SECONDS = int(os.getenv("AUTOSTAT_FIG_IMAGE_TIMEOUT_SECONDS", "20"))
USE_FIGURE_IMAGES = os.getenv("AUTOSTAT_USE_FIGURE_IMAGES", "1").strip().lower() not in {"0", "false", "no", "off"}
DISABLE_FIGURE_IMAGES_AFTER_TIMEOUT = (
    os.getenv("AUTOSTAT_DISABLE_FIGURE_IMAGES_AFTER_TIMEOUT", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)
_FIGURE_IMAGE_RENDER_DISABLED = False
_PLOTLY_IMAGE_RENDER_SCRIPT = r"""
import base64
import sys

import plotly.io as pio

raw_text = sys.stdin.read()
width = int(sys.argv[1])
height = int(sys.argv[2])
fig_obj = pio.from_json(raw_text)
image_bytes = fig_obj.to_image(
    format="jpg",
    width=width,
    height=height,
    scale=1,
)
sys.stdout.write(base64.b64encode(image_bytes).decode("ascii"))
"""
VAGUE_VIS_TITLES = {
    "变量关系与分布特征",
    "变量关系",
    "分布特征",
    "比较关系",
    "变化趋势",
    "关系",
    "分布",
    "比较",
    "趋势",
    "数据可视化",
    "可视化结果",
    "结果图",
    "比较图",
    "分析图",
    "数据展示",
    "模型表现",
}


def _truncate_prompt_text(value: Any, max_chars: int) -> str:
    text = to_str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated for visualization prompt]"


def _sanitize_visualization_code(code: str) -> str:
    code = _unwrap_code_block(code)
    if not code:
        return ""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    class _DropDfReassign(ast.NodeTransformer):
        @staticmethod
        def _is_df_name(target):
            return isinstance(target, ast.Name) and target.id == "df"

        def visit_Assign(self, node):
            if any(self._is_df_name(target) for target in node.targets):
                return None
            return self.generic_visit(node)

        def visit_AnnAssign(self, node):
            if self._is_df_name(node.target):
                return None
            return self.generic_visit(node)

        def visit_AugAssign(self, node):
            if self._is_df_name(node.target):
                return None
            return self.generic_visit(node)

        def visit_NamedExpr(self, node):
            if self._is_df_name(node.target):
                return None
            return self.generic_visit(node)

    sanitized_tree = _DropDfReassign().visit(tree)
    ast.fix_missing_locations(sanitized_tree)
    try:
        return ast.unparse(sanitized_tree).strip()
    except Exception:
        return code


def _build_ctx(
    *,
    data: str,
    shape0: int,
    shape1: int,
    cols: list,
    def_head: str,
    color: str = "",
    user_input: str = "",
    add_preference: str = "",
    preference_selected: str = "",
    ref_context: str = "",
) -> dict[str, Any]:
    """Build shared prompt context for visualization steps."""
    return {
        "data": _truncate_prompt_text(data, MAX_PROMPT_DATA_CHARS),
        "shape_0": shape0,
        "shape_1": shape1,
        "shape0": shape0,
        "shape1": shape1,
        "cols": cols,
        "def_head": _truncate_prompt_text(def_head, MAX_PROMPT_HEAD_CHARS),
        "df_head": _truncate_prompt_text(def_head, MAX_PROMPT_HEAD_CHARS),
        "color": color or "",
        "user_input": _truncate_prompt_text(user_input, MAX_PROMPT_USER_TEXT_CHARS),
        "add_preference": _truncate_prompt_text(add_preference, MAX_PROMPT_USER_TEXT_CHARS),
        "preference_selected": _truncate_prompt_text(preference_selected, MAX_PROMPT_USER_TEXT_CHARS),
        "ref_context": _truncate_prompt_text(ref_context or "（无参考资料）", MAX_PROMPT_CONTEXT_CHARS),
    }


def run_visualizing_phase1(
    *,
    data: str,
    shape0: int,
    shape1: int,
    cols: list,
    def_head: str,
    vis_auto: bool = True,
    color: str = "",
    user_input: str = "",
    add_preference: str = "",
    preference_selected: str = "",
    ref_context: str = "",
) -> dict[str, Any]:
    """Generate visualization recommendations and implementation requirements."""
    if not vis_auto:
        return {"visual_recommendatio": "", "refined_suggestions": "", "_ctx": {}}

    ctx = _build_ctx(
        data=data, shape0=shape0, shape1=shape1, cols=cols,
        def_head=def_head, color=color, user_input=user_input,
        add_preference=add_preference, preference_selected=preference_selected,
        ref_context=ref_context,
    )

    vr_sys = render_file("visualizing/sec3_get_visual_recommendation_llm_sys.txt", ctx)
    vr_user = render_file("visualizing/sec3_get_visual_recommendation_llm_user.txt", ctx)
    visual_recommendatio = chat(
        vr_sys, vr_user, name="viz.get_visual_recommendation"
    ).strip()
    ctx["visual_recommendation"] = _truncate_prompt_text(
        visual_recommendatio,
        MAX_PROMPT_CONTEXT_CHARS,
    )
    ctx["visual_recommendatio"] = ctx["visual_recommendation"]

    rs_sys = render_file("visualizing/sec3_refine_suggestions_llm_sys.txt", ctx)
    rs_user = render_file("visualizing/sec3_refine_suggestions_llm_user.txt", ctx)
    refined_suggestions = chat(rs_sys, rs_user, name="viz.refine_suggestions").strip()
    ctx["refined_suggestions"] = _truncate_prompt_text(
        refined_suggestions,
        MAX_PROMPT_CONTEXT_CHARS,
    )

    return {
        "visual_recommendatio": visual_recommendatio,
        "refined_suggestions": refined_suggestions,
        "_ctx": ctx,
    }


def run_visualizing_phase2(
    *,
    ctx: dict[str, Any],
    data: str,
    cols: list,
    def_head: str,
) -> dict[str, Any]:
    """Generate, validate, repair, execute, and summarize visualization code."""
    visual_recommendatio = ctx.get("visual_recommendatio", "")

    cg_sys = render_file("visualizing/sec3_code_generation_llm_sys.txt", ctx)
    cg_user = render_file("visualizing/sec3_code_generation_llm_user.txt", ctx)
    generation_code = chat(cg_sys, cg_user, name="viz.code_generation").strip()
    generation_code = _sanitize_visualization_code(generation_code)

    current_code = generation_code
    success = False
    last_error = ""
    for attempt in range(MAX_FIX_ATTEMPTS):
        validate = validate_viz_code(code=current_code, df_data=def_head)
        if validate["is_success"]:
            current_code = validate["final_code"]
            success = True
            break

        last_error = validate.get("error_msg", "")
        fix_ctx = {
            **ctx,
            "code": current_code,
            "code_vis": current_code,
            "error_msg": last_error,
            "error": last_error,
        }
        fix_sys = render_file("visualizing/sec3_fixed_code_llm_sys.txt", fix_ctx)
        fix_user = render_file("visualizing/sec3_fixed_code_llm_user.txt", fix_ctx)
        fixed = chat(
            fix_sys,
            fix_user,
            name=f"viz.fixed_code.{attempt + 1}",
            temperature=0.3,
        ).strip()
        fixed = _sanitize_visualization_code(fixed)
        if fixed:
            current_code = fixed

    final_code = current_code
    if not success:
        return {
            "full": "",
            "abstract_3": f"可视化代码生成失败：{last_error[:500]}",
            "summary_3": {"title": "数据可视化", "fig_analysis": []},
            "visual_recommendatio": visual_recommendatio,
            "final_code": final_code,
            "tu_title": [],
        }

    ctx["final_code"] = final_code
    exec_result = execute_and_extract(code=final_code, df_data=data)
    fig_task_list = exec_result.get("fig_task_list", [])
    fig_task_list = _attach_figure_llm_artifacts(fig_task_list)
    if not fig_task_list:
        return {
            "full": "",
            "abstract_3": "未能从代码中提取到任何图表。",
            "summary_3": {"title": "数据可视化", "fig_analysis": []},
            "visual_recommendatio": visual_recommendatio,
            "final_code": final_code,
            "tu_title": [],
        }

    cols_wo_id = _filter_id_columns(cols)
    dtype_info = to_str(def_head)

    # Each figure can generate its title and summary independently.
    def _process_single_fig(item: dict) -> dict[str, Any]:
        pack = _desc_fig_single(item, dtype_info, ctx)
        with ThreadPoolExecutor(max_workers=2) as inner_pool:
            title_fut = inner_pool.submit(
                _generate_title_single, pack, cols_wo_id, ctx
            )
            summary_fut = inner_pool.submit(
                _summary_fig_single, pack, cols_wo_id, ctx
            )
            title = title_fut.result()
            summary = summary_fut.result()
        return {"title": title, "summary": summary}

    fig_results = _batch_run(
        fig_task_list,
        _process_single_fig,
        concurrency=BATCH_CONCURRENCY,
    )

    tu_title = []
    aggregate_results = []
    for r in fig_results:
        if isinstance(r, dict):
            tu_title.append(
                r.get("title", "").strip() if isinstance(r.get("title"), str) else ""
            )
            aggregate_results.append(
                r.get("summary", {}) if isinstance(r.get("summary"), dict) else {}
            )
        else:
            tu_title.append("")
            aggregate_results.append({})

    full = sec3_check_full(analysis_list=aggregate_results)["full"]

    abs_ctx = {**ctx, "all_analyses": full, "full": full}
    abs_sys = render_file("visualizing/sec3_abstract_llm_sys.txt", abs_ctx)
    abs_user = render_file("visualizing/sec3_abstract_llm_user.txt", abs_ctx)
    abstract_3 = chat(abs_sys, abs_user, name="viz.abstract").strip()

    composed = sec3_composer(fig_analysis=aggregate_results)
    return {
        "full": full,
        "abstract_3": abstract_3,
        "summary_3": composed["summary_3"],
        "visual_recommendatio": visual_recommendatio,
        "final_code": final_code,
        "tu_title": tu_title,
    }


def run_visualizing_workflow(
    *,
    data: str,
    shape0: int,
    shape1: int,
    cols: list,
    def_head: str,
    vis_auto: bool = True,
    color: str = "",
    user_input: str = "",
    add_preference: str = "",
    preference_selected: str = "",
    ref_context: str = "",
) -> dict[str, Any]:
    """Run visualization recommendation and code-generation phases in sequence."""
    if not vis_auto:
        return _empty_result()

    p1 = run_visualizing_phase1(
        data=data, shape0=shape0, shape1=shape1, cols=cols,
        def_head=def_head, vis_auto=vis_auto, color=color,
        user_input=user_input, add_preference=add_preference,
        preference_selected=preference_selected, ref_context=ref_context,
    )
    ctx = p1["_ctx"]
    if not ctx:
        return _empty_result()

    return run_visualizing_phase2(ctx=ctx, data=data, cols=cols, def_head=def_head)


def _attach_figure_llm_artifacts(fig_task_list: Any) -> list[dict[str, Any]]:
    if not isinstance(fig_task_list, list):
        return []
    out: list[dict[str, Any]] = []
    for raw_item in fig_task_list:
        item = dict(raw_item) if isinstance(raw_item, dict) else {"fig": raw_item}
        fig = item.get("fig", "")
        artifact = compact_plotly_figure(fig)
        item["fig_artifact"] = json.dumps(artifact, ensure_ascii=False, default=str)
        item["fig_image"] = render_plotly_image_data_url(fig) if USE_FIGURE_IMAGES else ""
        out.append(item)
    return out


def compact_plotly_figure(fig: Any) -> dict[str, Any]:
    payload, raw_text = _coerce_plotly_payload(fig)
    if not isinstance(payload, dict):
        return {
            "available": False,
            "raw_json_chars": len(raw_text),
            "note": "Figure payload could not be parsed; only raw size is available.",
        }

    layout = payload.get("layout") if isinstance(payload.get("layout"), dict) else {}
    data = payload.get("data") if isinstance(payload.get("data"), list) else []
    traces = []
    total_points = 0
    chart_types: list[str] = []
    trace_names: list[str] = []

    for trace_index, trace in enumerate(data[:FIG_ARTIFACT_MAX_TRACES]):
        if not isinstance(trace, dict):
            continue
        trace_type = _clean_artifact_scalar(trace.get("type", "")) or "unknown"
        trace_name = _clean_artifact_scalar(trace.get("name", ""))
        if trace_type and trace_type not in chart_types:
            chart_types.append(trace_type)
        if trace_name and trace_name not in trace_names:
            trace_names.append(trace_name)

        dimensions: dict[str, Any] = {}
        point_count = 0
        for key in ("x", "y", "z", "labels", "values", "lat", "lon"):
            if key not in trace:
                continue
            values = trace.get(key)
            if isinstance(values, list):
                point_count = max(point_count, _count_leaf_values(values))
            dimensions[key] = _summarize_values(values)

        total_points += point_count
        traces.append(
            {
                "index": trace_index,
                "type": trace_type,
                "name": trace_name,
                "mode": _clean_artifact_scalar(trace.get("mode", "")),
                "point_count": point_count,
                "dimensions": dimensions,
            }
        )

    if len(data) > FIG_ARTIFACT_MAX_TRACES:
        traces.append(
            {
                "omitted_traces": len(data) - FIG_ARTIFACT_MAX_TRACES,
                "note": "Additional traces were omitted from the LLM artifact.",
            }
        )

    return {
        "available": True,
        "raw_json_chars": len(raw_text),
        "title": _layout_text(layout.get("title")),
        "x_axis_title": _axis_title_text(layout.get("xaxis")),
        "y_axis_title": _axis_title_text(layout.get("yaxis")),
        "legend_title": _axis_title_text(layout.get("legend")),
        "chart_types": chart_types,
        "trace_names": trace_names[:FIG_ARTIFACT_MAX_CATEGORIES],
        "trace_count": len(data),
        "total_point_count_included": total_points,
        "traces": traces,
    }


def _coerce_plotly_payload(fig: Any) -> tuple[dict[str, Any] | None, str]:
    if isinstance(fig, dict):
        return fig, json.dumps(fig, ensure_ascii=False, default=str)
    raw_text = to_str(fig)
    if not raw_text:
        return None, ""
    try:
        payload = json.loads(raw_text)
    except Exception:
        return None, raw_text
    return payload if isinstance(payload, dict) else None, raw_text


def render_plotly_image_data_url(fig: Any) -> str:
    global _FIGURE_IMAGE_RENDER_DISABLED
    if _FIGURE_IMAGE_RENDER_DISABLED or FIG_IMAGE_TIMEOUT_SECONDS <= 0:
        return ""
    payload, raw_text = _coerce_plotly_payload(fig)
    if not isinstance(payload, dict):
        return ""
    status, encoded = _render_plotly_image_base64(raw_text)
    if status == "timeout" and DISABLE_FIGURE_IMAGES_AFTER_TIMEOUT:
        _FIGURE_IMAGE_RENDER_DISABLED = True
    if not encoded:
        return ""
    return f"data:image/jpeg;base64,{encoded}"


def _render_plotly_image_base64(raw_text: str) -> tuple[str, str]:
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _PLOTLY_IMAGE_RENDER_SCRIPT,
                str(FIG_IMAGE_WIDTH),
                str(FIG_IMAGE_HEIGHT),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except Exception:
        return "error", ""

    try:
        stdout, _stderr = process.communicate(raw_text, timeout=FIG_IMAGE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        return "timeout", ""

    if process.returncode != 0:
        return "error", ""
    encoded = stdout.strip()
    return ("ok", encoded) if encoded else ("empty", "")


def _terminate_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass
    try:
        process.wait(timeout=2)
        return
    except Exception:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
    try:
        process.wait(timeout=2)
    except Exception:
        pass


def _figure_artifact_text(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    artifact = item.get("fig_artifact", "")
    if isinstance(artifact, str) and artifact.strip():
        return artifact
    if artifact:
        try:
            return json.dumps(artifact, ensure_ascii=False, default=str)
        except Exception:
            return str(artifact)
    fig = item.get("fig", "")
    return json.dumps(compact_plotly_figure(fig), ensure_ascii=False, default=str)


def _chat_with_optional_figure_image(
    sys_prompt: str,
    user_prompt: str,
    *,
    image_data_url: str = "",
    name: str,
    **kwargs: Any,
) -> str:
    if image_data_url:
        return chat_multimodal(
            sys_prompt,
            user_prompt,
            image_data_url=image_data_url,
            name=name,
            **kwargs,
        )
    return chat(sys_prompt, user_prompt, name=name, **kwargs)


def _summarize_values(values: Any) -> dict[str, Any]:
    if isinstance(values, list):
        flat = list(_iter_leaf_values(values, limit=5000))
        numeric = [_to_finite_float(value) for value in flat]
        numeric = [value for value in numeric if value is not None]
        summary: dict[str, Any] = {
            "count": _count_leaf_values(values),
            "sample": _sample_values(flat, FIG_ARTIFACT_MAX_SAMPLE_VALUES),
        }
        if numeric:
            sorted_numeric = sorted(numeric)
            summary["numeric_summary"] = {
                "min": round(sorted_numeric[0], 6),
                "max": round(sorted_numeric[-1], 6),
                "mean": round(sum(sorted_numeric) / len(sorted_numeric), 6),
                "median": round(sorted_numeric[len(sorted_numeric) // 2], 6),
            }
        else:
            categories = [
                _clean_artifact_scalar(value)
                for value in flat
                if _clean_artifact_scalar(value)
            ]
            if categories:
                summary["top_values"] = Counter(categories).most_common(FIG_ARTIFACT_MAX_CATEGORIES)
        return summary
    scalar = _clean_artifact_scalar(values)
    return {"value": scalar} if scalar else {}


def _iter_leaf_values(values: Any, *, limit: int) -> Any:
    count = 0
    stack = [values]
    while stack and count < limit:
        current = stack.pop(0)
        if isinstance(current, list):
            stack = current[:limit] + stack
            continue
        yield current
        count += 1


def _count_leaf_values(values: Any) -> int:
    if not isinstance(values, list):
        return 1
    total = 0
    stack = [values]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
        else:
            total += 1
    return total


def _sample_values(values: list[Any], max_items: int) -> list[Any]:
    if not values:
        return []
    if len(values) <= max_items:
        return [_clean_artifact_scalar(value) for value in values]
    if max_items <= 2:
        sample = values[:max_items]
    else:
        head_count = max_items // 2
        tail_count = max_items - head_count
        sample = values[:head_count] + values[-tail_count:]
    return [_clean_artifact_scalar(value) for value in sample]


def _to_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _clean_artifact_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return ""
        return round(number, 6) if isinstance(value, float) else value
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 120:
        return text[:117] + "..."
    return text


def _desc_fig_single(
    item: dict, dtype_info: str, base_ctx: dict[str, Any]
) -> dict[str, Any]:
    fig = item.get("fig", "") if isinstance(item, dict) else ""
    raw_title = item.get("title", "") if isinstance(item, dict) else ""
    fig_artifact_text = _figure_artifact_text(item)
    fig_image = item.get("fig_image", "") if isinstance(item, dict) else ""

    p_out = desc_fig_prompt(dtype_info=dtype_info, fig="", fig_artifact=fig_artifact_text)
    prompt_content = p_out.get("prompt_content", "")

    desc_ctx = {
        **base_ctx,
        "prompt_content": prompt_content,
        "fig": fig_artifact_text,
        "fig_artifact": fig_artifact_text,
        "raw_title": raw_title,
    }
    g_sys = render_file("visualizing/generate_desc_llm_sys.txt", desc_ctx)
    g_user = render_file("visualizing/generate_desc_llm_user.txt", desc_ctx)
    if not g_user.strip():
        g_user = prompt_content
    desc = _chat_with_optional_figure_image(
        g_sys or "你是数据可视化分析助手。",
        g_user,
        image_data_url=fig_image,
        name="viz.generate_desc",
    ).strip()

    return {
        "fig": fig,
        "fig_artifact": fig_artifact_text,
        "desc": desc,
        "raw_title": raw_title,
    }


def _generate_title_single(
    item: dict, cols_wo_id: list, base_ctx: dict[str, Any]
) -> str:
    fig = item.get("fig", "") if isinstance(item, dict) else ""
    desc = item.get("desc", "") if isinstance(item, dict) else ""
    raw_title = item.get("raw_title", "") if isinstance(item, dict) else ""
    fig_artifact_text = _figure_artifact_text(item)

    fig_meta = _extract_figure_metadata(fig)
    title_ctx = {
        **base_ctx,
        "fig": fig_artifact_text[:5000],
        "fig_artifact": fig_artifact_text,
        "desc": desc,
        "raw_title": raw_title,
        "cols_wo_id": cols_wo_id or [],
        "cols_wo_id_text": "，".join(str(c) for c in (cols_wo_id or [])),
        "existing_title": fig_meta.get("existing_title", ""),
        "chart_types_text": "，".join(fig_meta.get("chart_types", [])),
        "trace_names_text": "，".join(fig_meta.get("trace_names", [])),
        "x_axis_title": fig_meta.get("x_axis_title", ""),
        "y_axis_title": fig_meta.get("y_axis_title", ""),
        "legend_title": fig_meta.get("legend_title", ""),
        "trace_count": fig_meta.get("trace_count", 0),
    }

    title_sys = render_file("visualizing/title_llm_sys.txt", title_ctx)
    title_user = render_file("visualizing/title_llm_user.txt", title_ctx)
    title_text = chat(
        title_sys,
        title_user,
        name="viz.title",
        temperature=0.2,
    ).strip()

    parsed_title = _normalize_academic_title(_parse_single_title(title_text))
    if _is_usable_chinese_title(parsed_title):
        return parsed_title

    polished_title = _polish_title_to_chinese(
        candidate_title=parsed_title or _parse_single_title(title_text),
        item=item,
        cols_wo_id=cols_wo_id,
        base_ctx=base_ctx,
        fig_meta=fig_meta,
    )
    polished_title = _normalize_academic_title(polished_title)
    if _is_usable_chinese_title(polished_title):
        return polished_title

    fallback = _normalize_academic_title(
        _fallback_title_from_context(
            desc=desc,
            raw_title=raw_title,
            fig_meta=fig_meta,
            cols_wo_id=cols_wo_id,
        )
    )
    if _is_usable_chinese_title(fallback):
        return fallback

    return "主要变量分布"


def _summary_fig_single(
    item: dict, cols_wo_id: list, base_ctx: dict[str, Any]
) -> dict[str, Any]:
    fig = item.get("fig", "") if isinstance(item, dict) else ""
    desc = item.get("desc", "") if isinstance(item, dict) else ""
    fig_artifact_text = _figure_artifact_text(item)
    fig_image = item.get("fig_image", "") if isinstance(item, dict) else ""

    p_out = summary_fig_list_prompt(cols_wo_id=cols_wo_id, item=item)
    prompt = p_out.get("prompt", "")

    sfd_ctx = {
        **base_ctx,
        "prompt": prompt,
        "fig": fig_artifact_text,
        "fig_artifact": fig_artifact_text,
        "desc": desc,
    }
    sfd_sys = render_file("visualizing/summary_fig_desc_llm_sys.txt", sfd_ctx)
    sfd_user = render_file("visualizing/summary_fig_desc_llm_user.txt", sfd_ctx)
    if not sfd_user.strip():
        sfd_user = prompt
    analysis = _chat_with_optional_figure_image(
        sfd_sys or "你是数据可视化分析助手。",
        sfd_user,
        image_data_url=fig_image,
        name="viz.summary_fig_desc",
    ).strip()

    return {
        "fig": fig,
        "fig_artifact": fig_artifact_text,
        "desc": desc,
        "analysis": analysis,
    }


def _batch_run(items: list, func, concurrency: int = 10) -> list:
    if not items:
        return []

    results: list = [None] * len(items)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(func, item): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = {
                    **(items[idx] if isinstance(items[idx], dict) else {}),
                    "_error": str(exc)[:200],
                }
    return results


def _parse_title_list(text: str) -> list[str]:
    t = text.strip()
    if not t:
        return []

    try:
        parsed = json.loads(t)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    if t.startswith("```"):
        inner = _unwrap_code_block(t)
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    lines = []
    for line in t.splitlines():
        line = re.sub(r"^[\s\d.、,，\-\*]+", "", line).strip()
        if line:
            lines.append(line)
    return lines


def _parse_single_title(text: str) -> str:
    if not text:
        return ""

    title = text.strip()
    if title.startswith("```"):
        title = _unwrap_code_block(title)

    try:
        parsed = json.loads(title)
        if isinstance(parsed, str):
            title = parsed.strip()
        elif isinstance(parsed, list) and parsed:
            title = str(parsed[0]).strip()
    except Exception:
        pass

    title = title.strip().strip('"').strip("'").strip()
    if "\n" in title:
        parsed_lines = _parse_title_list(title)
        title = parsed_lines[0] if parsed_lines else ""
    return title[:200].strip()


def _normalize_academic_title(title: str) -> str:
    text = _parse_single_title(title)
    if not text:
        return ""

    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()

    chart_word_patterns = [
        r"(频率)?分布直方图",
        r"柱状图",
        r"条形图",
        r"折线图",
        r"散点图",
        r"箱线图",
        r"小提琴图",
        r"热力图",
        r"饼图",
        r"雷达图",
        r"bar chart",
        r"line chart",
        r"scatter plot",
        r"histogram",
        r"box plot",
        r"heatmap",
    ]
    for pattern in chart_word_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    compression_pairs = {
        "的分布特征及异常值分析": "分布特征",
        "分布特征及异常值分析": "分布特征",
        "的分布情况": "分布",
        "分布情况": "分布",
        "变化趋势分析": "变化趋势",
        "比较分析": "比较",
        "关系分析": "关系",
        "分布分析": "分布",
        "及异常值分析": "",
        "异常值分析": "",
    }
    for src, dst in compression_pairs.items():
        text = text.replace(src, dst)
    text = re.sub(r"的(分布|关系|比较|变化趋势|趋势|特征)", r"\1", text)

    english_map = {
        "relationship between": "关系",
        "relationship": "关系",
        "comparison of": "比较",
        "comparison": "比较",
        "distribution of": "分布",
        "distribution": "分布",
        "trend of": "变化趋势",
        "trend": "变化趋势",
        "change in": "变化",
        "changes in": "变化",
        "accuracy": "准确率",
        "correlation": "相关性",
    }
    lowered = text.lower()
    for src, dst in english_map.items():
        lowered = lowered.replace(src, dst)
    if _contains_ascii_letters(text):
        text = lowered

    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("，,。.;；:：")
    return _limit_title_length(text)


def _contains_ascii_letters(text: str) -> bool:
    return any(("a" <= ch.lower() <= "z") for ch in text)


def _limit_title_length(title: str) -> str:
    text = _parse_single_title(title)
    if not text:
        return ""

    text = re.sub(r"\s+", "", text).strip("，,。.;；:：")
    if len(text) <= MAX_VIS_TITLE_CHARS:
        return text

    removable_suffixes = (
        "综合分析",
        "特征分析",
        "趋势分析",
        "对比分析",
        "比较分析",
        "及异常值",
        "分析",
        "特征",
        "趋势",
    )
    for suffix in removable_suffixes:
        if text.endswith(suffix) and len(text) - len(suffix) >= 2:
            text = text[: -len(suffix)]
            if len(text) <= MAX_VIS_TITLE_CHARS:
                return text

    return text[:MAX_VIS_TITLE_CHARS].strip("，,。.;；:：")


def _is_vague_visualization_title(title: str) -> bool:
    text = re.sub(r"[\s，,。.;；:：]+", "", _parse_single_title(title))
    if not text:
        return True
    return text in VAGUE_VIS_TITLES


def _is_usable_chinese_title(title: str) -> bool:
    if not title:
        return False
    if _contains_ascii_letters(title) and not re.search(r"[\u4e00-\u9fff]", title):
        return False
    if len(title) > MAX_VIS_TITLE_CHARS:
        return False
    if len(title) < 2:
        return False
    return not _is_vague_visualization_title(title)


def _polish_title_to_chinese(
    *,
    candidate_title: str,
    item: dict,
    cols_wo_id: list,
    base_ctx: dict[str, Any],
    fig_meta: dict[str, Any],
) -> str:
    desc = item.get("desc", "") if isinstance(item, dict) else ""
    raw_title = item.get("raw_title", "") if isinstance(item, dict) else ""
    fig_artifact_text = _figure_artifact_text(item)

    sys_prompt = (
        "你是一名学术论文图表标题润色专家。"
        "请将候选标题改写为统一中文、正式、准确、简洁的论文图标题。"
        "不要解释，不要保留中英文混杂，不要把图类型词作为标题主体。"
        "标题必须不超过20个汉字或中文字符，且必须包含具体变量、指标、对象或组别。"
        "禁止输出“变量关系与分布特征”“变量关系”“分布特征”等泛化标题。"
        "如果候选标题不合格，请依据图表信息直接重写。"
    )
    user_prompt = (
        f"候选标题：{candidate_title}\n"
        f"用户需求：{base_ctx.get('user_input', '')}\n"
        f"可视化推荐：{base_ctx.get('visual_recommendation', '')}\n"
        f"精炼方案：{base_ctx.get('refined_suggestions', '')}\n"
        f"数据字段：{'，'.join(str(c) for c in (cols_wo_id or []))}\n"
        f"图中分组或系列：{'，'.join(fig_meta.get('trace_names', []))}\n"
        f"横轴信息：{fig_meta.get('x_axis_title', '')}\n"
        f"纵轴信息：{fig_meta.get('y_axis_title', '')}\n"
        f"图例信息：{fig_meta.get('legend_title', '')}\n"
        f"已有标题：{fig_meta.get('existing_title', '')}\n"
        f"图任务标识：{raw_title}\n"
        f"图表描述：{desc}\n"
        f"图表类型：{'，'.join(fig_meta.get('chart_types', []))}\n"
        f"图表压缩证据：{fig_artifact_text[:5000]}\n\n"
        "只输出一个最终中文标题，不超过20个字。"
    )
    try:
        rewritten = chat(
            sys_prompt,
            user_prompt,
            name="viz.title.polish",
            temperature=0.2,
        ).strip()
    except Exception:
        return ""
    return _limit_title_length(_parse_single_title(rewritten))


def _fallback_title_from_context(
    *,
    desc: str,
    raw_title: str,
    fig_meta: dict[str, Any],
    cols_wo_id: list,
) -> str:
    source_candidates = [
        fig_meta.get("existing_title", ""),
        raw_title,
        desc,
    ]
    for candidate in source_candidates:
        title = _normalize_academic_title(candidate)
        if title and not _is_vague_visualization_title(title):
            return title

    x_axis = _clean_title_token(fig_meta.get("x_axis_title", ""))
    y_axis = _clean_title_token(fig_meta.get("y_axis_title", ""))
    legend = _clean_title_token(fig_meta.get("legend_title", ""))
    trace_names = [_clean_title_token(name) for name in fig_meta.get("trace_names", [])]
    chart_types = [str(item).lower() for item in fig_meta.get("chart_types", [])]

    if _has_chart_type(chart_types, ("box", "histogram", "violin")):
        variable = y_axis or x_axis or _first_meaningful_token(cols_wo_id)
        if variable:
            return f"{variable}分布特征"

    if _has_chart_type(chart_types, ("scatter",)):
        if x_axis and y_axis:
            return f"{x_axis}与{y_axis}关系"

    if _has_chart_type(chart_types, ("bar", "pie")):
        metric = y_axis or _first_meaningful_token(trace_names) or _first_meaningful_token(cols_wo_id)
        group = x_axis or legend
        if group and metric and group != metric:
            return f"{group}{metric}比较"
        if metric:
            return f"{metric}比较"

    if _has_chart_type(chart_types, ("line",)):
        variable = y_axis or _first_meaningful_token(cols_wo_id)
        if variable:
            return f"{variable}变化趋势"

    variable = y_axis or x_axis or _first_meaningful_token(cols_wo_id)
    if variable:
        return f"{variable}分布"
    return "主要变量分布"


def _clean_title_token(value: Any) -> str:
    text = _normalize_academic_title(to_str(value))
    if not text or _is_vague_visualization_title(text):
        return ""
    return text


def _first_meaningful_token(values: Any) -> str:
    if not isinstance(values, list):
        values = [values]
    generic_tokens = {"id", "index", "count", "value", "variable", "class", "类别", "数量", "计数", "变量", "数值"}
    for value in values:
        text = _clean_title_token(value)
        if text and text.lower() not in generic_tokens:
            return text
    return ""


def _has_chart_type(chart_types: list[str], needles: tuple[str, ...]) -> bool:
    return any(any(needle in chart_type for needle in needles) for chart_type in chart_types)


def _extract_figure_metadata(fig: str) -> dict[str, Any]:
    empty = {
        "existing_title": "",
        "chart_types": [],
        "trace_names": [],
        "x_axis_title": "",
        "y_axis_title": "",
        "legend_title": "",
        "trace_count": 0,
    }

    try:
        payload = json.loads(to_str(fig) or "{}")
    except Exception:
        return empty

    if not isinstance(payload, dict):
        return empty

    layout = payload.get("layout") if isinstance(payload.get("layout"), dict) else {}
    data = payload.get("data") if isinstance(payload.get("data"), list) else []

    chart_types: list[str] = []
    trace_names: list[str] = []
    for trace in data:
        if not isinstance(trace, dict):
            continue
        trace_type = str(trace.get("type", "")).strip()
        if trace_type and trace_type not in chart_types:
            chart_types.append(trace_type)
        trace_name = str(trace.get("name", "")).strip()
        if trace_name and trace_name not in trace_names:
            trace_names.append(trace_name)

    return {
        "existing_title": _layout_text(layout.get("title")),
        "chart_types": chart_types,
        "trace_names": trace_names,
        "x_axis_title": _axis_title_text(layout.get("xaxis")),
        "y_axis_title": _axis_title_text(layout.get("yaxis")),
        "legend_title": _axis_title_text(layout.get("legend")),
        "trace_count": len(data),
    }


def _layout_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text.strip()
    return ""


def _axis_title_text(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    title = value.get("title")
    if isinstance(title, str):
        return title.strip()
    if isinstance(title, dict):
        text = title.get("text")
        if isinstance(text, str):
            return text.strip()
    return ""


def _filter_id_columns(cols: list) -> list:
    out = []
    for col in cols or []:
        name = str(col).lower()
        if name in ("id", "index", "idx", "序号", "编号", "_id"):
            continue
        if name.endswith("_id") or name.endswith("id"):
            if name != "grid":
                continue
        out.append(col)
    return out or list(cols or [])


def _unwrap_code_block(text: str) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _empty_result() -> dict[str, Any]:
    return {
        "full": "",
        "abstract_3": "",
        "summary_3": {"title": "", "fig_analysis": []},
        "visual_recommendatio": "",
        "final_code": "",
        "tu_title": [],
    }


if __name__ == "__main__":
    import sys

    import pandas as pd

    from workflows._plugins import df_to_meta

    if len(sys.argv) < 2:
        print("用法: python -m workflows.visualizing <csv_path> [user_input]")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    user_input = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"读取 {sys.argv[1]}: {df.shape}")

    meta = df_to_meta(df)
    result = run_visualizing_workflow(
        data=meta["df"],
        shape0=meta["shape_0"],
        shape1=meta["shape_1"],
        cols=list(df.columns),
        def_head=meta["head_dict_str"],
        vis_auto=True,
        user_input=user_input,
    )

    print("\n===== visual_recommendatio =====")
    print(result["visual_recommendatio"][:500])
    print("\n===== final_code =====")
    print(result["final_code"][:500])
    print(f"\n===== 图表数量: {len(result['summary_3']['fig_analysis'])} =====")
    for i, fa in enumerate(result["summary_3"]["fig_analysis"][:2]):
        print(f"\n--- 图 {i + 1} ---")
        print(f"analysis: {fa['analysis'][:200]}")
    print("\n===== abstract_3 =====")
    print(result["abstract_3"])
