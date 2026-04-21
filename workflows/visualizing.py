"""
Visualizing workflow local implementation.
"""

from __future__ import annotations

import ast
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from core.llm_client import chat
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
    if not vis_auto:
        return _empty_result()

    ctx: dict[str, Any] = {
        "data": data,
        "shape_0": shape0,
        "shape_1": shape1,
        "shape0": shape0,
        "shape1": shape1,
        "cols": cols,
        "def_head": def_head,
        "df_head": def_head,
        "color": color or "",
        "user_input": user_input or "",
        "add_preference": add_preference or "",
        "preference_selected": preference_selected or "",
        "ref_context": ref_context or "（无参考资料）",
    }

    vr_sys = render_file("visualizing/sec3_get_visual_recommendation_llm_sys.txt", ctx)
    vr_user = render_file("visualizing/sec3_get_visual_recommendation_llm_user.txt", ctx)
    visual_recommendatio = chat(
        vr_sys, vr_user, name="viz.get_visual_recommendation"
    ).strip()
    ctx["visual_recommendation"] = visual_recommendatio
    ctx["visual_recommendatio"] = visual_recommendatio

    rs_sys = render_file("visualizing/sec3_refine_suggestions_llm_sys.txt", ctx)
    rs_user = render_file("visualizing/sec3_refine_suggestions_llm_user.txt", ctx)
    refined_suggestions = chat(rs_sys, rs_user, name="viz.refine_suggestions").strip()
    ctx["refined_suggestions"] = refined_suggestions

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

    pack_data_list = _batch_run(
        fig_task_list,
        lambda item: _desc_fig_single(item, dtype_info, ctx),
        concurrency=BATCH_CONCURRENCY,
    )
    title_results = _batch_run(
        pack_data_list,
        lambda item: _generate_title_single(item, cols_wo_id, ctx),
        concurrency=BATCH_CONCURRENCY,
    )
    tu_title = [item.strip() if isinstance(item, str) else "" for item in title_results]

    aggregate_results = _batch_run(
        pack_data_list,
        lambda item: _summary_fig_single(item, cols_wo_id, ctx),
        concurrency=BATCH_CONCURRENCY,
    )

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


def _desc_fig_single(
    item: dict, dtype_info: str, base_ctx: dict[str, Any]
) -> dict[str, Any]:
    fig = item.get("fig", "") if isinstance(item, dict) else ""
    raw_title = item.get("title", "") if isinstance(item, dict) else ""

    p_out = desc_fig_prompt(dtype_info=dtype_info, fig=fig)
    prompt_content = p_out.get("prompt_content", "")

    desc_ctx = {
        **base_ctx,
        "prompt_content": prompt_content,
        "fig": fig,
        "raw_title": raw_title,
    }
    g_sys = render_file("visualizing/generate_desc_llm_sys.txt", desc_ctx)
    g_user = render_file("visualizing/generate_desc_llm_user.txt", desc_ctx)
    if not g_user.strip():
        g_user = prompt_content
    desc = chat(
        g_sys or "你是数据可视化分析助手。",
        g_user,
        name="viz.generate_desc",
    ).strip()

    return {"fig": fig, "desc": desc, "raw_title": raw_title}


def _generate_title_single(
    item: dict, cols_wo_id: list, base_ctx: dict[str, Any]
) -> str:
    fig = item.get("fig", "") if isinstance(item, dict) else ""
    desc = item.get("desc", "") if isinstance(item, dict) else ""
    raw_title = item.get("raw_title", "") if isinstance(item, dict) else ""

    fig_meta = _extract_figure_metadata(fig)
    title_ctx = {
        **base_ctx,
        "fig": fig[:4000],
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

    fallback = _normalize_academic_title(_fallback_title_from_desc(desc))
    if _is_usable_chinese_title(fallback):
        return fallback

    return "变量关系与分布特征"


def _summary_fig_single(
    item: dict, cols_wo_id: list, base_ctx: dict[str, Any]
) -> dict[str, Any]:
    fig = item.get("fig", "") if isinstance(item, dict) else ""
    desc = item.get("desc", "") if isinstance(item, dict) else ""

    p_out = summary_fig_list_prompt(cols_wo_id=cols_wo_id, item=item)
    prompt = p_out.get("prompt", "")

    sfd_ctx = {**base_ctx, "prompt": prompt, "fig": fig, "desc": desc}
    sfd_sys = render_file("visualizing/summary_fig_desc_llm_sys.txt", sfd_ctx)
    sfd_user = render_file("visualizing/summary_fig_desc_llm_user.txt", sfd_ctx)
    if not sfd_user.strip():
        sfd_user = prompt
    analysis = chat(
        sfd_sys or "你是数据可视化分析助手。",
        sfd_user,
        name="viz.summary_fig_desc",
    ).strip()

    return {"fig": fig, "desc": desc, "analysis": analysis}


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
    return text.strip("，,。.;；:：")


def _contains_ascii_letters(text: str) -> bool:
    return any(("a" <= ch.lower() <= "z") for ch in text)


def _is_usable_chinese_title(title: str) -> bool:
    if not title:
        return False
    if _contains_ascii_letters(title):
        return False
    if len(title) < 2:
        return False
    banned = {"结果图", "比较图", "分析图", "数据展示", "模型表现", "可视化结果"}
    return title not in banned


def _polish_title_to_chinese(
    *,
    candidate_title: str,
    item: dict,
    cols_wo_id: list,
    base_ctx: dict[str, Any],
    fig_meta: dict[str, Any],
) -> str:
    fig = item.get("fig", "") if isinstance(item, dict) else ""
    desc = item.get("desc", "") if isinstance(item, dict) else ""
    raw_title = item.get("raw_title", "") if isinstance(item, dict) else ""

    sys_prompt = (
        "你是一名学术论文图表标题润色专家。"
        "请将候选标题改写为统一中文、正式、准确、简洁的论文图标题。"
        "不要解释，不要保留中英文混杂，不要把图类型词作为标题主体。"
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
        f"图表 JSON 摘要：{fig[:2500]}\n\n"
        "只输出一个最终中文标题。"
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
    return _parse_single_title(rewritten)


def _fallback_title_from_desc(desc: str) -> str:
    text = _parse_single_title(desc)
    if not text:
        return "变量关系与分布特征"
    return text[:40].strip("，,。.;；:：")


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
