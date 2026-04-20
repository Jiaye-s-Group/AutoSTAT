"""
Visualizing workflow 本地实现。

原 Coze 流程：
    Start → Condition(vis_auto==True)
      → sec3_get_visual_recommendation(LLM)     [推荐可视化方案]
      → sec3_refine_suggestions(LLM)            [精炼]
      → sec3_code_generation(LLM)               [生成 plotly 代码]
      → Variable assign: code_vis = generation_code
      → Loop(max 5): [代码修复循环]
          ├─ validate_viz_code(plugin)
          ├─ if success: break
          └─ sec3_fixed_code(LLM) → 更新 code_vis
      → Code_1(取 Loop 的 final_code)
      → Title(LLM) → tu_title
      → sec3_execute_and_extract(plugin)        [真跑代码拿 fig_task_list]
      → Batch sec3_desc_fig [对每张图并发]:
          ├─ desc_fig_prompt(plugin)
          ├─ Generate_Desc(LLM)
          └─ Pack_Data(code): {fig, desc}
      → Batch sec3_summary_fig_list [对每张图并发]:
          ├─ summary_fig_list_prompt(plugin)
          ├─ summary_fig_Desc(LLM)  —— 给每张图生成「分析」
          └─ Aggregation(code): {fig, desc, analysis}
      → sec3_check_full(plugin) → full
      → sec3_abstract(LLM) → abstract_3
      → sec3_summary_html(plugin) → summary_3
      → Code(兜底) → End

输出:
    {
      "full": "所有图分析合并文本",
      "abstract_3": "...",
      "summary_3": {title, fig_analysis: [{fig, analysis}...]},
      "visual_recommendatio": "推荐方案",   # 注意是 "recommendatio" 少个 n，与 Coze 对齐
      "final_code": "...",
      "tu_title": [...],
    }
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from core.llm_client import chat
from core.prompt_template import render_file
from core.workflow_runner import to_str
from workflows._plugins import (
    validate_viz_code,
    execute_and_extract,
    desc_fig_prompt,
    summary_fig_list_prompt,
    sec3_composer,
    sec3_check_full,
)

MAX_FIX_ATTEMPTS = 5
BATCH_CONCURRENCY = 10  # 原 Coze 里 Batch 并发度就是 10


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
    # ---------- Condition: vis_auto ----------
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

    # ---------- 节点 1: sec3_get_visual_recommendation ----------
    vr_sys = render_file("visualizing/sec3_get_visual_recommendation_llm_sys.txt", ctx)
    vr_user = render_file("visualizing/sec3_get_visual_recommendation_llm_user.txt", ctx)
    visual_recommendatio = chat(
        vr_sys, vr_user, name="viz.get_visual_recommendation"
    ).strip()
    ctx["visual_recommendation"] = visual_recommendatio
    ctx["visual_recommendatio"] = visual_recommendatio  # 兼容拼写

    # ---------- 节点 2: sec3_refine_suggestions ----------
    rs_sys = render_file("visualizing/sec3_refine_suggestions_llm_sys.txt", ctx)
    rs_user = render_file("visualizing/sec3_refine_suggestions_llm_user.txt", ctx)
    refined_suggestions = chat(rs_sys, rs_user, name="viz.refine_suggestions").strip()
    ctx["refined_suggestions"] = refined_suggestions

    # ---------- 节点 3: sec3_code_generation ----------
    cg_sys = render_file("visualizing/sec3_code_generation_llm_sys.txt", ctx)
    cg_user = render_file("visualizing/sec3_code_generation_llm_user.txt", ctx)
    generation_code = chat(cg_sys, cg_user, name="viz.code_generation").strip()
    generation_code = _sanitize_visualization_code(generation_code)

    # ---------- 节点 4: Loop (代码修复，最多 5 次) ----------
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

        # ---------- 节点 4.x: sec3_fixed_code (LLM) ----------
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
            fix_sys, fix_user, name=f"viz.fixed_code.{attempt+1}", temperature=0.3
        ).strip()
        fixed = _sanitize_visualization_code(fixed)
        if fixed:
            current_code = fixed

    final_code = current_code

    if not success:
        # 代码最终没跑通，返回占位结果
        return {
            "full": "",
            "abstract_3": f"可视化代码生成失败：{last_error[:500]}",
            "summary_3": {"title": "数据可视化", "fig_analysis": []},
            "visual_recommendatio": visual_recommendatio,
            "final_code": final_code,
            "tu_title": [],
        }

    # ---------- 节点 5: Title LLM ----------
    ctx["final_code"] = final_code
    title_sys = render_file("visualizing/title_llm_sys.txt", ctx)
    title_user = render_file("visualizing/title_llm_user.txt", ctx)
    titles_raw = chat(title_sys, title_user, name="viz.title").strip()
    tu_title = _parse_title_list(titles_raw)

    # ---------- 节点 6: sec3_execute_and_extract (plugin) ----------
    exec_result = execute_and_extract(code=final_code, df_data=data)
    fig_task_list = exec_result.get("fig_task_list", [])
    if not fig_task_list:
        return {
            "full": "",
            "abstract_3": "未能从代码中提取到任何图表",
            "summary_3": {"title": "数据可视化", "fig_analysis": []},
            "visual_recommendatio": visual_recommendatio,
            "final_code": final_code,
            "tu_title": tu_title,
        }

    # 构建 cols_wo_id（去除明显的 ID 列，用于 summary_fig_list_prompt）
    cols_wo_id = _filter_id_columns(cols)

    # 获取 dtype_info（从 def_head 里）
    dtype_info = to_str(def_head)

    # ---------- 节点 7: Batch sec3_desc_fig (并发) ----------
    # 对每个 fig_task_list 元素跑: desc_fig_prompt → Generate_Desc → Pack_Data
    pack_data_list = _batch_run(
        fig_task_list,
        lambda item: _desc_fig_single(item, dtype_info, ctx),
        concurrency=BATCH_CONCURRENCY,
    )

    # ---------- 节点 8: Batch sec3_summary_fig_list (并发) ----------
    # 对每个 pack_data 元素跑: summary_fig_list_prompt → summary_fig_Desc → Aggregation
    aggregate_results = _batch_run(
        pack_data_list,
        lambda item: _summary_fig_single(item, cols_wo_id, ctx),
        concurrency=BATCH_CONCURRENCY,
    )

    # ---------- 节点 9: sec3_check_full (plugin) ----------
    full = sec3_check_full(analysis_list=aggregate_results)["full"]

    # ---------- 节点 10: sec3_abstract (LLM) ----------
    abs_ctx = {**ctx, "all_analyses": full, "full": full}
    abs_sys = render_file("visualizing/sec3_abstract_llm_sys.txt", abs_ctx)
    abs_user = render_file("visualizing/sec3_abstract_llm_user.txt", abs_ctx)
    abstract_3 = chat(abs_sys, abs_user, name="viz.abstract").strip()

    # ---------- 节点 11: sec3_summary_html (plugin: sec3_composer) ----------
    composed = sec3_composer(fig_analysis=aggregate_results)

    return {
        "full": full,
        "abstract_3": abstract_3,
        "summary_3": composed["summary_3"],
        "visual_recommendatio": visual_recommendatio,
        "final_code": final_code,
        "tu_title": tu_title,
    }


# ===================================================================
# 辅助函数
# ===================================================================


def _desc_fig_single(
    item: dict, dtype_info: str, base_ctx: dict[str, Any]
) -> dict[str, Any]:
    """一张图的 desc 生成 pipeline（对应 Batch 内的 3 个节点）。"""
    fig = item.get("fig", "") if isinstance(item, dict) else ""

    # 1. desc_fig_prompt(plugin)
    p_out = desc_fig_prompt(dtype_info=dtype_info, fig=fig)
    prompt_content = p_out.get("prompt_content", "")

    # 2. Generate_Desc(LLM)
    desc_ctx = {**base_ctx, "prompt_content": prompt_content, "fig": fig}
    g_sys = render_file("visualizing/generate_desc_llm_sys.txt", desc_ctx)
    g_user = render_file("visualizing/generate_desc_llm_user.txt", desc_ctx)
    # 如果 user prompt 本身很短（原版是直接把 prompt_content 传入），我们用 prompt_content 兜底
    if not g_user.strip():
        g_user = prompt_content
    desc = chat(g_sys or "你是数据可视化分析助手。", g_user, name="viz.generate_desc").strip()

    # 3. Pack_Data(code)
    return {"fig": fig, "desc": desc}


def _summary_fig_single(
    item: dict, cols_wo_id: list, base_ctx: dict[str, Any]
) -> dict[str, Any]:
    """一张图的 analysis 生成 pipeline。"""
    fig = item.get("fig", "") if isinstance(item, dict) else ""
    desc = item.get("desc", "") if isinstance(item, dict) else ""

    # 1. summary_fig_list_prompt(plugin)
    p_out = summary_fig_list_prompt(cols_wo_id=cols_wo_id, item=item)
    prompt = p_out.get("prompt", "")

    # 2. summary_fig_Desc(LLM)
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

    # 3. Aggregation(code)
    return {"fig": fig, "desc": desc, "analysis": analysis}


def _batch_run(items: list, func, concurrency: int = 10) -> list:
    """
    并发对 items 的每个元素跑 func。
    保持输出顺序与输入一致。失败的元素输出原始 item（保守策略）。
    """
    if not items:
        return []
    results: list = [None] * len(items)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(func, item): i for i, item in enumerate(items)}
        for f in as_completed(futures):
            idx = futures[f]
            try:
                results[idx] = f.result()
            except Exception as exc:
                results[idx] = {
                    **(items[idx] if isinstance(items[idx], dict) else {}),
                    "_error": str(exc)[:200],
                }
    return results


def _parse_title_list(text: str) -> list[str]:
    """LLM 返回的标题清单可能是 JSON 列表或每行一个标题，都兼容。"""
    import json

    t = text.strip()
    if not t:
        return []
    # 尝试 JSON list
    try:
        parsed = json.loads(t)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    # 尝试 ``` 包裹的 JSON
    if t.startswith("```"):
        inner = _unwrap_code_block(t)
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    # 退化：按行拆分，去掉序号前缀
    import re

    lines = []
    for line in t.splitlines():
        line = re.sub(r"^[\s\d.、，,\-\*•]+", "", line).strip()
        if line:
            lines.append(line)
    return lines


def _filter_id_columns(cols: list) -> list:
    """粗糙地过滤掉 id/index 列，Coze 的 Batch prompt 里也会跳过。"""
    out = []
    for c in cols or []:
        name = str(c).lower()
        if name in ("id", "index", "idx", "序号", "编号", "_id"):
            continue
        if name.endswith("_id") or name.endswith("id"):
            if name != "grid":  # 避免把 "grid" 这种名字误杀
                continue
        out.append(c)
    return out or list(cols or [])


def _unwrap_code_block(text: str) -> str:
    """去掉 ```python ... ``` 的包装。"""
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return t


def _empty_result() -> dict[str, Any]:
    return {
        "full": "",
        "abstract_3": "",
        "summary_3": {"title": "", "fig_analysis": []},
        "visual_recommendatio": "",
        "final_code": "",
        "tu_title": [],
    }


# ---------- CLI 测试入口 ----------

if __name__ == "__main__":
    import sys

    import pandas as pd

    from workflows._plugins import df_to_meta

    if len(sys.argv) < 2:
        print("用法: python -m workflows.visualizing <csv_path> [user_input]")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    user_input = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"✓ 读取 {sys.argv[1]}: {df.shape}")

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
        print(f"\n--- 图 {i+1} ---")
        print(f"analysis: {fa['analysis'][:200]}")
    print("\n===== abstract_3 =====")
    print(result["abstract_3"])
