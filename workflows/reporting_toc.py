"""
Reporting_toc workflow 本地实现。

修正版目标：
1. 保持旧版“两步法”原理：先 summarize，再 generate toc；
2. 目录阶段只吃“短摘要”，不再把大段 summary / full / code 全塞进 prompt；
3. 避免 token 爆炸；
4. 输出结构与 reporting_partly / report_render 保持兼容。

原 Coze 流程：
    Start → Condition(report_auto==True)
      → summarize_all_sections(LLM)       [综合前 4 个 summary]
      → generate_toc_from_summary(LLM)    [生成目录 markdown]
      → Code(规整目录 + 补齐"结论"章节)
      → End
"""
from __future__ import annotations

import re
from typing import Any

from core.llm_client import chat
from core.prompt_template import render_file
from workflow.report.report_content_utils import (
    truncate_text,
    shrink_summary_for_toc,
    normalize_toc_md_input,
)


def run_reporting_toc_workflow(
    *,
    load_summary: dict,
    preproc_summary: dict,
    visual_summary: dict,
    coding_summary: dict,
    load_abstract: str,
    preproc_abstract: str,
    visual_abstract: str,
    coding_abstract: str,
    selected_full_conten: str = "",
    toc_md: list | None = None,
    outline_length: str = "标准",
    report_auto: bool = True,
    user_input: str = "",
    add_preference: str = "",
    preference_selected: str = "",
    ref_context: str = "",
) -> dict[str, Any]:
    """Run Reporting_toc workflow."""
    if not report_auto:
        return _passthrough(
            load_abstract,
            preproc_abstract,
            visual_abstract,
            coding_abstract,
            selected_full_conten,
            add_preference,
            preference_selected,
            toc_text="",
        )

    # 关键修复：
    # 目录阶段只使用“瘦身后的 summary + 截断后的 abstract”
    # 不再把大段全文 / 代码 / 大图分析喂进来
    ctx: dict[str, Any] = {
        "load_summary": shrink_summary_for_toc(load_summary),
        "preproc_summary": shrink_summary_for_toc(preproc_summary),
        "visual_summary": shrink_summary_for_toc(visual_summary),
        "coding_summary": shrink_summary_for_toc(coding_summary),
        "load_abstract": truncate_text(load_abstract, 1200),
        "preproc_abstract": truncate_text(preproc_abstract, 1200),
        "visual_abstract": truncate_text(visual_abstract, 1200),
        "coding_abstract": truncate_text(coding_abstract, 1200),
        # toc 阶段不使用全文，显式清空，防止 prompt 模板误带入
        "selected_full_conten": "",
        "toc_md": normalize_toc_md_input(toc_md),
        "outline_length": outline_length or "标准",
        "user_input": truncate_text(user_input, 500),
        "add_preference": truncate_text(add_preference, 500),
        "preference_selected": truncate_text(preference_selected, 500),
        "ref_context": truncate_text(ref_context, 1500) if ref_context else "（无参考资料）",
    }

    # ---------- 节点 1: summarize_all_sections ----------
    s_sys = render_file("reporting_toc/summarize_all_sections_llm_sys.txt", ctx)
    s_user = render_file("reporting_toc/summarize_all_sections_llm_user.txt", ctx)
    full_summary = chat(s_sys, s_user, name="report_toc.summarize").strip()
    ctx["full_summary"] = truncate_text(full_summary, 2500)

    # ---------- 节点 2: generate_toc_from_summary ----------
    t_sys = render_file("reporting_toc/generate_toc_from_summary_llm_sys.txt", ctx)
    t_user = render_file("reporting_toc/generate_toc_from_summary_llm_user.txt", ctx)
    toc_raw = chat(t_sys, t_user, name="report_toc.generate_toc").strip()

    # ---------- 节点 3: 规整目录 + 补齐"结论" ----------
    toc_text = _normalize_toc(toc_raw, visual_summary=ctx.get("visual_summary"))

    return {
        "toc_text": toc_text,
        # 注意：这里只是 passthrough 给 reporting_partly，toc 阶段自己不使用全文
        "selected_full_conten": selected_full_conten or "",
        "load_abstract": load_abstract or "",
        "preproc_abstract": preproc_abstract or "",
        "visual_abstract": visual_abstract or "",
        "coding_abstract": coding_abstract or "",
        "add_preference": add_preference or "",
        "preference_select": preference_selected or "",
        "ref_context": ref_context or "",
        "_full_summary": full_summary,  # 调试用
    }


CHART_TYPE_ONLY_TITLES = {
    "小提琴图",
    "散点图",
    "散点图矩阵",
    "热力图",
    "平行坐标图",
    "PCA降维图",
    "主成分分析图",
    "箱线图",
    "柱状图",
    "条形图",
    "折线图",
    "直方图",
    "饼图",
    "雷达图",
}


def _split_toc_line(line: str) -> tuple[str, str, str] | None:
    match = re.match(r"^(\d+(?:[\.．]\d+)*)(?:[\.．、]|\s+)?\s*(.*)$", str(line or "").strip())
    if not match:
        return None

    num = match.group(1).replace("．", ".").strip()
    remainder = match.group(2).strip()
    outline = ""
    outline_match = re.search(r"[（(]([^()（）]*)[）)]\s*$", remainder)
    if outline_match:
        outline = outline_match.group(1).strip()
        title = remainder[: outline_match.start()].strip()
    else:
        title = remainder.strip()
    return num, title, outline


def _clean_generated_visual_title(text: str) -> str:
    cleaned = re.sub(r"\s+", "", str(text or ""))
    cleaned = re.sub(
        r"^(?:小提琴图|散点图矩阵|散点图|热力图|平行坐标图|PCA降维图|"
        r"主成分分析图|箱线图|柱状图|条形图|折线图|直方图|饼图|雷达图)"
        r"(?:展示|显示|反映|呈现|说明|揭示)?(?:了)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?:小提琴图|散点图矩阵|散点图|热力图|平行坐标图|PCA降维图|"
        r"主成分分析图|箱线图|柱状图|条形图|折线图|直方图|饼图|雷达图)$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip(" ：:，,。.;；、-_/\\|")
    return cleaned[:28].rstrip()


def _visual_topic_candidates(visual_summary: Any) -> list[str]:
    if not isinstance(visual_summary, dict):
        return []
    topics: list[str] = []
    fig_analysis = visual_summary.get("fig_analysis")
    if not isinstance(fig_analysis, list):
        return topics
    for item in fig_analysis:
        if not isinstance(item, dict):
            continue
        topic = _clean_generated_visual_title(item.get("topic", ""))
        if topic and topic not in topics:
            topics.append(topic)
    return topics


def _soften_visual_chart_type_titles(toc_lines: list[str], visual_summary: Any) -> list[str]:
    topic_candidates = _visual_topic_candidates(visual_summary)
    topic_index = 0
    softened: list[str] = []

    for line in toc_lines:
        parsed = _split_toc_line(line)
        if parsed is None:
            softened.append(line)
            continue

        num, title, outline = parsed
        compact_title = re.sub(r"\s+", "", title)
        if not num.startswith("3.") or compact_title not in CHART_TYPE_ONLY_TITLES:
            softened.append(line)
            continue

        replacement = _clean_generated_visual_title(outline)
        if not replacement and topic_index < len(topic_candidates):
            replacement = topic_candidates[topic_index]
            topic_index += 1
        if not replacement:
            replacement = "可视化发现"

        suffix = f"（{outline}）" if outline else ""
        softened.append(f"{num}{replacement}{suffix}")

    return softened


def _normalize_toc(raw: str, visual_summary: Any | None = None) -> str:
    """
    - 把 \\n 转成真换行
    - 只保留目录项
    - 缺失“结论/展望”就补一条
    """
    if not raw:
        return ""

    raw_text = str(raw).replace("\\n", "\n")
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

    toc_lines = []
    for line in lines:
        if re.match(r"^\d+(\.\d+)*[\.．]?", line):
            toc_lines.append(line)

    if not any("结论" in line or "展望" in line for line in toc_lines):
        toc_lines.append("5.结论与应用展望（总结分析发现及模型表现，提出后续优化方向）")

    toc_lines = _soften_visual_chart_type_titles(toc_lines, visual_summary)
    return "\n".join(toc_lines)


def _passthrough(
    la: str,
    pa: str,
    va: str,
    ca: str,
    sfc: str,
    ap: str,
    ps: str,
    toc_text: str = "",
) -> dict[str, Any]:
    return {
        "toc_text": toc_text,
        "selected_full_conten": sfc or "",
        "load_abstract": la or "",
        "preproc_abstract": pa or "",
        "visual_abstract": va or "",
        "coding_abstract": ca or "",
        "add_preference": ap or "",
        "preference_select": ps or "",
    }
