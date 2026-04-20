"""
Reporting_partly workflow 本地实现。

修正版目标：
1. history_content 只作为 writer 的上文参考，不再污染最终成品；
2. 每个章节最终只写入“当前节内容”，避免 1 / 1+2 / 1+2+3 叠加；
3. Markdown 标题层级本地统一包装，保证前端显示与下载一致；
4. 保留 [FIG:x] 占位符，交给 report_render / 导出层替换成真实图片；
5. 不再优先使用 validator 结果，避免 [FIG:x] 被吃掉；
6. 避免“模型已经写了标题 + 本地又包一层标题”导致标题重复。
"""
from __future__ import annotations

import json
import re
from typing import Any

from core.llm_client import chat
from core.prompt_template import render_file
from workflow.report.report_content_utils import (
    normalize_part,
    normalize_for_dedup,
    wrap_section_as_markdown,
    build_history_context,
    truncate_text,
)


def _clean_report_title(raw_title: Any) -> str:
    if raw_title is None:
        return ""

    if isinstance(raw_title, dict):
        for key in ("title", "text", "name", "label", "content"):
            cleaned = _clean_report_title(raw_title.get(key))
            if cleaned:
                return cleaned
        return ""

    if isinstance(raw_title, list):
        for item in raw_title:
            cleaned = _clean_report_title(item)
            if cleaned:
                return cleaned
        return ""

    text = str(raw_title).strip()
    if not text:
        return ""

    code_block_match = re.match(
        r"^```(?:json|text|markdown)?\s*(.*?)\s*```$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if code_block_match:
        text = code_block_match.group(1).strip()

    try:
        parsed_json = json.loads(text)
    except Exception:
        parsed_json = None

    if parsed_json is not None and parsed_json != raw_title:
        cleaned = _clean_report_title(parsed_json)
        if cleaned:
            return cleaned

    json_title_match = re.search(r'"title"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if json_title_match:
        return json_title_match.group(1).strip()

    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r'^[\s`"\']+', "", text)
    text = re.sub(r'[\s`"\']+$', "", text)
    text = re.sub(r"^[《【「『]+", "", text)
    text = re.sub(r"[》】」』]+$", "", text)
    return text.strip()


def run_reporting_partly_workflow(
    *,
    toc_text: str,
    selected_full_conten: str,
    load_abstract: str,
    preproc_abstract: str,
    visual_abstract: str,
    coding_abstract: str,
    user_input: str = "",
    add_preference: str = "",
    preference_select: str = "",
    ref_context: str = "",
) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "toc_text": toc_text or "",
        "toc_md": toc_text or "",
        "selected_full_conten": selected_full_conten or "",
        "selected_full_contents_vis": selected_full_conten or "",
        "load_abstract": load_abstract or "",
        "preproc_abstract": preproc_abstract or "",
        "visual_abstract": visual_abstract or "",
        "coding_abstract": coding_abstract or "",
        "user_input": user_input or "",
        "add_preference": add_preference or "",
        "preference_selected": preference_select or "",
        "preference_select": preference_select or "",
        "ref_context": ref_context or "（无参考资料）",
    }

    # ---------- 节点 1: selected_photo_update_toc ----------
    sp_sys = render_file("reporting_partly/selected_photo_update_toc_llm_sys.txt", ctx)
    sp_user = render_file("reporting_partly/selected_photo_update_toc_llm_user.txt", ctx)
    toc_list_raw = chat(sp_sys, sp_user, name="report_partly.select_photo").strip()
    toc_list = _parse_toc_list(toc_list_raw)
    ctx["toc_list"] = toc_list

    # ---------- 节点 2: update_toc_with_relevant_sections ----------
    up_sys = render_file("reporting_partly/update_toc_with_relevant_sections_llm_sys.txt", ctx)
    up_user = render_file("reporting_partly/update_toc_with_relevant_sections_llm_user.txt", ctx)
    toc_final_raw = chat(up_sys, up_user, name="report_partly.update_toc").strip()
    toc_list_final = _parse_toc_list(toc_final_raw)

    if not toc_list_final:
        toc_list_final = toc_list

    if not toc_list_final:
        toc_list_final = [
            line.strip() for line in (toc_text or "").splitlines() if line.strip()
        ]

    report_parts: list[str] = []
    seen_parts: set[str] = set()
    history_parts_for_prompt: list[str] = []

    for idx, section in enumerate(toc_list_final):
        section_ctx: dict[str, Any] = {
            **ctx,
            "t": section,
            "section": section if isinstance(section, str) else json.dumps(section, ensure_ascii=False),
            "toc_section": section,
            "toc": json.dumps(toc_list_final, ensure_ascii=False),
            "toc_list_final": toc_list_final,
            "history_content": build_history_context(history_parts_for_prompt, max_chars=1800),
            "section_index": idx,
            "total_sections": len(toc_list_final),
        }

        # ---------- writer ----------
        w_sys = render_file("reporting_partly/writer_llm_sys.txt", section_ctx)
        w_user = render_file("reporting_partly/writer_llm_user.txt", section_ctx)
        content = chat(w_sys, w_user, name=f"report_partly.writer.{idx+1}").strip()
        content = _unwrap_code_block(content)
        content = normalize_part(content)
        section_ctx["content"] = content

        # ---------- fill_report ----------
        f_sys = render_file("reporting_partly/fill_report_llm_sys.txt", section_ctx)
        f_user = render_file("reporting_partly/fill_report_llm_user.txt", section_ctx)
        filled = chat(f_sys, f_user, name=f"report_partly.fill.{idx+1}").strip()
        filled = _unwrap_code_block(filled)
        filled = normalize_part(filled)

        # debug：检查图号在哪一层还存在
        print(f"[REPORT][SECTION {idx+1}] HAS_FIG_IN_FILLED =", "[FIG:" in filled)
        print(f"[REPORT][SECTION {idx+1}] FILLED_PREVIEW =", filled[:300])

        # ---------- writer_validator ----------
        validator_ctx = {
            **section_ctx,
            "content": filled,
            "history_content": build_history_context(history_parts_for_prompt, max_chars=1200),
        }
        v_sys = render_file("reporting_partly/writer_validator_llm_sys.txt", validator_ctx)
        v_user = render_file("reporting_partly/writer_validator_llm_user.txt", validator_ctx)
        _ = chat(
            v_sys,
            v_user,
            name=f"report_partly.validator.{idx+1}",
            temperature=0.3,
        ).strip()

        # 最终正文优先用 filled，再退回 content
        final_part = filled or content
        final_part = normalize_part(final_part)

        print(f"[REPORT][SECTION {idx+1}] HAS_FIG_IN_FINAL =", "[FIG:" in final_part)

        if not final_part:
            continue

        # 关键修复：去掉“模型已经生成的重复标题”
        final_part = _strip_redundant_heading(final_part, section)
        final_part = normalize_part(final_part)

        if not final_part:
            continue

        dedup_key = normalize_for_dedup(final_part)
        if dedup_key in seen_parts:
            continue

        seen_parts.add(dedup_key)

        # 如果模型已经输出 markdown 标题，就不要再包一层，避免标题重复
        if _starts_with_markdown_heading(final_part):
            wrapped_part = final_part
        else:
            wrapped_part = wrap_section_as_markdown(section, final_part)

        wrapped_part = normalize_part(wrapped_part)
        if wrapped_part:
            report_parts.append(wrapped_part)

        # history 只给下一轮 writer 看，不参与最终成品拼接
        history_parts_for_prompt.append(truncate_text(final_part, 800))

    final_html = "\n\n".join(report_parts).strip()

    # ---------- title_maker ----------
    # 只生成 title 字段返回，不再把 title 注入正文，避免前端额外装饰
    title_ctx = {**ctx, "final_html": final_html}
    t_sys = render_file("reporting_partly/title_maker_llm_sys.txt", title_ctx)
    t_user = render_file("reporting_partly/title_maker_llm_user.txt", title_ctx)
    title = _clean_report_title(chat(t_sys, t_user, name="report_partly.title", temperature=0.3))

    return {
        "final_html": final_html,
        "final_html_parts": report_parts,
        "title": title or "数据分析报告",
    }


def _parse_toc_list(text: str) -> list:
    """
    兼容：
    - JSON array<object>
    - markdown/code block 包裹的 JSON
    - 普通纯文本逐行目录
    """
    if not text:
        return []

    t = text.strip()

    try:
        parsed = json.loads(t)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        inner = "\n".join(lines)
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            t = inner

    out = []
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^[\s\-\*•]+", "", line)
        if cleaned:
            out.append(cleaned)

    return out


def _unwrap_code_block(text: str) -> str:
    if not text:
        return ""

    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    return t


def _starts_with_markdown_heading(text: str) -> bool:
    if not text:
        return False
    return bool(re.match(r"^\s*#{1,6}\s+", text))


def _extract_section_title(section: Any) -> str:
    if isinstance(section, dict):
        num = str(section.get("num", "")).strip()
        raw_title = str(section.get("title", "")).strip()
        return f"{num} {raw_title}".strip()
    return str(section).strip()


def _strip_redundant_heading(text: str, section: Any) -> str:
    """
    去掉开头重复的标题：
    - '## 1 数据加载...\\n\\n## 1 数据加载...'
    - '1 数据加载...\\n1 数据加载...'
    """
    if not text:
        return ""

    lines = [line for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)

    if not lines:
        return ""

    expected_title = _extract_section_title(section)
    expected_title_norm = _normalize_heading_text(expected_title)

    # 取前两条非空行看看是否重复
    non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
    if len(non_empty_indices) >= 2:
        first_idx = non_empty_indices[0]
        second_idx = non_empty_indices[1]
        first_norm = _normalize_heading_text(lines[first_idx])
        second_norm = _normalize_heading_text(lines[second_idx])

        if first_norm and second_norm and first_norm == second_norm:
            del lines[second_idx]

    # 若第一行和预期标题一致，而第二行也是同标题，也删除第二个
    non_empty_lines = [line for line in lines if line.strip()]
    if len(non_empty_lines) >= 2:
        first_norm = _normalize_heading_text(non_empty_lines[0])
        second_norm = _normalize_heading_text(non_empty_lines[1])
        if expected_title_norm and first_norm == expected_title_norm and second_norm == expected_title_norm:
            removed = False
            new_lines = []
            seen_same = 0
            for line in lines:
                if line.strip() and _normalize_heading_text(line) == expected_title_norm:
                    seen_same += 1
                    if seen_same == 2 and not removed:
                        removed = True
                        continue
                new_lines.append(line)
            lines = new_lines

    return "\n".join(lines).strip()


def _normalize_heading_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^\s*#{1,6}\s*", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()
