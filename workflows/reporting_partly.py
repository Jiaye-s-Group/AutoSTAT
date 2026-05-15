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
from typing import Any, Callable

from core.llm_client import chat
from core.prompt_template import render_file
from workflow.report.report_content_utils import (
    normalize_part,
    normalize_for_dedup,
    sanitize_section_heading_text,
    wrap_section_as_markdown,
    build_history_context,
    truncate_text,
)

FIG_PLACEHOLDER_CAPTURE_RE = re.compile(
    r"(?<![A-Za-z0-9_])[\[\uFF3B\u3010]?\s*FIG\s*[:\uFF1A]?\s*(\d+)\s*[\]\uFF3D\u3011]?(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)


class ReportGenerationCancelled(RuntimeError):
    """Raised when a newer report generation supersedes this workflow."""


def _is_report_cancelled(cancel_check: Callable[[], bool] | None) -> bool:
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception:
        return False


def _raise_if_report_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if _is_report_cancelled(cancel_check):
        raise ReportGenerationCancelled()


def _clean_report_title(raw_title: Any) -> str:
    if raw_title is None:
        return ""

    if isinstance(raw_title, dict):
        for key in ("title", "标题", "题目", "text", "name", "label", "content"):
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

    json_title_match = re.search(r'"(?:title|标题|题目)"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if json_title_match:
        return json_title_match.group(1).strip()

    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r'^[\s`"\']+', "", text)
    text = re.sub(r'[\s`"\']+$', "", text)
    text = re.sub(r"^[《【「『]+", "", text)
    text = re.sub(r"[》】」』]+$", "", text)
    return text.strip()


def _extract_fig_numbers_from_text(text: Any) -> list[int]:
    figures: list[int] = []
    seen: set[int] = set()
    for match in FIG_PLACEHOLDER_CAPTURE_RE.finditer(str(text or "")):
        try:
            figure = int(match.group(1))
        except Exception:
            continue
        if figure not in seen:
            seen.add(figure)
            figures.append(figure)
    return figures


def _coerce_figures(value: Any) -> list[int]:
    figures: list[int] = []
    seen: set[int] = set()

    def add(raw: Any) -> None:
        if raw is None:
            return
        if isinstance(raw, bool):
            return
        if isinstance(raw, int):
            candidates = [raw]
        elif isinstance(raw, float) and raw.is_integer():
            candidates = [int(raw)]
        elif isinstance(raw, str):
            candidates = _extract_fig_numbers_from_text(raw)
            if not candidates:
                candidates = [int(item) for item in re.findall(r"\d+", raw)]
        elif isinstance(raw, (list, tuple, set)):
            for item in raw:
                add(item)
            return
        else:
            return

        for candidate in candidates:
            if candidate < 0 or candidate in seen:
                continue
            seen.add(candidate)
            figures.append(candidate)

    add(value)
    return figures


def _section_figures(section: Any) -> list[int]:
    if not isinstance(section, dict):
        return []

    for key in ("figures", "figs", "figure", "图片", "图号"):
        figures = _coerce_figures(section.get(key))
        if figures:
            return figures
    return []


def _toc_has_figures(toc_list: list) -> bool:
    return any(_section_figures(section) for section in toc_list)


def _normalize_toc_figures(toc_list: list) -> list:
    normalized: list[Any] = []
    for section in toc_list:
        if not isinstance(section, dict):
            normalized.append(section)
            continue
        section_copy = dict(section)
        section_copy["figures"] = _section_figures(section_copy)
        normalized.append(section_copy)
    return normalized


def _merge_toc_figures(toc_list: list, source_toc_list: list) -> list:
    if not toc_list or not source_toc_list:
        return _normalize_toc_figures(toc_list)

    merged: list[Any] = []
    for index, section in enumerate(toc_list):
        source_section = source_toc_list[index] if index < len(source_toc_list) else None
        source_figures = _section_figures(source_section)

        if isinstance(section, dict):
            section_copy = dict(section)
            if not _section_figures(section_copy) and source_figures:
                section_copy["figures"] = source_figures
            else:
                section_copy["figures"] = _section_figures(section_copy)
            merged.append(section_copy)
            continue

        if source_figures and isinstance(source_section, dict):
            merged.append(dict(source_section))
        else:
            merged.append(section)

    return merged


def _parse_toc_text(toc_text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for raw_line in str(toc_text or "").replace("\\r\\n", "\n").replace("\\n", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line = re.sub(r"^\s*#{1,6}\s*", "", line)
        line = re.sub(r"^[\s\-\*•]+", "", line).strip()
        match = re.match(r"^(\d+(?:[\.．]\d+)*)(?:[\.．、]|\s+)?\s*(.*)$", line)
        if not match:
            continue

        num = match.group(1).replace("．", ".").strip()
        remainder = match.group(2).strip()
        outline = ""

        outline_match = re.search(r"[（(]([^()（）]*)[）)]\s*$", remainder)
        if outline_match:
            outline = outline_match.group(1).strip()
            title = remainder[: outline_match.start()].strip()
        else:
            title = remainder.strip()

        title = sanitize_section_heading_text(title)
        if not title:
            continue

        sections.append(
            {
                "num": num,
                "title": title,
                "level": num.count(".") + 1,
                "outline": outline,
                "figures": [],
            }
        )

    return sections


def _section_identity(section: Any) -> tuple[str, str]:
    if isinstance(section, dict):
        num = str(section.get("num", "")).strip()
        title = sanitize_section_heading_text(section.get("title", ""))
        return num, normalize_for_dedup(title)

    text = sanitize_section_heading_text(section)
    match = re.match(r"^(\d+(?:[\.．]\d+)*)\s+(.+)$", text)
    if match:
        return match.group(1).replace("．", ".").strip(), normalize_for_dedup(match.group(2))
    return "", normalize_for_dedup(text)


def _merge_toc_with_authoritative_order(primary_toc: list, authoritative_toc: list) -> list:
    if not authoritative_toc:
        return primary_toc
    if not primary_toc:
        return authoritative_toc

    by_num: dict[str, Any] = {}
    by_title: dict[str, Any] = {}
    used_ids: set[int] = set()

    for section in primary_toc:
        num, title_key = _section_identity(section)
        if num:
            by_num.setdefault(num, section)
        if title_key:
            by_title.setdefault(title_key, section)

    merged: list[Any] = []
    for fallback_section in authoritative_toc:
        num, title_key = _section_identity(fallback_section)
        matched = by_num.get(num) if num else None
        if matched is None and title_key:
            matched = by_title.get(title_key)

        if isinstance(fallback_section, dict):
            section_out = dict(fallback_section)
            if isinstance(matched, dict):
                used_ids.add(id(matched))
                for key, value in matched.items():
                    if key in {"num", "title", "level", "outline"} and section_out.get(key):
                        continue
                    section_out[key] = value
                for key in ("num", "title", "level", "outline"):
                    if fallback_section.get(key):
                        section_out[key] = fallback_section[key]
            elif matched is not None:
                used_ids.add(id(matched))
            merged.append(section_out)
        else:
            if matched is not None:
                used_ids.add(id(matched))
                merged.append(matched)
            else:
                merged.append(fallback_section)

    for section in primary_toc:
        if id(section) not in used_ids:
            merged.append(section)

    return merged


def _ensure_visual_fig_placeholders(text: str) -> str:
    text = normalize_part(text or "")
    if not text or _extract_fig_numbers_from_text(text):
        return text

    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not blocks:
        return text

    return "\n\n".join(f"[FIG:{index}] {block}" for index, block in enumerate(blocks))


def _contains_all_figures(text: str, figures: list[int]) -> bool:
    present = set(_extract_fig_numbers_from_text(text))
    return all(figure in present for figure in figures)


def _dedupe_required_figures(text: str, figures: list[int]) -> str:
    if not text or not figures:
        return text

    expected = set(figures)
    seen: set[int] = set()

    def replace(match: re.Match[str]) -> str:
        try:
            figure = int(match.group(1))
        except Exception:
            return match.group(0)
        if figure not in expected:
            return match.group(0)
        if figure in seen:
            return ""
        seen.add(figure)
        return f"[FIG:{figure}]"

    return FIG_PLACEHOLDER_CAPTURE_RE.sub(replace, text)


def _preserve_required_figures(primary: str, fallback: str, section: Any) -> str:
    figures = _section_figures(section)
    primary = normalize_part(primary)
    fallback = normalize_part(fallback)
    if not figures:
        return primary

    if _contains_all_figures(primary, figures):
        return normalize_part(_dedupe_required_figures(primary, figures))

    if fallback and _contains_all_figures(fallback, figures):
        return normalize_part(_dedupe_required_figures(fallback, figures))

    base = primary or fallback
    present = set(_extract_fig_numbers_from_text(base))
    missing = [figure for figure in figures if figure not in present]
    if not missing:
        return normalize_part(_dedupe_required_figures(base, figures))

    suffix = " ".join(f"[FIG:{figure}]" for figure in missing)
    if not base:
        return suffix

    base = base.rstrip()
    if not re.search(r"[。！？!?\.]$", base):
        base += "。"
    return normalize_part(_dedupe_required_figures(f"{base} {suffix}", figures))


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
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    selected_full_conten = _ensure_visual_fig_placeholders(selected_full_conten or "")
    authoritative_toc = _parse_toc_text(toc_text)
    ctx: dict[str, Any] = {
        "toc_text": toc_text or "",
        "toc_md": toc_text or "",
        "selected_full_conten": selected_full_conten,
        "selected_full_contents_vis": selected_full_conten,
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
    _raise_if_report_cancelled(cancel_check)
    sp_sys = render_file("reporting_partly/selected_photo_update_toc_llm_sys.txt", ctx)
    sp_user = render_file("reporting_partly/selected_photo_update_toc_llm_user.txt", ctx)
    toc_list_raw = chat(sp_sys, sp_user, name="report_partly.select_photo").strip()
    _raise_if_report_cancelled(cancel_check)
    toc_list = _parse_toc_list(toc_list_raw)
    toc_list = _merge_toc_with_authoritative_order(toc_list, authoritative_toc)
    toc_list = _normalize_toc_figures(toc_list)
    ctx["toc_list"] = toc_list

    # ---------- 节点 2: update_toc_with_relevant_sections ----------
    _raise_if_report_cancelled(cancel_check)
    up_sys = render_file("reporting_partly/update_toc_with_relevant_sections_llm_sys.txt", ctx)
    up_user = render_file("reporting_partly/update_toc_with_relevant_sections_llm_user.txt", ctx)
    toc_final_raw = chat(up_sys, up_user, name="report_partly.update_toc").strip()
    _raise_if_report_cancelled(cancel_check)
    toc_list_final = _parse_toc_list(toc_final_raw)
    toc_list_final = _merge_toc_figures(toc_list_final, toc_list)
    toc_list_final = _merge_toc_with_authoritative_order(toc_list_final, toc_list or authoritative_toc)

    if not toc_list_final:
        toc_list_final = toc_list
    elif _toc_has_figures(toc_list) and not _toc_has_figures(toc_list_final):
        toc_list_final = toc_list

    if not toc_list_final:
        toc_list_final = [
            line.strip() for line in (toc_text or "").splitlines() if line.strip()
        ]

    report_parts: list[str] = []
    seen_parts: set[str] = set()
    history_parts_for_prompt: list[str] = []

    for idx, section in enumerate(toc_list_final):
        _raise_if_report_cancelled(cancel_check)
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
        _raise_if_report_cancelled(cancel_check)
        content = _unwrap_code_block(content)
        content = normalize_part(content)
        content = _preserve_required_figures(content, "", section)
        section_ctx["content"] = content

        # ---------- fill_report ----------
        f_sys = render_file("reporting_partly/fill_report_llm_sys.txt", section_ctx)
        f_user = render_file("reporting_partly/fill_report_llm_user.txt", section_ctx)
        filled = chat(f_sys, f_user, name=f"report_partly.fill.{idx+1}").strip()
        _raise_if_report_cancelled(cancel_check)
        filled = _unwrap_code_block(filled)
        filled = normalize_part(filled)
        filled = _preserve_required_figures(filled, content, section)

        # debug：检查图号在哪一层还存在
        print(f"[REPORT][SECTION {idx+1}] HAS_FIG_IN_FILLED =", bool(_extract_fig_numbers_from_text(filled)))
        print(f"[REPORT][SECTION {idx+1}] FILLED_PREVIEW =", filled[:300])

        # 最终正文优先用 filled，再退回 content
        final_part = filled or content
        final_part = normalize_part(final_part)
        final_part = _preserve_required_figures(final_part, content, section)

        print(f"[REPORT][SECTION {idx+1}] HAS_FIG_IN_FINAL =", bool(_extract_fig_numbers_from_text(final_part)))

        # 关键修复：去掉模型自己写在正文开头的章节标题，最终统一以目录标题为准
        original_final_part = final_part
        final_part = _strip_redundant_heading(final_part, section)
        final_part = normalize_part(final_part)
        if original_final_part and not final_part:
            final_part = original_final_part
        final_part = _preserve_required_figures(final_part, original_final_part, section)

        dedup_key = normalize_for_dedup(_extract_section_title(section))
        if dedup_key in seen_parts:
            continue

        seen_parts.add(dedup_key)

        wrapped_part = wrap_section_as_markdown(section, final_part)

        wrapped_part = normalize_part(wrapped_part)
        if wrapped_part:
            report_parts.append(wrapped_part)

        # history 只给下一轮 writer 看，不参与最终成品拼接
        history_parts_for_prompt.append(truncate_text(final_part, 800))

    final_html = "\n\n".join(report_parts).strip()

    # ---------- title_maker ----------
    # 只生成 title 字段返回，不再把 title 注入正文，避免前端额外装饰
    _raise_if_report_cancelled(cancel_check)
    title_ctx = {**ctx, "final_html": final_html}
    t_sys = render_file("reporting_partly/title_maker_llm_sys.txt", title_ctx)
    t_user = render_file("reporting_partly/title_maker_llm_user.txt", title_ctx)
    title = _clean_report_title(chat(t_sys, t_user, name="report_partly.title", temperature=0.3))
    _raise_if_report_cancelled(cancel_check)

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
        raw_title = sanitize_section_heading_text(section.get("title", ""))
        return f"{num} {raw_title}".strip()
    return sanitize_section_heading_text(section)


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

    first_non_empty_idx = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first_non_empty_idx is not None:
        first_line_norm = _normalize_heading_text(lines[first_non_empty_idx])
        if expected_title_norm and first_line_norm == expected_title_norm:
            del lines[first_non_empty_idx]

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
    t = sanitize_section_heading_text(t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()
