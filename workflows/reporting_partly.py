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
    normalize_figure_placeholders,
    remove_figure_placeholders,
)

FIG_PLACEHOLDER_CAPTURE_RE = re.compile(
    r"(?<![A-Za-z0-9_])[\[\uFF3B\u3010]?\s*FIG\s*[:\uFF1A]?\s*(\d+)\s*[\]\uFF3D\u3011]?(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)
FIG_REFERENCE_PHRASE_RE = re.compile(
    r"(?:(?:另|并)?(?:如|见|参见|详见|参考|根据|结合|从))?\s*"
    r"(?<![A-Za-z0-9_])[\[\uFF3B\u3010]?\s*FIG\s*[:\uFF1A]?\s*(\d+)\s*[\]\uFF3D\u3011]?(?![A-Za-z0-9_])"
    r"\s*(?:所示|可见|可以看出|显示|展示)?\s*[，,、:：；;]?",
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
    normalized_text = normalize_figure_placeholders(str(text or ""))
    for match in FIG_PLACEHOLDER_CAPTURE_RE.finditer(normalized_text):
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


def _toc_assigned_figures(toc_list: list) -> list[int]:
    figures: list[int] = []
    seen: set[int] = set()
    for section in toc_list or []:
        for figure in _section_figures(section):
            if figure in seen:
                continue
            seen.add(figure)
            figures.append(figure)
    return figures


def _log_toc_figure_state(label: str, toc_list: list) -> None:
    assigned_figures = _toc_assigned_figures(toc_list)
    print(
        f"[REPORT][TOC_FIG] {label}: sections={len(toc_list or [])}, "
        f"assigned_figures={assigned_figures}"
    )


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


def _toc_section_level(section: Any) -> int:
    if isinstance(section, dict):
        try:
            return int(section.get("level", 1))
        except Exception:
            num = str(section.get("num", "")).strip()
            return num.count(".") + 1 if num else 1

    match = re.match(r"^\s*(\d+(?:[\.．]\d+)*)", str(section or ""))
    return match.group(1).replace("．", ".").count(".") + 1 if match else 1


def _toc_section_num(section: Any) -> str:
    if isinstance(section, dict):
        return str(section.get("num", "")).replace("．", ".").strip()
    match = re.match(r"^\s*(\d+(?:[\.．]\d+)*)", str(section or ""))
    return match.group(1).replace("．", ".").strip() if match else ""


def _toc_section_text(section: Any) -> str:
    if isinstance(section, dict):
        return " ".join(
            str(section.get(key, "") or "")
            for key in ("title", "outline", "desc", "description")
        )
    return str(section or "")


def _toc_section_has_child(toc_list: list, index: int) -> bool:
    current_num = _toc_section_num(toc_list[index])
    if not current_num:
        return False
    prefix = f"{current_num}."
    return any(_toc_section_num(section).startswith(prefix) for section in toc_list[index + 1 :])


def _toc_section_has_visual_ancestor(toc_list: list, index: int) -> bool:
    current_num = _toc_section_num(toc_list[index])
    if not current_num or "." not in current_num:
        return False

    ancestor_nums = {
        ".".join(current_num.split(".")[:level])
        for level in range(1, len(current_num.split(".")))
    }
    strong_keywords = ("可视化", "图表", "图像", "图片", "可视分析")
    for section in toc_list[:index]:
        if _toc_section_num(section) not in ancestor_nums:
            continue
        ancestor_text = re.sub(r"\s+", "", _toc_section_text(section))
        if any(keyword in ancestor_text for keyword in strong_keywords):
            return True
    return False


def _visual_toc_section_score(toc_list: list, index: int) -> int:
    section = toc_list[index]
    text = _toc_section_text(section)
    text_no_space = re.sub(r"\s+", "", text)

    negative_keywords = ("结论", "总结", "展望", "建议", "数据加载", "数据导入", "预处理", "建模", "模型")
    if any(keyword in text_no_space for keyword in negative_keywords):
        score = -20
    else:
        score = 0

    strong_keywords = ("可视化", "图表", "图像", "图片", "可视分析")
    analysis_keywords = ("分布", "趋势", "关系", "相关", "对比", "比较", "差异", "特征", "占比")
    if any(keyword in text_no_space for keyword in strong_keywords):
        score += 30
    if any(keyword in text_no_space for keyword in analysis_keywords):
        score += 12

    if isinstance(section, dict):
        modules = section.get("modules")
        if isinstance(modules, list) and 2 in modules:
            score += 18

    if _toc_section_has_visual_ancestor(toc_list, index):
        score += 20
    score += _toc_section_level(section) * 2
    if not _toc_section_has_child(toc_list, index):
        score += 5
    return score


def _apply_toc_figure_fallback(toc_list: list, figures: list[int]) -> list:
    if not toc_list or not figures:
        return toc_list

    normalized = _normalize_toc_figures(toc_list)
    assigned = set(_toc_assigned_figures(normalized))
    missing = [figure for figure in figures if figure not in assigned]
    if not missing:
        return normalized

    scored_indices = [
        (index, _visual_toc_section_score(normalized, index))
        for index in range(len(normalized))
    ]
    scored_indices.sort(key=lambda item: item[1], reverse=True)
    if not scored_indices or scored_indices[0][1] <= 0:
        print(f"[REPORT][TOC_FIG] fallback skipped: no visual section for missing={missing}")
        return normalized

    target_index, target_score = scored_indices[0]
    target_section = normalized[target_index]
    target_figures = _section_figures(target_section)
    merged_figures = target_figures + [figure for figure in missing if figure not in target_figures]

    if isinstance(target_section, dict):
        updated_section = dict(target_section)
        updated_section["figures"] = merged_figures
    else:
        updated_section = {
            "num": _toc_section_num(target_section),
            "title": sanitize_section_heading_text(target_section),
            "level": _toc_section_level(target_section),
            "outline": "",
            "figures": merged_figures,
        }

    normalized[target_index] = updated_section
    print(
        "[REPORT][TOC_FIG] fallback assigned missing figures "
        f"{missing} to section={_toc_section_text(updated_section)!r}, score={target_score}"
    )
    return normalized


def _dedupe_toc_figure_assignments(toc_list: list) -> list:
    if not toc_list:
        return toc_list

    normalized = _normalize_toc_figures(toc_list)
    owners: dict[int, tuple[tuple[int, int, int, int], int]] = {}

    for index, section in enumerate(normalized):
        figures = _section_figures(section)
        if not figures:
            continue

        score = (
            _toc_section_level(section),
            0 if _toc_section_has_child(normalized, index) else 1,
            _visual_toc_section_score(normalized, index),
            -index,
        )
        for figure in figures:
            current = owners.get(figure)
            if current is None or score > current[0]:
                owners[figure] = (score, index)

    duplicate_count = 0
    deduped: list[Any] = []
    for index, section in enumerate(normalized):
        figures = _section_figures(section)
        kept_figures = [
            figure
            for figure in figures
            if owners.get(figure, (None, None))[1] == index
        ]
        duplicate_count += max(0, len(figures) - len(kept_figures))

        if isinstance(section, dict):
            section_copy = dict(section)
            section_copy["figures"] = kept_figures
            deduped.append(section_copy)
        else:
            deduped.append(section)

    if duplicate_count:
        print(f"[REPORT][TOC_FIG] removed duplicate figure assignments: {duplicate_count}")
    return deduped


TITLE_MATCH_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "data",
    "chart",
    "figure",
    "plot",
    "analysis",
    "report",
    "result",
    "results",
    "distribution",
    "comparison",
    "visualization",
    "可视化",
    "分析",
    "报告",
    "数据",
    "结果",
    "图表",
    "图像",
    "图片",
    "分布",
    "比较",
    "对比",
    "关系",
    "趋势",
    "特征",
}


def _dynamic_match_text(text: Any) -> str:
    text = remove_figure_placeholders(normalize_figure_placeholders(str(text or "")))
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _compact_match_text(text: Any) -> str:
    return re.sub(r"[\W_]+", "", _dynamic_match_text(text), flags=re.UNICODE)


def _tokenize_dynamic_text(text: Any) -> set[str]:
    normalized = _dynamic_match_text(text)
    tokens: set[str] = set()

    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]{1,}", normalized):
        if token not in TITLE_MATCH_STOPWORDS:
            tokens.add(token)

    for seq in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        if seq not in TITLE_MATCH_STOPWORDS:
            tokens.add(seq)
        max_n = min(5, len(seq))
        for n in range(2, max_n + 1):
            for i in range(0, len(seq) - n + 1):
                gram = seq[i : i + n]
                if gram not in TITLE_MATCH_STOPWORDS:
                    tokens.add(gram)

    return tokens


def _toc_token_frequencies(toc_list: list) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for section in toc_list:
        for token in _tokenize_dynamic_text(_section_match_text(section)):
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def _weighted_dynamic_overlap_score(
    overlap: set[str],
    token_frequencies: dict[str, int],
    section_count: int,
    *,
    base_score: int,
) -> int:
    score = 0
    for token in overlap:
        frequency = max(token_frequencies.get(token, 1), 1)
        rarity_bonus = max(section_count - frequency + 1, 1)
        length_bonus = min(len(token), 6)
        score += base_score + rarity_bonus * 3 + length_bonus
    return score


def _is_informative_compact_match(text: str) -> bool:
    return len(text) >= 4 and text not in TITLE_MATCH_STOPWORDS


def _extract_figure_title(figure: int, figure_contexts: dict[int, str]) -> str:
    context = normalize_figure_placeholders(normalize_part(figure_contexts.get(figure, "")))
    if not context:
        return ""

    context_without_placeholder = remove_figure_placeholders(context).strip()
    for raw_line in context_without_placeholder.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        title_match = re.match(r"^(?:图题|标题|图表标题|title)\s*[:：]\s*(.+)$", line, flags=re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()

    first_line = next((line.strip() for line in context_without_placeholder.splitlines() if line.strip()), "")
    if not first_line:
        return ""

    first_sentence = re.split(r"[。.!！?？；;]\s*", first_line, maxsplit=1)[0].strip()
    return first_sentence[:80]


def _section_match_text(section: Any) -> str:
    base = _toc_section_text(section)
    if isinstance(section, dict):
        base = " ".join(
            str(section.get(key, "") or "")
            for key in ("num", "title", "outline", "desc", "description")
        )
    return base


def _figure_section_match_score(
    *,
    figure: int,
    figure_contexts: dict[int, str],
    toc_list: list,
    section_index: int,
) -> int:
    section = toc_list[section_index]
    figure_title = _extract_figure_title(figure, figure_contexts)
    figure_text = figure_contexts.get(figure, f"[FIG:{figure}]")
    title_tokens = _tokenize_dynamic_text(figure_title)
    figure_tokens = _tokenize_dynamic_text(figure_text)
    section_text = _section_match_text(section)
    section_tokens = _tokenize_dynamic_text(section_text)
    token_frequencies = _toc_token_frequencies(toc_list)
    section_count = max(len(toc_list), 1)
    modules = _section_modules(section)

    score = max(_visual_toc_section_score(toc_list, section_index), 0) // 3
    if 2 in modules:
        score += 12
    if figure in _section_figures(section):
        score += 8

    title_overlap = title_tokens & section_tokens
    context_overlap = (figure_tokens & section_tokens) - title_overlap
    score += _weighted_dynamic_overlap_score(
        title_overlap,
        token_frequencies,
        section_count,
        base_score=28,
    )
    score += _weighted_dynamic_overlap_score(
        context_overlap,
        token_frequencies,
        section_count,
        base_score=4,
    )

    compact_title = _compact_match_text(figure_title)
    compact_section = _compact_match_text(section_text)
    if (
        compact_title
        and compact_section
        and _is_informative_compact_match(compact_title)
        and _is_informative_compact_match(compact_section)
    ):
        if compact_title in compact_section or compact_section in compact_title:
            score += 80

    if _toc_section_has_visual_ancestor(toc_list, section_index):
        score += 8
    if not _toc_section_has_child(toc_list, section_index):
        score += 6
    score += _toc_section_level(section)

    if not title_overlap and not context_overlap and figure not in _section_figures(section):
        score -= 15

    return score


def _best_section_index_for_figure(
    *,
    figure: int,
    toc_list: list,
    figure_contexts: dict[int, str],
) -> int | None:
    if not toc_list:
        return None

    scored = [
        (
            _figure_section_match_score(
                figure=figure,
                figure_contexts=figure_contexts,
                toc_list=toc_list,
                section_index=index,
            ),
            index,
        )
        for index in range(len(toc_list))
    ]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    if scored:
        return scored[0][1]
    return None


def _assign_figures_to_best_sections(
    *,
    toc_list: list,
    figures: list[int],
    figure_contexts: dict[int, str],
) -> list:
    if not toc_list or not figures:
        return toc_list

    normalized = _normalize_toc_figures(toc_list)
    assignments: dict[int, list[int]] = {index: [] for index in range(len(normalized))}
    for figure in figures:
        target_index = _best_section_index_for_figure(
            figure=figure,
            toc_list=normalized,
            figure_contexts=figure_contexts,
        )
        if target_index is None:
            continue
        assignments[target_index].append(figure)

    assigned_count = sum(len(value) for value in assignments.values())
    reassigned: list[Any] = []
    for index, section in enumerate(normalized):
        if isinstance(section, dict):
            section_copy = dict(section)
            section_copy["figures"] = assignments.get(index, [])
            reassigned.append(section_copy)
        else:
            reassigned.append(section)

    print(
        "[REPORT][TOC_FIG] title assignment complete: "
        f"expected={figures}, assigned={assigned_count}"
    )
    return reassigned


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

    text = re.sub(r"^\s*#{1,6}\s*", "", str(section or "")).strip()
    text = sanitize_section_heading_text(text)
    match = re.match(r"^(\d+(?:[\.．]\d+)*)\s+(.+)$", text)
    if match:
        return match.group(1).replace("．", ".").strip(), normalize_for_dedup(match.group(2))
    return "", normalize_for_dedup(text)


def _merge_section_figures(target: Any, source: Any) -> Any:
    source_figures = _section_figures(source)
    if not source_figures:
        return target

    target_figures = _section_figures(target)
    merged_figures = target_figures + [figure for figure in source_figures if figure not in target_figures]

    if isinstance(target, dict):
        updated = dict(target)
        updated["figures"] = merged_figures
        return updated

    return {
        "num": _toc_section_num(target),
        "title": sanitize_section_heading_text(target),
        "level": _toc_section_level(target),
        "outline": "",
        "figures": merged_figures,
    }


def _dedupe_toc_sections(toc_list: list) -> list:
    deduped: list[Any] = []
    index_by_num: dict[str, int] = {}
    index_by_title: dict[str, int] = {}
    skipped = 0

    for section in toc_list or []:
        num, title_key = _section_identity(section)
        existing_index = None
        if num and num in index_by_num:
            existing_index = index_by_num[num]
        elif title_key and title_key in index_by_title:
            existing_index = index_by_title[title_key]

        if existing_index is not None:
            deduped[existing_index] = _merge_section_figures(deduped[existing_index], section)
            skipped += 1
            continue

        index = len(deduped)
        deduped.append(section)
        if num:
            index_by_num[num] = index
        if title_key:
            index_by_title[title_key] = index

    if skipped:
        print(f"[REPORT][TOC] removed duplicate toc sections before writing: {skipped}")
    return deduped


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

    skipped_extra = 0
    for section in primary_toc:
        if id(section) not in used_ids:
            num, title_key = _section_identity(section)
            if num or title_key:
                skipped_extra += 1

    if skipped_extra:
        print(f"[REPORT][TOC] ignored non-authoritative toc sections: {skipped_extra}")

    return _dedupe_toc_sections(merged)


def _ensure_visual_fig_placeholders(text: str) -> str:
    text = normalize_figure_placeholders(normalize_part(text or ""))
    if not text or _extract_fig_numbers_from_text(text):
        return text

    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not blocks:
        return text

    return "\n\n".join(f"[FIG:{index}] {block}" for index, block in enumerate(blocks))


def _extract_figure_contexts(text: str) -> dict[int, str]:
    text = normalize_figure_placeholders(normalize_part(text or ""))
    contexts: dict[int, str] = {}
    if not text:
        return contexts

    matches = list(FIG_PLACEHOLDER_CAPTURE_RE.finditer(text))
    if not matches:
        return contexts

    for index, match in enumerate(matches):
        try:
            figure = int(match.group(1))
        except Exception:
            continue

        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : next_start].strip()
        block = normalize_part(block)
        if not block:
            block = f"[FIG:{figure}]"
        contexts.setdefault(figure, block)
    return contexts


def _figure_context_with_placeholder(figure: int, figure_contexts: dict[int, str]) -> str:
    context = normalize_figure_placeholders(normalize_part(figure_contexts.get(figure, "")))
    if not context:
        return f"[FIG:{figure}]"

    context = _dedupe_required_figures(context, [figure])
    if _contains_all_figures(context, [figure]):
        return context

    context_text = remove_figure_placeholders(context).strip()
    return f"{context_text} [FIG:{figure}]".strip()


def _format_figure_contexts(figures: list[int], figure_contexts: dict[int, str]) -> str:
    parts = [
        _figure_context_with_placeholder(figure, figure_contexts)
        for figure in figures
    ]
    return "\n\n".join(part for part in parts if part)


def _strip_fig_placeholders_for_context(text: str) -> str:
    return normalize_part(remove_figure_placeholders(normalize_figure_placeholders(text or "")))


def _section_modules(section: Any) -> list[int]:
    if not isinstance(section, dict):
        return []

    modules = section.get("modules")
    if not isinstance(modules, list):
        return []

    normalized: list[int] = []
    for item in modules:
        try:
            module = int(item)
        except Exception:
            continue
        if module not in normalized:
            normalized.append(module)
    return normalized


def _build_section_reference_context(
    *,
    section: Any,
    ctx: dict[str, Any],
    figure_contexts: dict[int, str],
) -> str:
    figures = _section_figures(section)
    modules = _section_modules(section)
    parts: list[str] = []

    if 0 in modules and ctx.get("load_abstract"):
        parts.append(_strip_fig_placeholders_for_context(str(ctx.get("load_abstract", ""))))
    if 1 in modules and ctx.get("preproc_abstract"):
        parts.append(_strip_fig_placeholders_for_context(str(ctx.get("preproc_abstract", ""))))

    figure_context = _format_figure_contexts(figures, figure_contexts)
    if figure_context:
        parts.append(figure_context)
    elif 2 in modules and ctx.get("visual_abstract"):
        parts.append(_strip_fig_placeholders_for_context(str(ctx.get("visual_abstract", ""))))

    if 3 in modules and ctx.get("coding_abstract"):
        parts.append(_strip_fig_placeholders_for_context(str(ctx.get("coding_abstract", ""))))

    if not parts:
        fallback_context = ctx.get("selected_full_conten", "")
        if figures:
            parts.append(_format_figure_contexts(figures, figure_contexts))
        else:
            parts.append(_strip_fig_placeholders_for_context(str(fallback_context)))

    return "\n\n".join(part for part in parts if normalize_part(part)).strip()


def _contains_all_figures(text: str, figures: list[int]) -> bool:
    present = set(_extract_fig_numbers_from_text(normalize_figure_placeholders(text or "")))
    return all(figure in present for figure in figures)


def _dedupe_required_figures(text: str, figures: list[int]) -> str:
    if not text or not figures:
        return text

    text = normalize_figure_placeholders(text)
    expected = set(figures)
    seen: set[int] = set()

    def remove_disallowed_phrase(match: re.Match[str]) -> str:
        try:
            figure = int(match.group(1))
        except Exception:
            return match.group(0)
        if figure not in expected:
            return ""
        return match.group(0)

    text = FIG_REFERENCE_PHRASE_RE.sub(remove_disallowed_phrase, text)

    def replace(match: re.Match[str]) -> str:
        try:
            figure = int(match.group(1))
        except Exception:
            return match.group(0)
        if figure not in expected:
            return ""
        if figure in seen:
            return ""
        seen.add(figure)
        return f"[FIG:{figure}]"

    text = FIG_PLACEHOLDER_CAPTURE_RE.sub(replace, text)
    text = re.sub(r"\s+([，,。.!！?？；;：:、])", r"\1", text)
    text = re.sub(r"[，,、；;：:]\s*([。.!！?？])", r"\1", text)
    return text.strip()


def _log_final_figure_ledger(final_html: str, expected_figures: list[int]) -> None:
    refs: list[int] = []
    for match in FIG_PLACEHOLDER_CAPTURE_RE.finditer(normalize_figure_placeholders(final_html or "")):
        try:
            refs.append(int(match.group(1)))
        except Exception:
            continue

    expected = list(dict.fromkeys(expected_figures))
    ref_set = set(refs)
    duplicates = sorted({figure for figure in refs if refs.count(figure) > 1})
    missing = [figure for figure in expected if figure not in ref_set]
    unexpected = sorted(figure for figure in ref_set if figure not in set(expected))
    print(
        "[REPORT][FIG_LEDGER] "
        f"expected={expected}, final_refs={refs}, missing={missing}, "
        f"duplicates={duplicates}, unexpected={unexpected}"
    )


def _fig_refs_in_text(text: str) -> list[int]:
    refs: list[int] = []
    for match in FIG_PLACEHOLDER_CAPTURE_RE.finditer(normalize_figure_placeholders(text or "")):
        try:
            refs.append(int(match.group(1)))
        except Exception:
            continue
    return refs


def _find_report_part_index_for_section(report_sections: list[Any], section: Any) -> int | None:
    section_num, section_title = _section_identity(section)
    for index, existing_section in enumerate(report_sections):
        existing_num, existing_title = _section_identity(existing_section)
        if section_num and existing_num == section_num:
            return index
        if section_title and existing_title == section_title:
            return index
    return None


def _section_for_figure(toc_list: list, figure: int) -> Any | None:
    for section in toc_list:
        if figure in _section_figures(section):
            return section
    return None


def _ensure_all_figures_in_report_parts(
    *,
    report_parts: list[str],
    report_sections: list[Any],
    toc_list: list,
    expected_figures: list[int],
    figure_contexts: dict[int, str],
) -> list[str]:
    if not expected_figures:
        return report_parts

    existing_refs = set(_fig_refs_in_text("\n\n".join(report_parts)))
    missing = [figure for figure in expected_figures if figure not in existing_refs]
    if not missing:
        return report_parts

    print(f"[REPORT][FIG_LEDGER] repairing missing figure refs: {missing}")
    repaired = list(report_parts)
    repaired_sections = list(report_sections)

    for figure in missing:
        target_section = _section_for_figure(toc_list, figure)
        if target_section is None:
            target_index = _best_section_index_for_figure(
                figure=figure,
                toc_list=toc_list,
                figure_contexts=figure_contexts,
            )
            target_section = toc_list[target_index] if target_index is not None else None

        insertion_text = _figure_context_with_placeholder(figure, figure_contexts)
        if not insertion_text:
            insertion_text = f"[FIG:{figure}]"

        target_part_index = (
            _find_report_part_index_for_section(repaired_sections, target_section)
            if target_section is not None
            else None
        )

        if target_part_index is None and target_section is not None:
            repaired.append(normalize_part(wrap_section_as_markdown(target_section, insertion_text)))
            repaired_sections.append(target_section)
            continue

        if target_part_index is None:
            repaired.append(insertion_text)
            repaired_sections.append({"title": f"Figure {figure}", "figures": [figure]})
            continue

        repaired[target_part_index] = normalize_part(
            f"{repaired[target_part_index].rstrip()}\n\n{insertion_text}"
        )

    return repaired


def _preserve_required_figures(
    primary: str,
    fallback: str,
    section: Any,
    figure_contexts: dict[int, str] | None = None,
) -> str:
    figures = _section_figures(section)
    figure_contexts = figure_contexts or {}
    primary = normalize_figure_placeholders(normalize_part(primary))
    fallback = normalize_figure_placeholders(normalize_part(fallback))
    if not figures:
        return normalize_part(remove_figure_placeholders(primary))

    if _contains_all_figures(primary, figures):
        return normalize_part(_dedupe_required_figures(primary, figures))

    if fallback and _contains_all_figures(fallback, figures):
        return normalize_part(_dedupe_required_figures(fallback, figures))

    base = primary or fallback
    present = set(_extract_fig_numbers_from_text(_dedupe_required_figures(base, figures)))
    missing = [figure for figure in figures if figure not in present]
    if not missing:
        return normalize_part(_dedupe_required_figures(base, figures))

    suffix = "\n\n".join(
        _figure_context_with_placeholder(figure, figure_contexts)
        for figure in missing
    )
    if not base:
        return suffix

    base = _dedupe_required_figures(base, figures).rstrip()
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
    source_figures = _extract_fig_numbers_from_text(selected_full_conten)
    figure_contexts = _extract_figure_contexts(selected_full_conten)
    print(
        f"[REPORT][INPUT] partly selected_full_conten length={len(selected_full_conten)}, "
        f"fig_refs={source_figures}"
    )
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
    _log_toc_figure_state("toc_list before title assignment", toc_list)
    toc_list = _assign_figures_to_best_sections(
        toc_list=toc_list,
        figures=source_figures,
        figure_contexts=figure_contexts,
    )
    if not _toc_has_figures(toc_list):
        toc_list = _apply_toc_figure_fallback(toc_list, source_figures)
    toc_list = _dedupe_toc_figure_assignments(toc_list)
    _log_toc_figure_state("toc_list after title assignment", toc_list)
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

    _log_toc_figure_state("toc_list_final before title assignment", toc_list_final)
    toc_list_final = _assign_figures_to_best_sections(
        toc_list=toc_list_final,
        figures=source_figures,
        figure_contexts=figure_contexts,
    )
    if not _toc_has_figures(toc_list_final):
        toc_list_final = _apply_toc_figure_fallback(toc_list_final, source_figures)
    toc_list_final = _dedupe_toc_figure_assignments(toc_list_final)
    toc_list_final = _dedupe_toc_sections(toc_list_final)
    _log_toc_figure_state("toc_list_final after title assignment", toc_list_final)

    report_parts: list[str] = []
    report_part_sections: list[Any] = []
    seen_parts: set[str] = set()
    history_parts_for_prompt: list[str] = []

    for idx, section in enumerate(toc_list_final):
        _raise_if_report_cancelled(cancel_check)
        section_reference_context = _build_section_reference_context(
            section=section,
            ctx=ctx,
            figure_contexts=figure_contexts,
        )
        section_ctx: dict[str, Any] = {
            **ctx,
            "t": section,
            "section": section if isinstance(section, str) else json.dumps(section, ensure_ascii=False),
            "toc_section": section,
            "toc": json.dumps(toc_list_final, ensure_ascii=False),
            "toc_list_final": toc_list_final,
            "selected_full_conten": section_reference_context,
            "selected_full_contents_vis": section_reference_context,
            "section_fig_context": _format_figure_contexts(_section_figures(section), figure_contexts),
            "assigned_figures": _section_figures(section),
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
        content = _preserve_required_figures(content, "", section, figure_contexts)
        section_ctx["content"] = content

        # ---------- fill_report ----------
        f_sys = render_file("reporting_partly/fill_report_llm_sys.txt", section_ctx)
        f_user = render_file("reporting_partly/fill_report_llm_user.txt", section_ctx)
        filled = chat(f_sys, f_user, name=f"report_partly.fill.{idx+1}").strip()
        _raise_if_report_cancelled(cancel_check)
        filled = _unwrap_code_block(filled)
        filled = normalize_part(filled)
        filled = _preserve_required_figures(filled, content, section, figure_contexts)

        # debug：检查图号在哪一层还存在
        print(f"[REPORT][SECTION {idx+1}] HAS_FIG_IN_FILLED =", bool(_extract_fig_numbers_from_text(filled)))
        print(f"[REPORT][SECTION {idx+1}] FILLED_PREVIEW =", filled[:300])

        # 最终正文优先用 filled，再退回 content
        final_part = filled or content
        final_part = normalize_part(final_part)
        final_part = _preserve_required_figures(final_part, content, section, figure_contexts)

        print(f"[REPORT][SECTION {idx+1}] HAS_FIG_IN_FINAL =", bool(_extract_fig_numbers_from_text(final_part)))

        # 关键修复：去掉模型自己写在正文开头的章节标题，最终统一以目录标题为准
        original_final_part = final_part
        final_part = _strip_redundant_heading(final_part, section)
        final_part = normalize_part(final_part)
        if original_final_part and not final_part:
            final_part = original_final_part
        final_part = _preserve_required_figures(final_part, original_final_part, section, figure_contexts)

        dedup_key = normalize_for_dedup(_extract_section_title(section))
        if dedup_key in seen_parts:
            continue

        seen_parts.add(dedup_key)

        wrapped_part = wrap_section_as_markdown(section, final_part)

        wrapped_part = normalize_part(wrapped_part)
        if wrapped_part:
            report_parts.append(wrapped_part)
            report_part_sections.append(section)

        # history 只给下一轮 writer 看，不参与最终成品拼接
        history_parts_for_prompt.append(truncate_text(final_part, 800))

    report_parts = _ensure_all_figures_in_report_parts(
        report_parts=report_parts,
        report_sections=report_part_sections,
        toc_list=toc_list_final,
        expected_figures=source_figures,
        figure_contexts=figure_contexts,
    )
    final_html = "\n\n".join(report_parts).strip()
    _log_final_figure_ledger(final_html, source_figures)

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
        cleaned = re.sub(r"^\s*#{1,6}\s*", "", line)
        cleaned = re.sub(r"^[\s\-\*•]+", "", cleaned).strip()
        if not cleaned:
            continue
        parsed_lines = _parse_toc_text(cleaned)
        if len(parsed_lines) == 1:
            out.append(parsed_lines[0])

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
