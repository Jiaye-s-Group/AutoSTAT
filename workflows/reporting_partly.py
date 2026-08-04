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
import os
import re
from typing import Any, Callable

from core.figure_artifacts import figure_artifact_contexts, successful_figure_artifacts
from core.llm_client import chat
from core.prompt_template import render_file
from core.report_language import (
    REPORT_LANGUAGE_ZH,
    is_english_report,
    normalize_report_language,
    report_language_instruction,
    report_language_name,
)
from frontend.workflow.report.report_content_utils import (
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
NATURAL_FIGURE_REFERENCE_RE = re.compile(
    r"(?:(?:另|并)?(?:如|见|参见|详见|参考|根据|结合|从)\s*)?"
    r"(?:图表|图片|插图|图|Figure|Fig\.)\s*"
    r"(?:第\s*)?(?:\d+|[一二三四五六七八九十]+)"
    r"(?:\s*(?:所示|可见|可以看出|显示|展示|中|里))?\s*[，,、:：；;]?",
    flags=re.IGNORECASE,
)

# Limits are opt-in. Set the corresponding environment variables to positive
# integers when a deployment needs an explicit report-size guardrail.
DEFAULT_MAX_REPORT_FIGURES: int | None = None
DEFAULT_MAX_REPORT_FIGURES_PER_SECTION: int | None = None
DEFAULT_FIGURE_MATCH_CANDIDATE_SECTIONS = 5
DEFAULT_LLM_FIGURE_MATCH_MIN_CONFIDENCE = 0.45

CORE_REPORT_SECTION_TITLES = {
    "1": "数据加载与结构梳理",
    "2": "数据预处理",
    "3": "可视化分析",
    "4": "模型构建与评估",
    "5": "结论与应用展望",
}

CORE_REPORT_SECTION_TITLES_EN = {
    "1": "Data Loading and Structure Review",
    "2": "Data Preprocessing",
    "3": "Visualization Analysis",
    "4": "Modeling and Evaluation",
    "5": "Conclusions and Practical Implications",
}


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


def _emit_report_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    payload: dict[str, Any],
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(payload)
    except Exception as exc:
        print("[REPORT][PROGRESS] callback failed:", repr(exc))


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


def _clear_toc_figures(toc_list: list) -> list:
    cleared: list[Any] = []
    for section in toc_list or []:
        if isinstance(section, dict):
            section_copy = dict(section)
            section_copy["figures"] = []
            cleared.append(section_copy)
        else:
            cleared.append(section)
    return cleared


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


def _toc_section_title_text(section: Any) -> str:
    if isinstance(section, dict):
        num = _toc_section_num(section)
        title = str(section.get("title", "") or "").strip()
        return f"{num} {title}".strip()
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
    text_no_space = re.sub(r"\s+", "", text).lower()

    negative_keywords = (
        "结论", "总结", "展望", "建议", "数据加载", "数据导入", "预处理", "建模", "模型",
        "conclusion", "summary", "implication", "recommendation", "dataloading",
        "dataimport", "preprocessing", "modeling", "model",
    )
    if any(keyword in text_no_space for keyword in negative_keywords):
        score = -20
    else:
        score = 0

    strong_keywords = (
        "可视化", "图表", "图像", "图片", "可视分析",
        "visualization", "chart", "figure", "image", "plot",
    )
    analysis_keywords = (
        "分布", "趋势", "关系", "相关", "对比", "比较", "差异", "特征", "占比",
        "distribution", "trend", "relationship", "association", "correlation",
        "comparison", "difference", "feature", "proportion",
    )
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


def _is_visualization_report_section(toc_list: list, index: int) -> bool:
    if index < 0 or index >= len(toc_list):
        return False
    section = toc_list[index]
    text = re.sub(r"\s+", "", _toc_section_text(section)).lower()
    strong_keywords = (
        "可视化", "图表", "图像", "图片", "可视分析",
        "visualization", "visualisation", "chart", "figure", "image", "plot",
    )
    if any(keyword in text for keyword in strong_keywords):
        return True
    if _toc_section_has_visual_ancestor(toc_list, index):
        return True
    modules = _section_modules(section)
    return 2 in modules


def _figure_section_scope(figure: int, figure_artifacts: list[dict[str, Any]] | None) -> str:
    if not figure_artifacts or figure < 0 or figure >= len(figure_artifacts):
        return ""
    artifact = figure_artifacts[figure]
    if not isinstance(artifact, dict):
        return ""
    return str(artifact.get("section_scope") or artifact.get("stage") or "").strip().lower()


def _figure_allowed_in_section(
    *,
    figure: int,
    toc_list: list,
    section_index: int,
    figure_artifacts: list[dict[str, Any]] | None,
) -> bool:
    scope = _figure_section_scope(figure, figure_artifacts)
    if scope in {"visualization", "visualisation", "visual", "viz"}:
        visual_sections = {
            index
            for index in range(len(toc_list))
            if _is_visualization_report_section(toc_list, index)
        }
        if visual_sections:
            return section_index in visual_sections
    return True


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


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return value if value > 0 else default


def _report_figure_limit_env(name: str, default: int | None) -> int | None:
    """Read an optional figure limit; missing or zero means unlimited."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    if value == 0:
        return None
    return value if value > 0 else default


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except Exception:
        return default
    return value if value > 0 else default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _figure_context_filter_score(figure: int, figure_contexts: dict[int, str]) -> int:
    """Apply only hard negative filters to invalid or failed figure contexts."""
    context = normalize_part(figure_contexts.get(figure, ""))
    if not context:
        return -40

    text = remove_figure_placeholders(context).strip()
    lowered = text.lower()
    if len(text) < 40:
        return -20

    weak_markers = (
        "无法", "未能", "失败", "错误", "无有效", "没有生成", "空图",
        "error", "failed", "cannot", "unable", "no valid",
    )
    if any(marker in text or marker in lowered for marker in weak_markers):
        return -45

    return 0


def _figure_redundancy_penalty(
    *,
    figure: int,
    selected: list[int],
    figure_contexts: dict[int, str],
) -> int:
    if not selected:
        return 0

    tokens = _tokenize_dynamic_text(figure_contexts.get(figure, ""))
    if not tokens:
        return 0

    penalty = 0
    for selected_figure in selected:
        selected_tokens = _tokenize_dynamic_text(figure_contexts.get(selected_figure, ""))
        if not selected_tokens:
            continue
        overlap = len(tokens & selected_tokens)
        union = len(tokens | selected_tokens)
        if union <= 0:
            continue
        jaccard = overlap / union
        if jaccard >= 0.72:
            penalty = max(penalty, 35)
        elif jaccard >= 0.55:
            penalty = max(penalty, 18)
    return penalty


def _is_pipeline_setup_section(section: Any) -> bool:
    title = re.sub(r"\s+", "", _toc_section_title_text(section))
    if not title:
        return False

    setup_keywords = (
        "数据加载", "数据导入", "结构梳理", "字段说明", "数据概览",
        "数据预处理", "预处理", "清洗", "编码", "标准化", "归一化",
    )
    analysis_keywords = (
        "可视化", "图表", "分布分析", "相关性", "模型", "评估", "结果", "性能",
    )
    return any(keyword in title for keyword in setup_keywords) and not any(
        keyword in title for keyword in analysis_keywords
    )


def _is_modeling_section(section: Any) -> bool:
    title = re.sub(r"\s+", "", _toc_section_title_text(section))
    if not title:
        return False

    modeling_keywords = (
        "模型", "建模", "分类器", "回归器", "预测", "评估", "性能",
        "准确率", "精确率", "召回率", "混淆矩阵", "AUC", "ROC",
        "RMSE", "MAE", "F1", "特征重要性", "残差",
    )
    return any(keyword.lower() in title.lower() for keyword in modeling_keywords)


def _figure_is_modeling_result(figure: int, figure_contexts: dict[int, str]) -> bool:
    text = re.sub(
        r"\s+",
        "",
        _dynamic_match_text(figure_contexts.get(figure, "")),
    )
    if not text:
        return False

    modeling_keywords = (
        "模型", "建模", "分类器", "回归器", "预测结果", "预测值",
        "准确率", "精确率", "召回率", "混淆矩阵", "auc", "roc",
        "rmse", "mae", "f1", "特征重要性", "残差", "学习曲线",
        "交叉验证", "svm", "randomforest", "logisticregression",
        "linearregression", "xgboost", "lightgbm",
    )
    return any(keyword in text for keyword in modeling_keywords)


def _has_strong_figure_section_anchor(figure_title: str, section: Any) -> bool:
    section_title = _toc_section_title_text(section)
    compact_title = _compact_match_text(figure_title)
    compact_section_title = _compact_match_text(section_title)
    if (
        compact_title
        and compact_section_title
        and _is_informative_compact_match(compact_title)
        and _is_informative_compact_match(compact_section_title)
        and (compact_title in compact_section_title or compact_section_title in compact_title)
    ):
        return True

    overlap = _tokenize_dynamic_text(figure_title) & _tokenize_dynamic_text(section_title)
    return any(len(token) >= 3 for token in overlap) or len(overlap) >= 2


def _select_report_figures(
    *,
    candidate_figures: list[int],
    figure_contexts: dict[int, str],
    toc_list: list,
) -> list[int]:
    """Choose a bounded candidate pool. Final insertion is decided by pair planning."""
    candidates = list(dict.fromkeys(candidate_figures))
    if not candidates:
        return []

    max_figures = _report_figure_limit_env("AUTOSTAT_REPORT_MAX_FIGURES", DEFAULT_MAX_REPORT_FIGURES)
    max_per_section = _report_figure_limit_env(
        "AUTOSTAT_REPORT_MAX_FIGURES_PER_SECTION",
        DEFAULT_MAX_REPORT_FIGURES_PER_SECTION,
    )
    if max_figures is None or len(candidates) <= max_figures:
        limit_label = "unlimited" if max_figures is None else str(max_figures)
        print(f"[REPORT][FIG_SELECT] keep all figures (max_figures={limit_label}): {candidates}")
        return candidates

    normalized_toc = _normalize_toc_figures(toc_list or [])
    records: list[dict[str, Any]] = []
    for position, figure in enumerate(candidates):
        best_index = _best_section_index_for_figure(
            figure=figure,
            toc_list=normalized_toc,
            figure_contexts=figure_contexts,
        )
        match_score = 0
        if best_index is not None:
            match_score = _figure_section_match_score(
                figure=figure,
                figure_contexts=figure_contexts,
                toc_list=normalized_toc,
                section_index=best_index,
            )

        filter_score = _figure_context_filter_score(figure, figure_contexts)
        total_score = match_score + filter_score
        records.append(
            {
                "figure": figure,
                "position": position,
                "section": best_index,
                "match_score": match_score,
                "filter_score": filter_score,
                "score": total_score,
            }
        )

    ranked = sorted(records, key=lambda item: (item["score"], -item["position"]), reverse=True)
    selected: list[int] = []
    selected_set: set[int] = set()
    per_section: dict[int | None, int] = {}

    for item in ranked:
        if len(selected) >= max_figures:
            break
        section = item["section"]
        if max_per_section is not None and per_section.get(section, 0) >= max_per_section:
            continue
        figure = int(item["figure"])
        adjusted_score = int(item["score"]) - _figure_redundancy_penalty(
            figure=figure,
            selected=selected,
            figure_contexts=figure_contexts,
        )
        if adjusted_score < 0 and len(selected) >= max(3, max_figures // 3):
            continue
        selected.append(figure)
        selected_set.add(figure)
        per_section[section] = per_section.get(section, 0) + 1

    if len(selected) < max_figures:
        for item in ranked:
            if len(selected) >= max_figures:
                break
            figure = int(item["figure"])
            if figure in selected_set:
                continue
            section = item["section"]
            if max_per_section is not None and per_section.get(section, 0) >= max_per_section:
                continue
            if _figure_redundancy_penalty(
                figure=figure,
                selected=selected,
                figure_contexts=figure_contexts,
            ) >= 35:
                continue
            selected.append(figure)
            selected_set.add(figure)
            per_section[section] = per_section.get(section, 0) + 1

    if not selected:
        selected = [int(item["figure"]) for item in ranked[:max_figures]]
        selected_set = set(selected)

    selected.sort(key=lambda figure: candidates.index(figure))
    dropped = [figure for figure in candidates if figure not in selected_set]
    score_debug = {
        int(item["figure"]): {
            "match": int(item["match_score"]),
            "filter": int(item["filter_score"]),
            "total": int(item["score"]),
            "section": item["section"],
        }
        for item in records
    }
    print(
        "[REPORT][FIG_SELECT] "
        f"candidates={candidates}, selected={selected}, dropped={dropped}, "
        f"max_figures={max_figures}, max_per_section={max_per_section}, "
        f"scores={score_debug}"
    )
    return selected


def _figure_section_alignment_score(
    *,
    figure: int,
    figure_contexts: dict[int, str],
    toc_list: list,
    section_index: int,
    figure_artifacts: list[dict[str, Any]] | None = None,
) -> int:
    section = toc_list[section_index]
    if not _figure_allowed_in_section(
        figure=figure,
        toc_list=toc_list,
        section_index=section_index,
        figure_artifacts=figure_artifacts,
    ):
        return -100
    if _is_pipeline_setup_section(section):
        return -100
    if _is_modeling_section(section) and not _figure_is_modeling_result(figure, figure_contexts):
        return -100

    figure_title = _extract_figure_title(figure, figure_contexts)
    figure_text = figure_contexts.get(figure, f"[FIG:{figure}]")
    section_text = _section_match_text(section)

    title_tokens = _tokenize_dynamic_text(figure_title)
    figure_tokens = _tokenize_dynamic_text(figure_text)
    section_tokens = _tokenize_dynamic_text(section_text)
    section_title_tokens = _tokenize_dynamic_text(_toc_section_title_text(section))
    title_overlap = title_tokens & section_tokens
    title_section_overlap = title_tokens & section_title_tokens
    context_overlap = (figure_tokens & section_tokens) - title_overlap
    explicit_assignment = figure in _section_figures(section)
    strong_title_anchor = _has_strong_figure_section_anchor(figure_title, section)

    token_frequencies = _toc_token_frequencies(toc_list)
    section_count = max(len(toc_list), 1)

    score = 0
    if explicit_assignment:
        score += 35 if strong_title_anchor else 12

    score += _weighted_dynamic_overlap_score(
        title_overlap,
        token_frequencies,
        section_count,
        base_score=26,
    )
    score += _weighted_dynamic_overlap_score(
        title_section_overlap,
        token_frequencies,
        section_count,
        base_score=18,
    )
    score += _weighted_dynamic_overlap_score(
        context_overlap,
        token_frequencies,
        section_count,
        base_score=5,
    )

    compact_title = _compact_match_text(figure_title)
    compact_section = _compact_match_text(section_text)
    if (
        compact_title
        and compact_section
        and _is_informative_compact_match(compact_title)
        and _is_informative_compact_match(compact_section)
        and (compact_title in compact_section or compact_section in compact_title)
    ):
        score += 70

    # Visual-section score is only a weak tie-breaker; direct overlap decides.
    score += max(_visual_toc_section_score(toc_list, section_index), 0) // 10
    if not _toc_section_has_child(toc_list, section_index):
        score += 4

    if not title_overlap and not context_overlap:
        score -= 20 if explicit_assignment else 45

    score += _figure_context_filter_score(figure, figure_contexts)
    return score


def _section_match_payload(section: Any) -> dict[str, Any]:
    return {
        "num": _toc_section_num(section),
        "title": _extract_section_title(section),
        "level": _toc_section_level(section),
        "outline": str(section.get("outline", "") if isinstance(section, dict) else ""),
        "modules": _section_modules(section),
    }


def _parse_json_array(text: str) -> list[Any]:
    raw = _unwrap_code_block(text or "")
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _build_semantic_match_items(
    *,
    records: list[dict[str, Any]],
    toc_list: list,
    figure_contexts: dict[int, str],
    max_candidates_per_figure: int,
) -> list[dict[str, Any]]:
    by_figure: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_figure.setdefault(int(record["figure"]), []).append(record)

    items: list[dict[str, Any]] = []
    for figure in by_figure:
        figure_records = sorted(
            by_figure[figure],
            key=lambda item: (int(item["score"]), -int(item["section_index"])),
            reverse=True,
        )[:max_candidates_per_figure]
        context = _figure_context_with_placeholder(figure, figure_contexts)
        items.append(
            {
                "figure": figure,
                "title": _extract_figure_title(figure, figure_contexts),
                "context": truncate_text(remove_figure_placeholders(context), 900),
                "candidate_sections": [
                    {
                        **_section_match_payload(toc_list[int(record["section_index"])]),
                        "local_score": int(record["score"]),
                    }
                    for record in figure_records
                ],
            }
        )
    return items


def _run_figure_semantic_matcher(
    *,
    records: list[dict[str, Any]],
    toc_list: list,
    figure_contexts: dict[int, str],
) -> dict[int, dict[str, Any]]:
    if not _bool_env("AUTOSTAT_USE_LLM_FIGURE_MATCHER", True):
        return {}
    if not records:
        return {}

    max_candidates = _positive_int_env(
        "AUTOSTAT_FIGURE_MATCH_CANDIDATE_SECTIONS",
        DEFAULT_FIGURE_MATCH_CANDIDATE_SECTIONS,
    )
    items = _build_semantic_match_items(
        records=records,
        toc_list=toc_list,
        figure_contexts=figure_contexts,
        max_candidates_per_figure=max_candidates,
    )
    if not items:
        return {}

    prompt_ctx = {
        "figure_match_items": json.dumps(items, ensure_ascii=False, indent=2),
    }
    try:
        sys_prompt = render_file("reporting_partly/figure_semantic_matcher_llm_sys.txt", prompt_ctx)
        user_prompt = render_file("reporting_partly/figure_semantic_matcher_llm_user.txt", prompt_ctx)
        raw = chat(sys_prompt, user_prompt, name="report_partly.figure_semantic_match", temperature=0.1)
    except Exception as exc:
        print(f"[REPORT][FIG_MATCH] semantic matcher failed: {exc!r}")
        return {}

    decisions: dict[int, dict[str, Any]] = {}
    for item in _parse_json_array(raw):
        if not isinstance(item, dict):
            continue
        try:
            figure = int(item.get("figure"))
        except Exception:
            continue
        best_section = str(item.get("best_section") or "").replace("．", ".").strip()
        should_insert = bool(item.get("should_insert", True))
        try:
            confidence = float(item.get("confidence", 0))
        except Exception:
            confidence = 0.0
        decisions[figure] = {
            "figure": figure,
            "best_section": best_section,
            "should_insert": should_insert,
            "confidence": max(0.0, min(confidence, 1.0)),
            "reason": str(item.get("reason", "") or "").strip(),
        }

    print(f"[REPORT][FIG_MATCH] semantic decisions={decisions}")
    return decisions


def _apply_semantic_match_decisions(
    *,
    records: list[dict[str, Any]],
    decisions: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not decisions:
        return records

    min_confidence = _positive_float_env(
        "AUTOSTAT_LLM_FIGURE_MATCH_MIN_CONFIDENCE",
        DEFAULT_LLM_FIGURE_MATCH_MIN_CONFIDENCE,
    )
    adjusted: list[dict[str, Any]] = []
    for record in records:
        figure = int(record["figure"])
        decision = decisions.get(figure)
        if not decision:
            adjusted.append(record)
            continue

        confidence = float(decision.get("confidence", 0.0))
        section_num = str(record.get("section_num", "")).replace("．", ".").strip()
        best_section = str(decision.get("best_section", "")).replace("．", ".").strip()
        record_copy = dict(record)
        if decision.get("should_insert", True) and confidence >= min_confidence and best_section and section_num == best_section:
            record_copy["score"] = int(record_copy["score"]) + 100 + int(confidence * 100)
            record_copy["semantic"] = True
            record_copy["semantic_confidence"] = confidence
            record_copy["semantic_reason"] = decision.get("reason", "")
        else:
            record_copy["score"] = max(0, int(record_copy["score"]) - 25)
        adjusted.append(record_copy)
    return adjusted


def _best_available_record_for_figure(
    *,
    figure: int,
    toc_list: list,
    figure_contexts: dict[int, str],
    figure_artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if _figure_context_filter_score(figure, figure_contexts) <= -40:
        return None

    scored: list[dict[str, Any]] = []
    for section_index, section in enumerate(toc_list):
        is_modeling_figure_section = (
            _is_modeling_section(section)
            and _figure_is_modeling_result(figure, figure_contexts)
        )
        if (
            figure not in _section_figures(section)
            and _visual_toc_section_score(toc_list, section_index) < 10
            and not is_modeling_figure_section
        ):
            continue
        score = _figure_section_alignment_score(
            figure=figure,
            figure_contexts=figure_contexts,
            toc_list=toc_list,
            section_index=section_index,
            figure_artifacts=figure_artifacts,
        )
        # -100 marks a hard-incompatible section (for example, a raw-data
        # setup section or a non-model figure inside a modeling section).
        # Other negative scores only mean weak title overlap; a valid figure
        # still needs a deterministic placement when no stronger pair exists.
        if score <= -100:
            continue
        scored.append(
            {
                "figure": figure,
                "section_index": section_index,
                "section_num": _toc_section_num(section),
                "section_title": _extract_section_title(section),
                "score": max(score, 35),
                "explicit": figure in _section_figures(section),
                "fallback": True,
            }
        )

    if not scored:
        return None
    scored.sort(key=lambda item: (int(item["score"]), -int(item["section_index"])), reverse=True)
    return scored[0]


def _build_figure_insert_plan(
    *,
    toc_list: list,
    candidate_figures: list[int],
    figure_contexts: dict[int, str],
    figure_artifacts: list[dict[str, Any]] | None = None,
) -> tuple[list, list[int], list[dict[str, Any]]]:
    normalized = _normalize_toc_figures(toc_list)
    candidates = list(dict.fromkeys(candidate_figures))
    if not normalized or not candidates:
        return normalized, [], []

    max_figures = _report_figure_limit_env("AUTOSTAT_REPORT_MAX_FIGURES", DEFAULT_MAX_REPORT_FIGURES)
    max_per_section = _report_figure_limit_env(
        "AUTOSTAT_REPORT_MAX_FIGURES_PER_SECTION",
        DEFAULT_MAX_REPORT_FIGURES_PER_SECTION,
    )
    min_score = _positive_int_env("AUTOSTAT_REPORT_MIN_FIGURE_PAIR_SCORE", 35)

    records: list[dict[str, Any]] = []
    for figure in candidates:
        for section_index, section in enumerate(normalized):
            score = _figure_section_alignment_score(
                figure=figure,
                figure_contexts=figure_contexts,
                toc_list=normalized,
                section_index=section_index,
                figure_artifacts=figure_artifacts,
            )
            if score < min_score:
                continue
            records.append(
                {
                    "figure": figure,
                    "section_index": section_index,
                    "section_num": _toc_section_num(section),
                    "section_title": _extract_section_title(section),
                    "score": score,
                    "explicit": figure in _section_figures(section),
                }
            )

    recorded_figures = {int(record["figure"]) for record in records}
    for figure in candidates:
        if figure in recorded_figures:
            continue
        fallback_record = _best_available_record_for_figure(
            figure=figure,
            toc_list=normalized,
            figure_contexts=figure_contexts,
            figure_artifacts=figure_artifacts,
        )
        if fallback_record is not None:
            records.append(fallback_record)

    records = _apply_semantic_match_decisions(
        records=records,
        decisions=_run_figure_semantic_matcher(
            records=records,
            toc_list=normalized,
            figure_contexts=figure_contexts,
        ),
    )
    records.sort(
        key=lambda item: (
            int(item["score"]),
            1 if item.get("semantic") else 0,
            1 if item.get("explicit") else 0,
            -int(item["section_index"]),
            -candidates.index(int(item["figure"])),
        ),
        reverse=True,
    )

    selected_records: list[dict[str, Any]] = []
    selected_figures: set[int] = set()
    per_section: dict[int, int] = {}

    for record in records:
        figure = int(record["figure"])
        section_index = int(record["section_index"])
        if figure in selected_figures:
            continue
        if max_figures is not None and len(selected_figures) >= max_figures:
            break
        if max_per_section is not None and per_section.get(section_index, 0) >= max_per_section:
            continue
        if max_figures is not None and _figure_redundancy_penalty(
            figure=figure,
            selected=[int(item["figure"]) for item in selected_records],
            figure_contexts=figure_contexts,
        ) >= 35:
            continue
        selected_records.append(record)
        selected_figures.add(figure)
        per_section[section_index] = per_section.get(section_index, 0) + 1

    assigned_by_section: dict[int, list[int]] = {}
    for record in selected_records:
        section_index = int(record["section_index"])
        assigned_by_section.setdefault(section_index, []).append(int(record["figure"]))

    planned_toc: list[Any] = []
    for index, section in enumerate(normalized):
        if isinstance(section, dict):
            section_copy = dict(section)
            section_copy["figures"] = assigned_by_section.get(index, [])
            planned_toc.append(section_copy)
        else:
            planned_toc.append(section)

    planned_figures = [int(item["figure"]) for item in selected_records]
    dropped = [figure for figure in candidates if figure not in selected_figures]
    print(
        "[REPORT][FIG_PLAN] "
        f"planned={selected_records}, dropped={dropped}, "
        f"max_figures={max_figures}, max_per_section={max_per_section}, min_score={min_score}"
    )
    return planned_toc, planned_figures, selected_records


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


def _num_sort_key(num: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in str(num or "").replace("．", ".").split("."):
        try:
            parts.append(int(part))
        except Exception:
            parts.append(999)
    return tuple(parts) if parts else (999,)


def _make_core_section(num: str, title: str) -> dict[str, Any]:
    return {
        "num": num,
        "title": title,
        "level": num.count(".") + 1,
        "outline": "",
        "figures": [],
    }


def _core_report_section_titles(ctx: dict[str, Any]) -> dict[str, str]:
    return CORE_REPORT_SECTION_TITLES_EN if is_english_report(ctx.get("report_language")) else CORE_REPORT_SECTION_TITLES


def _fallback_core_title(num: str, ctx: dict[str, Any]) -> str:
    if is_english_report(ctx.get("report_language")):
        return f"{num} Analysis Results"
    return f"{num} 分析结果"


def _ensure_core_report_sections(toc_list: list, ctx: dict[str, Any]) -> list:
    if not toc_list:
        return toc_list

    normalized = _normalize_toc_figures(toc_list)
    by_num: dict[str, Any] = {
        _toc_section_num(section): section
        for section in normalized
        if _toc_section_num(section)
    }
    if not by_num:
        return normalized

    core_titles = _core_report_section_titles(ctx)
    required: dict[str, str] = {}
    stage_refs = ctx.get("stage_reference_contexts")
    if not isinstance(stage_refs, dict):
        stage_refs = {}
    if ctx.get("load_abstract") or stage_refs.get("loading"):
        required["1"] = core_titles["1"]
    if ctx.get("preproc_abstract") or stage_refs.get("preprocessing"):
        required["2"] = core_titles["2"]
    if ctx.get("visual_abstract") or stage_refs.get("visualization") or by_num.get("3") or any(num.startswith("3.") for num in by_num):
        required["3"] = core_titles["3"]
    if ctx.get("coding_abstract") or stage_refs.get("modeling"):
        required["4"] = core_titles["4"]
    required["5"] = core_titles["5"]

    for num in list(by_num):
        if "." not in num:
            continue
        parent = num.rsplit(".", 1)[0]
        while parent:
            required.setdefault(parent, core_titles.get(parent, _fallback_core_title(parent, ctx)))
            parent = parent.rsplit(".", 1)[0] if "." in parent else ""

    combined: dict[str, Any] = {}
    for num, title in required.items():
        combined[num] = _make_core_section(num, title)
    combined.update(by_num)

    sorted_sections = [
        combined[num]
        for num in sorted(combined, key=_num_sort_key)
    ]
    extra_sections = [
        section for section in normalized if not _toc_section_num(section)
    ]
    return _dedupe_toc_sections(sorted_sections + extra_sections)


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
    return _strip_natural_figure_references(
        remove_figure_placeholders(normalize_figure_placeholders(text or ""))
    )


def _strip_natural_figure_references(text: str) -> str:
    value = NATURAL_FIGURE_REFERENCE_RE.sub("", normalize_part(text or ""))
    value = re.sub(r"\s+([，,。.!！?？；;：:、])", r"\1", value)
    value = re.sub(r"[，,、；;：:]\s*([。.!！?？])", r"\1", value)
    return normalize_part(value)


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


def _section_default_modules(section: Any) -> list[int]:
    num = _toc_section_num(section)
    text = _toc_section_text(section).lower()
    if num.startswith("1") or any(token in text for token in ("loading", "profiling", "加载", "数据结构", "数据概况")):
        return [0]
    if num.startswith("2") or any(token in text for token in ("preprocess", "clean", "预处理", "清洗", "特征工程")):
        return [1]
    if num.startswith("3") or any(token in text for token in ("visual", "figure", "可视化", "图表", "分布")):
        return [2]
    if num.startswith("4") or any(token in text for token in ("model", "模型", "建模", "评估", "预测", "推断", "回归")):
        return [3]
    if num.startswith("5") or any(token in text for token in ("conclusion", "summary", "结论", "总结", "展望")):
        return [0, 1, 2, 3]
    return []


def _stage_reference(ctx: dict[str, Any], key: str) -> str:
    refs = ctx.get("stage_reference_contexts")
    if not isinstance(refs, dict):
        return ""
    return str(refs.get(key) or "").strip()


def _build_section_reference_context(
    *,
    section: Any,
    ctx: dict[str, Any],
    figure_contexts: dict[int, str],
) -> str:
    figures = _section_figures(section)
    modules = _section_modules(section) or _section_default_modules(section)
    parts: list[str] = []

    if 0 in modules:
        parts.append(_strip_fig_placeholders_for_context(
            _stage_reference(ctx, "loading") or str(ctx.get("load_abstract", ""))
        ))
    if 1 in modules:
        parts.append(_strip_fig_placeholders_for_context(
            _stage_reference(ctx, "preprocessing") or str(ctx.get("preproc_abstract", ""))
        ))

    figure_context = _format_figure_contexts(figures, figure_contexts)
    if figure_context:
        parts.append(figure_context)
    elif 2 in modules:
        parts.append(_strip_fig_placeholders_for_context(
            _stage_reference(ctx, "visualization") or str(ctx.get("visual_abstract", ""))
        ))

    if 3 in modules:
        parts.append(_strip_fig_placeholders_for_context(
            _stage_reference(ctx, "modeling") or str(ctx.get("coding_abstract", ""))
        ))

    if not parts:
        fallback_context = ctx.get("selected_full_conten", "")
        if figures:
            parts.append(_format_figure_contexts(figures, figure_contexts))
        else:
            parts.append(truncate_text(_strip_fig_placeholders_for_context(str(fallback_context)), 1800))

    return "\n\n".join(part for part in parts if normalize_part(part)).strip()


def _contains_all_figures(text: str, figures: list[int]) -> bool:
    present = set(_extract_fig_numbers_from_text(normalize_figure_placeholders(text or "")))
    return all(figure in present for figure in figures)


def _dedupe_required_figures(text: str, figures: list[int]) -> str:
    if not text or not figures:
        return text

    text = _strip_natural_figure_references(normalize_figure_placeholders(text))
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
    report_language: Any = REPORT_LANGUAGE_ZH,
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
        base += "." if is_english_report(report_language) else "。"
    return normalize_part(_dedupe_required_figures(f"{base} {suffix}", figures))


def run_reporting_partly_workflow(
    *,
    toc_text: str,
    selected_full_conten: str,
    load_abstract: str,
    preproc_abstract: str,
    visual_abstract: str,
    coding_abstract: str,
    figure_artifacts: Any = None,
    user_input: str = "",
    add_preference: str = "",
    preference_select: str = "",
    ref_context: str = "",
    stage_reference_contexts: dict[str, str] | None = None,
    respect_user_toc: bool = False,
    report_language: str = REPORT_LANGUAGE_ZH,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    report_language = normalize_report_language(report_language)
    successful_artifacts = successful_figure_artifacts(figure_artifacts)
    selected_full_conten = _ensure_visual_fig_placeholders(selected_full_conten or "")
    if successful_artifacts:
        candidate_figures = list(range(len(successful_artifacts)))
        candidate_figure_contexts = figure_artifact_contexts(successful_artifacts)
        selected_full_conten = _format_figure_contexts(candidate_figures, candidate_figure_contexts)
    else:
        candidate_figures = _extract_fig_numbers_from_text(selected_full_conten)
        candidate_figure_contexts = _extract_figure_contexts(selected_full_conten)
    authoritative_toc = _parse_toc_text(toc_text)
    candidate_pool_figures = _select_report_figures(
        candidate_figures=candidate_figures,
        figure_contexts=candidate_figure_contexts,
        toc_list=authoritative_toc,
    )
    figure_contexts = {
        figure: candidate_figure_contexts.get(figure, f"[FIG:{figure}]")
        for figure in candidate_pool_figures
    }
    selected_full_conten = _format_figure_contexts(candidate_pool_figures, figure_contexts)
    print(
        f"[REPORT][INPUT] partly selected_full_conten length={len(selected_full_conten)}, "
        f"candidate_fig_refs={candidate_figures}, candidate_pool_fig_refs={candidate_pool_figures}"
    )
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
        "stage_reference_contexts": stage_reference_contexts or {},
        "report_language": report_language,
        "language_name": report_language_name(report_language),
        "language_instruction": report_language_instruction(report_language),
    }

    # ---------- 节点 1: base_toc ----------
    toc_list = _dedupe_toc_sections(_normalize_toc_figures(authoritative_toc))
    if not respect_user_toc:
        toc_list = _ensure_core_report_sections(toc_list, ctx)
    _log_toc_figure_state("toc_list before module planning", toc_list)
    ctx["toc_list"] = toc_list

    # ---------- 节点 2: update_toc_with_relevant_sections ----------
    _raise_if_report_cancelled(cancel_check)
    up_sys = render_file("reporting_partly/update_toc_with_relevant_sections_llm_sys.txt", ctx)
    up_user = render_file("reporting_partly/update_toc_with_relevant_sections_llm_user.txt", ctx)
    toc_final_raw = chat(up_sys, up_user, name="report_partly.update_toc").strip()
    _raise_if_report_cancelled(cancel_check)
    toc_list_final = _parse_toc_list(toc_final_raw)
    toc_list_final = _merge_toc_with_authoritative_order(toc_list_final, toc_list or authoritative_toc)

    if not toc_list_final:
        toc_list_final = toc_list

    if not toc_list_final:
        toc_list_final = [
            line.strip() for line in (toc_text or "").splitlines() if line.strip()
        ]

    toc_list_final = _clear_toc_figures(_normalize_toc_figures(toc_list_final))
    toc_list_final = _dedupe_toc_sections(toc_list_final)
    if not respect_user_toc:
        toc_list_final = _ensure_core_report_sections(toc_list_final, ctx)
    _log_toc_figure_state("toc_list before figure matching", toc_list_final)
    toc_list_final, planned_figures, figure_insert_plan = _build_figure_insert_plan(
        toc_list=toc_list_final,
        candidate_figures=candidate_pool_figures,
        figure_contexts=figure_contexts,
        figure_artifacts=successful_artifacts,
    )
    ctx["figure_insert_plan"] = figure_insert_plan
    _log_toc_figure_state("toc_list_final after insert plan", toc_list_final)

    report_parts: list[str] = []
    report_part_sections: list[Any] = []
    seen_parts: set[str] = set()
    history_parts_for_prompt: list[str] = []
    total_sections = len(toc_list_final)

    def emit_progress(
        phase: str,
        *,
        section: Any | None = None,
        section_index: int = 0,
        markdown: str = "",
        completed_sections: int | None = None,
        status: str = "writing",
        draft: bool | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "status": status,
            "phase": phase,
            "section_index": section_index,
            "total_sections": total_sections,
            "completed_sections": len(report_parts) if completed_sections is None else completed_sections,
            "markdown": markdown,
        }
        if section is not None:
            payload["section_num"] = _toc_section_num(section)
            payload["section_title"] = _extract_section_title(section)
        payload["report_language"] = report_language
        if draft is not None:
            payload["draft"] = draft
        _emit_report_progress(progress_callback, payload)

    emit_progress("ready", completed_sections=0)

    for idx, section in enumerate(toc_list_final):
        _raise_if_report_cancelled(cancel_check)
        emit_progress(
            "section_started",
            section=section,
            section_index=idx + 1,
            markdown="\n\n".join(report_parts).strip(),
        )
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
        content = _preserve_required_figures(content, "", section, figure_contexts, report_language)
        section_ctx["content"] = content
        draft_part = normalize_part(wrap_section_as_markdown(section, content))
        if draft_part:
            emit_progress(
                "section_draft",
                section=section,
                section_index=idx + 1,
                markdown="\n\n".join([*report_parts, draft_part]).strip(),
                draft=True,
            )

        # ---------- fill_report ----------
        f_sys = render_file("reporting_partly/fill_report_llm_sys.txt", section_ctx)
        f_user = render_file("reporting_partly/fill_report_llm_user.txt", section_ctx)
        filled = chat(f_sys, f_user, name=f"report_partly.fill.{idx+1}").strip()
        _raise_if_report_cancelled(cancel_check)
        filled = _unwrap_code_block(filled)
        filled = normalize_part(filled)
        filled = _preserve_required_figures(filled, content, section, figure_contexts, report_language)

        final_part = filled or content
        final_part = normalize_part(final_part)
        final_part = _preserve_required_figures(final_part, content, section, figure_contexts, report_language)

        # 统一用目录标题，避免正文开头重复标题。
        original_final_part = final_part
        final_part = _strip_redundant_heading(final_part, section)
        final_part = normalize_part(final_part)
        if original_final_part and not final_part:
            final_part = original_final_part
        final_part = _preserve_required_figures(final_part, original_final_part, section, figure_contexts, report_language)

        dedup_key = normalize_for_dedup(_extract_section_title(section))
        if dedup_key in seen_parts:
            emit_progress(
                "section_skipped",
                section=section,
                section_index=idx + 1,
                markdown="\n\n".join(report_parts).strip(),
            )
            continue

        seen_parts.add(dedup_key)

        wrapped_part = wrap_section_as_markdown(section, final_part)

        wrapped_part = normalize_part(wrapped_part)
        if wrapped_part:
            report_parts.append(wrapped_part)
            report_part_sections.append(section)
            emit_progress(
                "section_completed",
                section=section,
                section_index=idx + 1,
                markdown="\n\n".join(report_parts).strip(),
                draft=False,
            )

        history_parts_for_prompt.append(truncate_text(final_part, 800))

    report_parts = _ensure_all_figures_in_report_parts(
        report_parts=report_parts,
        report_sections=report_part_sections,
        toc_list=toc_list_final,
        expected_figures=planned_figures,
        figure_contexts=figure_contexts,
    )
    final_html = "\n\n".join(report_parts).strip()
    _log_final_figure_ledger(final_html, planned_figures)
    emit_progress(
        "body_completed",
        section_index=total_sections,
        markdown=final_html,
        status="finalizing",
    )

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
        "title": title or ("Data Analysis Report" if is_english_report(report_language) else "数据分析报告"),
        "report_language": report_language,
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
