"""Shared language helpers for UI-facing and report-facing generation."""
from __future__ import annotations

from typing import Any


REPORT_LANGUAGE_ZH = "zh"
REPORT_LANGUAGE_EN = "en"
REPORT_LANGUAGE_DEFAULT = REPORT_LANGUAGE_ZH

APP_LANGUAGE_ZH = REPORT_LANGUAGE_ZH
APP_LANGUAGE_EN = REPORT_LANGUAGE_EN
APP_LANGUAGE_DEFAULT = APP_LANGUAGE_ZH


def normalize_report_language(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("value", "label", "name", "text"):
            if key in value:
                normalized = normalize_report_language(value.get(key))
                if normalized:
                    return normalized
        return REPORT_LANGUAGE_DEFAULT

    if isinstance(value, (list, tuple, set)):
        for item in value:
            normalized = normalize_report_language(item)
            if normalized:
                return normalized
        return REPORT_LANGUAGE_DEFAULT

    text = str(value or "").strip().lower()
    if text in {"en", "eng", "english", "english report"} or "english" in text or "英文" in text:
        return REPORT_LANGUAGE_EN
    if (
        text in {"zh", "cn", "chinese", "中文", "中文报告", "zh-cn", "zh_cn"}
        or "中文" in text
        or "chinese" in text
    ):
        return REPORT_LANGUAGE_ZH
    return REPORT_LANGUAGE_DEFAULT


def is_english_report(language: Any) -> bool:
    return normalize_report_language(language) == REPORT_LANGUAGE_EN


def report_language_name(language: Any) -> str:
    return "English" if is_english_report(language) else "中文"


def report_language_html_lang(language: Any) -> str:
    return "en" if is_english_report(language) else "zh-CN"


def report_language_instruction(language: Any) -> str:
    if is_english_report(language):
        return (
            "Write all report-facing titles, outline items, section prose, captions, and summary text "
            "in polished professional English. Preserve dataset field names, formulas, code identifiers, "
            "metric names, numeric values, tables, and [FIG:x] placeholders exactly. Do not mix Chinese "
            "unless it is part of a source field name, uploaded reference, or user-provided term."
        )
    return (
        "使用正式、专业、自然的中文撰写所有面向报告读者的标题、目录、正文、图题和总结。"
        "保留数据字段名、公式、代码标识符、指标名、数值、表格和 [FIG:x] 占位符，不要擅自改写。"
    )


def report_language_progress_label(language: Any) -> str:
    return "English report" if is_english_report(language) else "报告"


def normalize_app_language(value: Any) -> str:
    return normalize_report_language(value)


def is_english_language(language: Any) -> bool:
    return normalize_app_language(language) == APP_LANGUAGE_EN


def app_language_name(language: Any) -> str:
    return "English" if is_english_language(language) else "中文"


def app_language_ref_context_empty(language: Any) -> str:
    return "No reference materials." if is_english_language(language) else "（无参考资料）"


def app_language_instruction(language: Any) -> str:
    """Instruction injected into every workflow prompt for generated answers."""
    if is_english_language(language):
        return (
            "Output language requirement: produce every user-facing answer, suggestion, "
            "status sentence, chart title, report excerpt, and explanatory paragraph in "
            "polished professional English. This requirement overrides any earlier "
            "language-specific wording in the prompt. Preserve dataset field names, "
            "source text, formulas, code identifiers, JSON keys, metric names, numeric "
            "values, tables, and placeholders exactly. Only keep Chinese when it is part "
            "of an original field name, uploaded reference, user-provided term, or code."
        )
    return (
        "输出语言要求：所有面向用户的回答、建议、状态描述、图表标题、报告片段和解释性段落均使用"
        "正式、专业、自然的中文。本要求优先于 prompt 中其他语言描述。保留数据字段名、原始资料文本、"
        "公式、代码标识符、JSON key、指标名、数值、表格和占位符，不要擅自改写。"
    )
