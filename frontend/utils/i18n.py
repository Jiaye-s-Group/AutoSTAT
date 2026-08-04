"""Small Streamlit i18n helpers for AutoSTAT."""
from __future__ import annotations

from typing import Any

import streamlit as st

from core.report_language import (
    APP_LANGUAGE_DEFAULT,
    APP_LANGUAGE_EN,
    APP_LANGUAGE_ZH,
    app_language_instruction,
    app_language_name,
    normalize_app_language,
)


UI_LANGUAGE_SESSION_KEY = "ui_language"
LANGUAGE_LABELS = {
    APP_LANGUAGE_ZH: "中文",
    APP_LANGUAGE_EN: "English",
}

_TRANSLATIONS: dict[str, dict[str, str]] = {
    "common.language": {"zh": "界面语言", "en": "Language"},
    "common.zh": {"zh": "中文", "en": "中文"},
    "common.en": {"zh": "English", "en": "English"},
    "common.default": {"zh": "默认", "en": "Default"},
    "common.cancel": {"zh": "取消", "en": "Cancel"},
    "common.confirm": {"zh": "确认", "en": "Confirm"},
    "common.success": {"zh": "成功", "en": "Success"},
    "common.failed": {"zh": "失败", "en": "Failed"},
    "common.no_reference": {"zh": "（无参考资料）", "en": "No reference materials."},
    "common.load_data_first": {
        "zh": "请先在数据导入页面加载数据。",
        "en": "Please load data on the Data Import page first.",
    },
    "common.workflow_error": {
        "zh": "工作流执行失败：{error}",
        "en": "Workflow execution failed: {error}",
    },
    "app.nav.analysis_flow": {"zh": "分析流程", "en": "Analysis Flow"},
    "app.nav.system_config": {"zh": "系统配置", "en": "System Settings"},
    "app.page.data_loading": {"zh": "📥 数据导入", "en": "📥 Data Import"},
    "app.page.preprocessing": {"zh": "🛠️ 数据预处理", "en": "🛠️ Preprocessing"},
    "app.page.visualization": {"zh": "📊 数据可视化", "en": "📊 Visualization"},
    "app.page.modeling": {"zh": "🧠 建模分析", "en": "🧠 Modeling"},
    "app.page.report": {"zh": "📝 报告生成", "en": "📝 Report"},
    "app.page.preference": {"zh": "⚙️ 偏好设置", "en": "⚙️ Preferences"},
    "sidebar.llm_config": {"zh": "🔧 大模型配置", "en": "🔧 LLM Configuration"},
    "sidebar.llm_caption": {
        "zh": "选择预设服务商，或使用任意 OpenAI-compatible API。",
        "en": "Choose a preset provider, or use any OpenAI-compatible API.",
    },
    "sidebar.llm_provider": {"zh": "模型服务商", "en": "Model Provider"},
    "sidebar.provider.custom": {"zh": "自定义", "en": "Custom"},
    "sidebar.preset_placeholder": {
        "zh": "快捷预设（选择后自动填充 Base URL 和模型名）",
        "en": "Quick preset (fills Base URL and model automatically)",
    },
    "sidebar.remember_config": {"zh": "保存到本机用户配置", "en": "Save to local user config"},
    "sidebar.remember_config_help": {
        "zh": "保存到 {path}，不会写入项目仓库。",
        "en": "Save to {path}. This will not write to the project repository.",
    },
    "sidebar.save_config": {"zh": "保存配置", "en": "Save Configuration"},
    "sidebar.config_saved": {"zh": "配置已保存。", "en": "Configuration saved."},
    "sidebar.config_invalid": {"zh": "配置无效：{error}", "en": "Invalid configuration: {error}"},
    "sidebar.save_connect": {"zh": "💾 保存并连接", "en": "💾 Save and Connect"},
    "sidebar.connected": {"zh": "✅ 连接成功！", "en": "✅ Connected successfully."},
    "sidebar.connect_failed": {"zh": "连接失败：{error}", "en": "Connection failed: {error}"},
    "sidebar.fill_llm_fields": {
        "zh": "请填写 API Key、Base URL 和 Model 三项。",
        "en": "Please fill in API Key, Base URL, and Model.",
    },
    "sidebar.ready": {"zh": "✅ 已就绪", "en": "✅ Ready"},
    "sidebar.status_ready": {"zh": "已就绪", "en": "Ready"},
    "sidebar.unknown_model": {"zh": "未知模型", "en": "Unknown model"},
    "sidebar.not_connected": {
        "zh": "⚠️ 未连接，请填写完整配置后保存",
        "en": "⚠️ Not connected. Fill in the configuration and save.",
    },
    "sidebar.status_not_connected": {
        "zh": "未连接，请填写完整配置后保存。",
        "en": "Not connected. Fill in the complete configuration and save.",
    },
    "sidebar.clear_all": {"zh": "🧹 清空所有数据", "en": "🧹 Clear All Data"},
    "sidebar.start_auto": {"zh": "🚗 开启自动模式", "en": "🚗 Start Auto Mode"},
    "sidebar.stop_auto": {"zh": "❌ 结束自动模式", "en": "❌ Stop Auto Mode"},
    "auto.need_data": {
        "zh": "请先上传或导入数据，再开启自动模式。",
        "en": "Please upload or import data before starting auto mode.",
    },
    "auto.no_stage": {
        "zh": "Planning 未开启任何后续自动阶段。",
        "en": "Planning did not enable any downstream auto stage.",
    },
    "auto.planning_failed": {
        "zh": "Planning workflow 执行失败：{error}",
        "en": "Planning workflow failed: {error}",
    },
}


def get_language(default: Any = None) -> str:
    value = st.session_state.get(UI_LANGUAGE_SESSION_KEY, default or APP_LANGUAGE_DEFAULT)
    language = normalize_app_language(value)
    st.session_state[UI_LANGUAGE_SESSION_KEY] = language
    return language


def set_language(language: Any) -> str:
    normalized = normalize_app_language(language)
    st.session_state[UI_LANGUAGE_SESSION_KEY] = normalized
    return normalized


def is_english_ui() -> bool:
    return get_language() == APP_LANGUAGE_EN


def t(key: str, **kwargs: Any) -> str:
    language = get_language()
    bundle = _TRANSLATIONS.get(key, {})
    text = bundle.get(language) or bundle.get(APP_LANGUAGE_ZH) or key
    if kwargs:
        return text.format(**kwargs)
    return text


def bt(zh: str, en: str, **kwargs: Any) -> str:
    text = en if is_english_ui() else zh
    if kwargs:
        return text.format(**kwargs)
    return text


def language_instruction(language: Any = None) -> str:
    return app_language_instruction(normalize_app_language(language) if language is not None else get_language())


def language_name(language: Any = None) -> str:
    return app_language_name(normalize_app_language(language) if language is not None else get_language())


def language_options() -> list[str]:
    return [LANGUAGE_LABELS[APP_LANGUAGE_ZH], LANGUAGE_LABELS[APP_LANGUAGE_EN]]


def language_to_label(language: Any) -> str:
    return LANGUAGE_LABELS[normalize_app_language(language)]


def language_from_label(label: str) -> str:
    normalized = normalize_app_language(label)
    if normalized in LANGUAGE_LABELS:
        return normalized
    return APP_LANGUAGE_EN if str(label).strip().lower() == "english" else APP_LANGUAGE_ZH


def language_index(language: Any) -> int:
    return 1 if normalize_app_language(language) == APP_LANGUAGE_EN else 0


def sync_report_language(report_agent: Any, language: Any = None) -> str:
    normalized = normalize_app_language(language or get_language())
    if report_agent is None:
        return normalized
    if hasattr(report_agent, "save_report_language"):
        report_agent.save_report_language(normalized)
    else:
        setattr(report_agent, "report_language", normalized)
        setattr(report_agent, "report_current_language", normalized)
    return normalized
