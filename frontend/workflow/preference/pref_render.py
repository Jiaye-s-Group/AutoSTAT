import streamlit as st

from utils.i18n import bt
from utils.workflow_state import invalidate_from, stable_fingerprint


PREFERENCE_FIELDS = (
    {
        "id": "report_style",
        "label": ("1. 报告风格", "1. Report Style"),
        "summary_label": ("报告风格", "Report Style"),
        "default": "balanced",
        "options": (
            ("concise", "简洁直观", "Concise and direct"),
            ("balanced", "适中平衡", "Balanced"),
            ("technical", "深度技术型", "Deep technical"),
        ),
    },
    {
        "id": "analysis_type",
        "label": ("2. 分析方向偏好", "2. Analysis Focus"),
        "summary_label": ("分析方向偏好", "Analysis Focus"),
        "default": "business",
        "options": (
            ("business", "商业分析", "Business analysis"),
            ("academic", "学术分析", "Academic analysis"),
            ("engineering", "工程/产品分析", "Engineering/product analysis"),
        ),
    },
    {
        "id": "model_pref",
        "label": ("3. 模型偏好", "3. Model Preference"),
        "summary_label": ("模型偏好", "Model Preference"),
        "default": "interpretable",
        "options": (
            ("interpretable", "可解释性强", "High interpretability"),
            ("predictive", "预测性能最优", "Best predictive performance"),
            ("fast", "训练时间短", "Short training time"),
        ),
    },
    {
        "id": "missing_pref",
        "label": ("4. 缺失值处理方式", "4. Missing Value Handling"),
        "summary_label": ("缺失值处理方式", "Missing Value Handling"),
        "default": "simple",
        "options": (
            ("simple", "简单填补", "Simple imputation"),
            ("frequency", "频率填补", "Frequency imputation"),
            ("advanced", "高级填补（KNN/MICE）", "Advanced imputation (KNN/MICE)"),
        ),
    },
    {
        "id": "lang_style",
        "label": ("5. 报告语言风格", "5. Report Tone"),
        "summary_label": ("报告语言风格", "Report Tone"),
        "default": "plain",
        "options": (
            ("plain", "通俗易懂", "Plain and accessible"),
            ("business", "商业风", "Business tone"),
            ("academic", "学术论文风", "Academic paper style"),
        ),
    },
    {
        "id": "feature_pref",
        "label": ("6. 特征工程偏好", "6. Feature Engineering Preference"),
        "summary_label": ("特征工程偏好", "Feature Engineering Preference"),
        "default": "few_key",
        "options": (
            ("few_key", "少量关键特征", "A small set of key features"),
            ("many_candidates", "大量候选特征", "Many candidate features"),
            ("basic_only", "只做基础处理", "Basic processing only"),
        ),
    },
)


def _field_by_id(field_id: str) -> dict:
    for field in PREFERENCE_FIELDS:
        if field["id"] == field_id:
            return field
    raise KeyError(field_id)


def _option_ids(field: dict) -> list[str]:
    return [str(option[0]) for option in field["options"]]


def _option_label(field: dict, option_id: str) -> str:
    for value, zh, en in field["options"]:
        if value == option_id:
            return bt(zh, en)
    return str(option_id)


def _legacy_selected_value(field: dict, preferences: object) -> str:
    if not isinstance(preferences, dict):
        return str(field["default"])
    summary_zh, summary_en = field["summary_label"]
    selected = preferences.get(summary_zh)
    if selected is None:
        selected = preferences.get(summary_en)
    if selected is None:
        return str(field["default"])
    selected_text = str(selected)
    for value, zh, en in field["options"]:
        if selected_text in {str(value), zh, en}:
            return str(value)
    return str(field["default"])


def _saved_form_values() -> dict[str, str]:
    saved = st.session_state.get("preference_form_values")
    values = saved if isinstance(saved, dict) else {}
    out: dict[str, str] = {}
    for field in PREFERENCE_FIELDS:
        field_id = str(field["id"])
        candidate = values.get(field_id)
        if candidate in _option_ids(field):
            out[field_id] = str(candidate)
        else:
            out[field_id] = _legacy_selected_value(
                field,
                st.session_state.get("preference_selected"),
            )
    return out


def _sync_preference_widget_defaults() -> None:
    if "modeling_requirements" not in st.session_state:
        st.session_state.modeling_requirements = st.session_state.get("add_preference") or ""
    saved_values = _saved_form_values()
    for field in PREFERENCE_FIELDS:
        widget_key = f"preference_{field['id']}"
        if st.session_state.get(widget_key) not in _option_ids(field):
            st.session_state[widget_key] = saved_values[str(field["id"])]


def _localized_preferences(values: dict[str, str]) -> dict[str, str]:
    preferences: dict[str, str] = {}
    for field in PREFERENCE_FIELDS:
        summary_zh, summary_en = field["summary_label"]
        field_id = str(field["id"])
        preferences[bt(summary_zh, summary_en)] = _option_label(
            field,
            values.get(field_id, str(field["default"])),
        )
    return preferences


def _radio_field(field_id: str) -> str:
    field = _field_by_id(field_id)
    label_zh, label_en = field["label"]
    option_ids = _option_ids(field)
    selected = st.radio(
        bt(label_zh, label_en),
        option_ids,
        format_func=lambda value: _option_label(field, str(value)),
        key=f"preference_{field_id}",
    )
    return str(selected)


def preferences_select():
    _sync_preference_widget_defaults()

    modeling_requirements = st.text_area(
        bt(
            "请描述你的数据分析目标与需求",
            "Describe your data analysis goals and requirements",
        ),
        placeholder=bt(
            "例如：比较不同组别在 XX 指标上的差异，分析潜在影响因素。",
            "For example: compare groups on an outcome of interest and analyze potential influencing factors.",
        ),
        height=200,
        key="modeling_requirements"
    )
    
    saved_requirement = st.session_state.get("add_preference")
    if saved_requirement:
        st.chat_message("assistant").write(bt(
            "用户的需求是：{content}",
            "User requirement: {content}",
            content=saved_requirement,
        ))
    
    col1, col2, col3 = st.columns(3)

    with col1:
        report_style = _radio_field("report_style")

    with col2:
        analysis_type = _radio_field("analysis_type")

    with col3:
        model_pref = _radio_field("model_pref")

    col1, col2, col3 = st.columns(3)

    with col1:
        missing_pref = _radio_field("missing_pref")

    with col2:
        lang_style = _radio_field("lang_style")

    with col3:
        feature_pref = _radio_field("feature_pref")

    preferences = None
    if st.button(bt("保存偏好设置", "Save Preferences"), use_container_width=True):
        form_values = {
            "report_style": report_style,
            "analysis_type": analysis_type,
            "model_pref": model_pref,
            "missing_pref": missing_pref,
            "lang_style": lang_style,
            "feature_pref": feature_pref,
        }
        preferences = _localized_preferences(form_values)

        preference_fingerprint = stable_fingerprint(modeling_requirements, form_values)
        if preference_fingerprint != st.session_state.get("preference_fingerprint"):
            invalidate_from(
                st.session_state,
                "preferences",
                reason="analysis preferences changed",
            )
            st.session_state.preference_fingerprint = preference_fingerprint
        st.session_state.add_preference = modeling_requirements
        st.session_state.preference_form_values = form_values
        st.session_state.preference_selected = preferences
        st.success(bt("偏好设置已保存！", "Preferences saved."))
        st.rerun()
    return preferences


def prep_chat(agent):
    """渲染对话式建议区"""

    with st.chat_message("assistant"):
        st.write(bt(
            "我是 Autostat 自动模式决策助手，很高兴为您服务！\n\n您可以在左侧边栏开启自动模式，我会协助您决策并一键完成所有分析",
            "I am the Autostat auto-mode decision assistant.\n\nStart auto mode from the left sidebar, and I will help choose the workflow and complete the analysis.",
        ))

    if agent.plan is not None:
        st.chat_message("assistant").write(agent.plan)
  

if __name__ == "__main__":

    st.title(bt("偏好设置", "Preferences"))
    st.markdown("---")

    c = st.columns(2)

    planner = st.session_state.planner_agent

    with c[0].expander(bt("偏好设置", "Preferences"), True):
        preferences_select()
    with c[1].expander(bt("自动模式决策报告", "Auto-Mode Decision Report"), True):
        prep_chat(planner)
