import streamlit as st

from utils.i18n import bt
from utils.workflow_state import invalidate_from, stable_fingerprint


def preferences_select():

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
    
    # 如果用户有输入（非空）
    if st.session_state.add_preference is not None:
        st.chat_message("assistant").write(bt(
            "用户的需求是：{content}",
            "User requirement: {content}",
            content=st.session_state.add_preference,
        ))
    
    col1, col2, col3 = st.columns(3)

    with col1:
        report_style = st.radio(
            bt("1. 报告风格", "1. Report Style"),
            [bt("简洁直观", "Concise and direct"), bt("适中平衡", "Balanced"), bt("深度技术型", "Deep technical")],
            index=1,
        )

    with col2:
        analysis_type = st.radio(
            bt("2. 分析方向偏好", "2. Analysis Focus"),
            [bt("商业分析", "Business analysis"), bt("学术分析", "Academic analysis"), bt("工程/产品分析", "Engineering/product analysis")],
        )

    with col3:
        model_pref = st.radio(
            bt("3. 模型偏好", "3. Model Preference"),
            [bt("可解释性强", "High interpretability"), bt("预测性能最优", "Best predictive performance"), bt("训练时间短", "Short training time")],
            index=0,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        missing_pref = st.radio(
            bt("4. 缺失值处理方式", "4. Missing Value Handling"),
            [bt("简单填补", "Simple imputation"), bt("频率填补", "Frequency imputation"), bt("高级填补（KNN/MICE）", "Advanced imputation (KNN/MICE)")],
        )

    with col2:
        lang_style = st.radio(
            bt("5. 报告语言风格", "5. Report Tone"),
            [bt("通俗易懂", "Plain and accessible"), bt("商业风", "Business tone"), bt("学术论文风", "Academic paper style")],
        )

    with col3:
        feature_pref = st.radio(
            bt("6. 特征工程偏好", "6. Feature Engineering Preference"),
            [bt("少量关键特征", "A small set of key features"), bt("大量候选特征", "Many candidate features"), bt("只做基础处理", "Basic processing only")],
        )

    preferences = None
    if st.button(bt("保存偏好设置", "Save Preferences"), use_container_width=True):
        preferences = {
            bt("报告风格", "Report Style"): report_style,
            bt("模型偏好", "Model Preference"): model_pref,
            bt("缺失值处理方式", "Missing Value Handling"): missing_pref,
            bt("特征工程偏好", "Feature Engineering Preference"): feature_pref,
            bt("报告语言风格", "Report Tone"): lang_style,
            bt("分析方向偏好", "Analysis Focus"): analysis_type,
        }

        preference_fingerprint = stable_fingerprint(modeling_requirements, preferences)
        if preference_fingerprint != st.session_state.get("preference_fingerprint"):
            invalidate_from(
                st.session_state,
                "preferences",
                reason="analysis preferences changed",
            )
            st.session_state.preference_fingerprint = preference_fingerprint
        st.session_state.add_preference = modeling_requirements
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
