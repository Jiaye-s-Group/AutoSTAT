import streamlit as st


def preferences_select():

    modeling_requirements = st.text_area(
        "请描述你的数据分析目标与需求",
        placeholder="例如：请帮我对数据进行可视化",
        height=200,
        key="modeling_requirements"
    )
    
    # 如果用户有输入（非空）
    if st.session_state.add_preference is not None:
        st.chat_message("assistant").write(f"用户的需求是：{st.session_state.add_preference}")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        report_style = st.radio(
            "1. 报告风格",
            ["简洁直观", "适中平衡", "深度技术型"],
            index=1,
        )

    with col2:
        analysis_type = st.radio(
            "2. 分析方向偏好",
            ["商业分析", "学术分析", "工程/产品分析"],
        )

    with col3:
        model_pref = st.radio(
            "3. 模型偏好",
            ["可解释性强", "预测性能最优", "训练时间短"],
            index=0,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        missing_pref = st.radio(
            "4. 缺失值处理方式",
            ["简单填补", "频率填补", "高级填补（KNN/MICE）"],
        )

    with col2:
        lang_style = st.radio(
            "5. 报告语言风格",
            ["通俗易懂", "商业风", "学术论文风"],
        )

    with col3:
        feature_pref = st.radio(
            "6. 特征工程偏好",
            ["少量关键特征", "大量候选特征", "只做基础处理"],
        )

    preferences = None
    if st.button("▶️ 保存偏好设置", use_container_width=True):
        preferences = {
            "报告风格": report_style,
            "模型偏好": model_pref,
            "缺失值处理方式": missing_pref,
            "特征工程偏好": feature_pref,
            "报告语言风格": lang_style,
            "分析方向偏好": analysis_type,
        }

        st.success("✅ 偏好设置已保存！")
        st.session_state.add_preference = modeling_requirements
        st.session_state.preference_selected = preferences
        st.rerun()
    return preferences


def prep_chat(agent):
    """渲染对话式建议区"""

    with st.chat_message("assistant"):
        st.write("我是 Autostat 自动模式决策助手，很高兴为您服务！\n\n"
            "您可以在左侧边栏开启自动模式，我会协助您决策并一键完成所有分析")

    if agent.plan is not None:
        st.chat_message("assistant").write(agent.plan)
  

if __name__ == "__main__":

    st.title("偏好设置")
    st.markdown("---")

    c = st.columns(2)

    planner = st.session_state.planner_agent

    with c[0].expander('偏好设置', True):
        preferences_select()
    with c[1].expander('自动模式决策报告', True):
        prep_chat(planner)
