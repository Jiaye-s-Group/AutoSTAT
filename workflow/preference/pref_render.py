import streamlit as st


def preferences_select():

    modeling_requirements = st.text_area(
        "Describe your analysis goals and requirements",
        placeholder="For example: please help me visualize the data.",
        height=200,
    )
    st.session_state.additional_preference = modeling_requirements

    col1, col2, col3 = st.columns(3)

    with col1:
        report_style = st.radio(
            "1. Report Style",
            ["Concise & Intuitive", "Balanced", "Technical & In-depth"],
            index=1,
        )

    with col2:
        analysis_type = st.radio(
            "2. Preferred Analysis Focus",
            ["Business Analysis", "Academic Analysis", "Engineering/Product Analysis"],
        )

    with col3:
        model_pref = st.radio(
            "3. Model Preference",
            ["High Interpretability", "Best Predictive Performance", "Short Training Time"],
            index=0,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        missing_pref = st.radio(
            "4. Missing Value Handling",
            ["Simple Imputation", "Frequency-based Imputation", "Advanced Imputation (KNN/MICE)"],
        )

    with col2:
        lang_style = st.radio(
            "5. Report Language Style",
            ["Easy to Understand", "Business Style", "Academic Paper Style"],
        )

    with col3:
        feature_pref = st.radio(
            "6. Feature Engineering Preference",
            ["Few Key Features", "Many Candidate Features", "Basic Processing Only"],
        )

    preferences = None
    if st.button("▶️ Save Preferences"):
        preferences = {
            "Report Style": report_style,
            "Model Preference": model_pref,
            "Missing Value Handling": missing_pref,
            "Feature Engineering Preference": feature_pref,
            "Report Language Style": lang_style,
            "Preferred Analysis Focus": analysis_type,
        }

        st.success("✅ Preferences Saved!")
        st.session_state.preference_select = preferences

    return preferences


if __name__ == "__main__":

    st.title("Preferences")
    st.markdown("---")

    c = st.columns([2, 1])
    with c[0].expander("Preferences", True):
        preferences_select()
