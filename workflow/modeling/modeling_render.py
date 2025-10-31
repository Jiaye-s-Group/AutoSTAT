import streamlit as st
import streamlit_antd_components as sac
from streamlit_ace import st_ace

from utils.sanitize_code import sanitize_code
from workflow.modeling.model_training import train_execution, modeling_code_gen, train_download_model
from workflow.modeling.model_inference import infer_load_data, infer_execution
from workflow.preprocessing.preprocessing_core import prep_meta_execution


def modeling_quick_actions(agent):

    st.write("Select one or more models:")
    selected_models = sac.chip(
        items=[
            sac.ChipItem(label='Linear Regression'),
            sac.ChipItem(label='XGBoost'),
            sac.ChipItem(label='Random Forest'),
            sac.ChipItem(label='Neural Network'),
        ], index=[0, 2], align='center', radius='md', color='#44658C', multiple=True
    )
    
    df = agent.load_df()

    if st.button("🖋️ Quick Modeling"):
        if not selected_models:
            st.error("Please select training models first.")
        else:
            with st.spinner("Modeling Agent is generating training script..."):
                raw = agent.code_generation(df.head().to_string(), selected_models)
                code = sanitize_code(raw)
                agent.save_code(code)
                agent.save_suggestion(selected_models)
                agent.save_user_selection(selected_models)
                st.success("Training script generated and saved.")
                st.rerun()
                    
    return selected_models


def modeling_execution(agent, auto = False) -> None:

    code = agent.load_code()

    edited = st_ace(
        value=code,
        height=450,
        theme="tomorrow_night",
        language="python",
        auto_update=True
    )

    not_executed = agent.load_modeling_result() == None

    if edited is not None:
        if st.button("▶️ Run Modeling", key="modeling_run_code") or (auto and not_executed):
            code = sanitize_code(edited)
            agent.save_code(code)
            train_execution(agent)
            agent.finish_auto()
            st.rerun()

        modeling_result = agent.load_modeling_result()
        if modeling_result is None:
            result_expand = False
        else:
            result_expand = True
            train_download_model(agent)
            with st.expander("Training Results", result_expand):
                if modeling_result:
                    st.subheader("Training Results")
                    try:
                        st.markdown(modeling_result)
                    except Exception:
                        st.write(modeling_result)


def modeling_inference(agent, preproc_agent):

    infer_load_data(agent)
    inference_processed_data = agent.load_inference_processed_df()
    inference_data = agent.load_inference_data()

    code = agent.load_inference_code()

    if st.button("▶️ Run Inference"):

        with st.spinner("Preprocessing inference data..."):
 
            inference_data = agent.load_inference_data()
            if preproc_agent.code is not None:
                inference_processed_df = prep_meta_execution(preproc_agent, preproc_agent.code, inference_data)
                inference_data = inference_processed_df
            agent.save_inference_processed_df(inference_data)
            st.write("Inference data preview:")
            st.dataframe(inference_data.head())

        with st.spinner("Modeling Agent is generating inference script..."):
            
            raw = agent.code_generation_for_inference(agent.load_code(), inference_data.head())
            code = sanitize_code(raw)
            agent.save_inference_code(code)

    if code is not None:
        edited_code = st_ace(
            value=code,
            height=450,
            theme="tomorrow_night",
            language="python",
            auto_update=True
        )
        agent.save_inference_code(code)
        if st.button("▶️ Run Modeling"):
            infer_execution(agent)


def modeling_chat(agent, auto) -> None:

    user_input = st.text_input("Modeling Objective", "Default")
    agent.save_target(user_input)

    with st.chat_message("assistant"):
        st.write(
            "I am the Autostat data analysis assistant, pleased to serve you!\n\n"
            "You can enter modeling-related questions below or directly click the button to get modeling suggestions."
        )
        analyze_btn = st.button("🔍 Modeling Recommendation", key='modeling_suggest')
        result_placeholder = st.empty()
        
    chat_history = agent.load_memory()

    for idx, entry in enumerate(chat_history):
        bubble = st.chat_message(entry["role"])
        content = entry["content"]
        if isinstance(content, str):
            bubble.write(content)
        
    already_generated = any(
        entry["role"] == "assistant" and "modeling" in str(entry["content"]).lower()
        for entry in chat_history
    )
    
    if analyze_btn or (auto and not already_generated):
        st.chat_message("user").write("Please help me get modeling suggestions")
        agent.add_memory({"role": "user", "content": "Please help me get modeling suggestions"})
        with st.spinner("Analyzing..."):
            suggestion = agent.get_model_suggestion()
            agent.save_suggestion(suggestion)
            agent.refine_suggestions()
        st.chat_message("assistant").write(suggestion)
        agent.add_memory({"role": "assistant", "content": suggestion})
        st.chat_message("assistant").write("Need further optimization? Click the button again for the next suggestion")
        agent.add_memory({"role": "assistant", "content": "Need further optimization? Click the button again for the next suggestion"})

    user_input = st.chat_input("Please enter your question, e.g., 'How to optimize this model'")
    if user_input:
        st.chat_message("user").write(user_input)
        agent.add_memory({"role": "user", "content": user_input})
        with st.spinner("Processing..."):
            reply = agent.get_model_suggestion(user_input)
            agent.save_suggestion(reply)
            agent.refine_suggestions()
        st.chat_message("assistant").write(reply)
        agent.add_memory({"role": "assistant", "content": reply})
        st.chat_message("assistant").write("Need further optimization? Click the button again for the next suggestion")
        agent.add_memory({"role": "assistant", "content": "Need further optimization? Click the button again for the next suggestion"})


if __name__ == "__main__":

    st.title("Modeling")
    st.markdown("---")

    preproc_agent = st.session_state.data_preprocess_agent
    load_agent   = st.session_state.data_loading_agent

    processed_df = preproc_agent.load_processed_df()
    if processed_df is None:
        df = load_agent.load_df()
    else:
        df = processed_df

    if df is None:
        st.warning("⚠️ Please load data on the data import page first")
        st.stop()

    agent = st.session_state.modeling_coding_agent
    agent.add_df(df)
    planner = st.session_state.planner_agent
    auto = planner.modeling_auto

    if st.session_state.auto_mode == True:
        if (agent.finish_auto_task == True and planner.switched_modeling == False) or planner.modeling_auto == False:
            planner.finish_modeling_auto()
            st.switch_page("workflow/report/report_render.py")

    code = agent.load_code()
    if code is None:
        expand = False
    else:
        expand = True

    inference_model = agent.load_best_model()
    if inference_model is None:
        inference_expand = False
    else:
        inference_expand = True

    c = st.columns(2)
    with c[0].expander('Quick Modeling', True):
        modeling_quick_actions(agent)
    with c[1].expander('Modeling Suggestions', True):
        modeling_chat(agent, auto)
        modeling_code_gen(agent, auto=auto)
    with c[0].expander('Modeling Execution', expand):
        modeling_execution(agent, auto)
    # with c[0].expander('Inference Analysis', inference_expand):
    #     modeling_inference(agent, preproc_agent)
