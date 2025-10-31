import streamlit as st
import streamlit_antd_components as sac
from streamlit_ace import st_ace

from utils.sanitize_code import sanitize_code
from workflow.modeling.model_training import train_execution, modeling_code_gen, train_download_model
from workflow.modeling.model_inference import infer_load_data, infer_execution
from workflow.preprocessing.preprocessing_core import prep_meta_execution


def modeling_quick_actions(agent):

    st.write("选择一个或多个model：")
    selected_models = sac.chip(
        items=[
            sac.ChipItem(label='线性回归'),
            sac.ChipItem(label='XGBoost'),
            sac.ChipItem(label='随机森林'),
            sac.ChipItem(label='神经网络'),
        ], index=[0, 2], align='center', radius='md', color='#44658C', multiple=True
    )
    
    df = agent.load_df()

    if st.button("🖋️ 快速建模"):
        if not selected_models:
            st.error("请先选择训练model。")
        else:
            with st.spinner("建模 Agent 正在生成训练脚本..."):
                raw = agent.code_generation(df.head().to_string(), selected_models)
                code = sanitize_code(raw)
                agent.save_code(code)
                agent.save_suggestion(selected_models)
                agent.save_user_selection(selected_models)
                st.success("训练脚本已生成并保存。")
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
        if st.button("▶️ 执行建模", key="modeling_run_code") or (auto and not_executed):
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
            with st.expander("训练结果", result_expand):
                if modeling_result:
                    st.subheader("训练结果")
                    try:
                        st.markdown(modeling_result)
                    except Exception:
                        st.write(modeling_result)


def modeling_inference(agent, preproc_agent):

    infer_load_data(agent)
    inference_processed_data = agent.load_inference_processed_df()
    inference_data = agent.load_inference_data()

    code = agent.load_inference_code()

    if st.button("▶️ 执行推断"):

        with st.spinner("正在对推理数据进行预处理..."):
 
            inference_data = agent.load_inference_data()
            if preproc_agent.code is not None:
                inference_processed_df = prep_meta_execution(preproc_agent, preproc_agent.code, inference_data)
                inference_data = inference_processed_df
            agent.save_inference_processed_df(inference_data)
            st.write("推断数据预览：")
            st.dataframe(inference_data.head())

        with st.spinner("建模 Agent 正在生成推理脚本..."):
            
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
        if st.button("▶️ 执行建模"):
            infer_execution(agent)


def modeling_chat(agent, auto) -> None:

    user_input = st.text_input("建模目标", "默认")
    agent.save_target(user_input)

    with st.chat_message("assistant"):
        st.write(
            "我是 Autostat 数据分析助手，很高兴为您服务！\n\n"
            "您可以在下方输入建模相关问题，或直接点击按钮获取建模建议。"
        )
        analyze_btn = st.button("🔍 建模推荐", key='modeling_suggest')
        result_placeholder = st.empty()
        
    chat_history = agent.load_memory()

    for idx, entry in enumerate(chat_history):
        bubble = st.chat_message(entry["role"])
        content = entry["content"]
        if isinstance(content, str):
            bubble.write(content)
        
    already_generated = any(
        entry["role"] == "assistant" and "模" in str(entry["content"])
        for entry in chat_history
    )
    
    if analyze_btn or (auto and not already_generated):
        st.chat_message("user").write("请帮我获取建模建议")
        agent.add_memory({"role": "user", "content": "请帮我获取建模建议"})
        with st.spinner("分析中..."):
            suggestion = agent.get_model_suggestion()
            agent.save_suggestion(suggestion)
            agent.refine_suggestions()
        st.chat_message("assistant").write(suggestion)
        agent.add_memory({"role": "assistant", "content": suggestion})
        st.chat_message("assistant").write("需要进一步优化？再次点击按钮获取下一条建议")
        agent.add_memory({"role": "assistant", "content": "需要进一步优化？再次点击按钮获取下一条建议"})

    user_input = st.chat_input("请输入您的问题，例如“如何优化这个模型”")
    if user_input:
        st.chat_message("user").write(user_input)
        agent.add_memory({"role": "user", "content": user_input})
        with st.spinner("处理中…"):
            reply = agent.get_model_suggestion(user_input)
            agent.save_suggestion(reply)
            agent.refine_suggestions()
        st.chat_message("assistant").write(reply)
        agent.add_memory({"role": "assistant", "content": reply})
        st.chat_message("assistant").write("需要进一步优化？再次点击按钮获取下一条建议")
        agent.add_memory({"role": "assistant", "content": "需要进一步优化？再次点击按钮获取下一条建议"})


if __name__ == "__main__":

    st.title("数据建模")
    st.markdown("---")

    preproc_agent = st.session_state.data_preprocess_agent
    load_agent   = st.session_state.data_loading_agent

    processed_df = preproc_agent.load_processed_df()
    if processed_df is None:
        df = load_agent.load_df()
    else:
        df = processed_df

    if df is None:
        st.warning("⚠️ 请先在数据导入页面加载数据")
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
    with c[0].expander('快速建模', True):
        modeling_quick_actions(agent)
    with c[1].expander('建模建议', True):
        modeling_chat(agent, auto)
        modeling_code_gen(agent, auto=auto)
    with c[0].expander('建模执行', expand):
        modeling_execution(agent, auto)
    # with c[0].expander('推断分析', inference_expand):
    #     modeling_inference(agent, preproc_agent)
