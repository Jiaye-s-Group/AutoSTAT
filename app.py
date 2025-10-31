import sys, os
import tempfile
import streamlit as st

from config import MODEL_CONFIGS
from utils.save_secrets import *
from prompt_engineer.sec1_call_llm import DataLoadingAgent
from prompt_engineer.sec2_call_llm import DataPreprocessAgent
from prompt_engineer.sec3_call_llm import VisualizationAgent
from prompt_engineer.sec4_call_llm import ModelingCodingAgent
from prompt_engineer.sec5_call_llm import ReportAgent
from prompt_engineer.planner import PlannerAgent

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

import numpy as np
np.set_printoptions(edgeitems=250, threshold=501)

sys.path.append(os.path.dirname(__file__))

SECRETS_PATH = Path(".streamlit") / "secrets.toml"


st.set_page_config(
    page_title="Autostat",
    page_icon="🤖",
    layout="wide"
)


def init_session_state():

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "DeepSeek"
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = load_local_api_keys()
    if 'auto_mode' not in st.session_state:
        st.session_state.auto_mode = False

    if 'preference_select' not in st.session_state:
        st.session_state.preference_select = None
    if 'additional_preference' not in st.session_state:
        st.session_state.additional_preference = None
    if "from_auto" not in st.session_state:
        st.session_state.from_auto = False

    if 'data_loading_agent' not in st.session_state:
        st.session_state.data_loading_agent = DataLoadingAgent(
            api_keys=st.session_state.api_keys,
            model_configs=MODEL_CONFIGS,
            model=st.session_state.selected_model
        )
    if 'data_preprocess_agent' not in st.session_state:
        st.session_state.data_preprocess_agent = DataPreprocessAgent(
            api_keys=st.session_state.api_keys,
            model_configs=MODEL_CONFIGS,
            model=st.session_state.selected_model
        )
    if 'visualization_agent' not in st.session_state:
        st.session_state.visualization_agent = VisualizationAgent(
            api_keys=st.session_state.api_keys,
            model_configs=MODEL_CONFIGS,
            model=st.session_state.selected_model
        )
    if 'modeling_coding_agent' not in st.session_state:
        st.session_state.modeling_coding_agent = ModelingCodingAgent(
            api_keys=st.session_state.api_keys,
            model_configs=MODEL_CONFIGS,
            model=st.session_state.selected_model
        )
    if 'report_agent' not in st.session_state:
        st.session_state.report_agent = ReportAgent(
            api_keys=st.session_state.api_keys,
            model_configs=MODEL_CONFIGS,
            model=st.session_state.selected_model
        )
    if 'planner_agent' not in st.session_state:
        st.session_state.planner_agent = PlannerAgent(
            api_keys=st.session_state.api_keys,
            model_configs=MODEL_CONFIGS,
            model=st.session_state.selected_model
        )


def on_model_selector_change():
    """
    Callback when the model selector in the sidebar changes.
    """
    st.session_state.selected_model = st.session_state.model_selector
    

def run_app():
    """
    Main entry point to render the Streamlit app.
    """
    init_session_state()
    with st.sidebar:
        st.subheader("选择大模型")
        models = list(MODEL_CONFIGS.keys())
        st.selectbox(
            "选择要使用的大模型",
            models,
            index=models.index(st.session_state.selected_model),
            key="model_selector",
            on_change=on_model_selector_change,
        )

        st.subheader("API 密钥设置")
        selected = st.session_state.selected_model

        api_key_input = st.text_input(
            f"{selected} API 密钥",
            value=st.session_state.api_keys.get(selected, ""),
            type="password",
            key="api_key_input",
        )


        if st.button("💾 保存密钥", use_container_width=True, key="save_key"):
            # 保存在 utils/.streamlit/secrets.toml
            update_local_api_key(selected, api_key_input)

            st.session_state.api_keys[selected] = api_key_input
            st.success("已保存")
            st.rerun()

        if st.button("🧹 清空数据", use_container_width=True, key="clear_data"):

            st.session_state.data_loading_agent = DataLoadingAgent(
                api_keys=st.session_state.api_keys,
                model_configs=MODEL_CONFIGS,
                model=st.session_state.selected_model
            )
            st.session_state.data_preprocess_agent = DataPreprocessAgent(
                api_keys=st.session_state.api_keys,
                model_configs=MODEL_CONFIGS,
                model=st.session_state.selected_model
            )
            st.session_state.visualization_agent = VisualizationAgent(
                api_keys=st.session_state.api_keys,
                model_configs=MODEL_CONFIGS,
                model=st.session_state.selected_model
            )
            st.session_state.modeling_coding_agent = ModelingCodingAgent(
                api_keys=st.session_state.api_keys,
                model_configs=MODEL_CONFIGS,
                model=st.session_state.selected_model
            )
            st.session_state.report_agent = ReportAgent(
                api_keys=st.session_state.api_keys,
                model_configs=MODEL_CONFIGS,
                model=st.session_state.selected_model
            )
            st.session_state.planner_agent = PlannerAgent(
                api_keys=st.session_state.api_keys,
                model_configs=MODEL_CONFIGS,
                model=st.session_state.selected_model
            )
            st.session_state.auto_mode = False
            st.rerun()

        if st.session_state.data_loading_agent.load_df() is not None:
            planner = st.session_state.planner_agent

            if st.session_state.auto_mode is False:
                if st.button("🚗 自动模式", use_container_width=True):
                    st.session_state.auto_mode = True
                    planner.self_driving(st.session_state.data_loading_agent.load_df())
                    st.switch_page("workflow/dataloading/dataloading_render.py")
                    st.rerun()
            else:
                if st.button("❌ 结束自动模式", use_container_width=True):
                    st.session_state.auto_mode = False
                    st.session_state.planner_agent = PlannerAgent(
                    api_keys=st.session_state.api_keys,
                    model_configs=MODEL_CONFIGS,
                    model=st.session_state.selected_model
                    )
                    st.rerun()

        st.image(
            "logo/logo_big.png",
            use_container_width=True
        )

    # Define pages
    preference = st.Page(
        "workflow/preference/pref_render.py",
        title="⚙️ 偏好设置",
    )
    data_loading = st.Page(
        "workflow/dataloading/dataloading_render.py",
        title="📥 数据导入",
    )
    preprocessing = st.Page(
        "workflow/preprocessing/preprocessing_render.py",
        title="🛠️ 数据预处理",
    )
    visualization = st.Page(
        "workflow/visualization/viz_render.py",
        title="📊 数据可视化",
    )
    report = st.Page(
        "workflow/report/report_render.py",
        title="📝 报告生成",
    )
    coding_modeling = st.Page(
        "workflow/modeling/modeling_render.py",
        title="🧠 建模分析",
    )
    # Navigation
    pg = st.navigation(
        {
            "功能": [data_loading, preprocessing, visualization, coding_modeling, report],
            "设置": [preference]
        }
    )
    pg.run()
    
if __name__ == "__main__":
    run_app()

