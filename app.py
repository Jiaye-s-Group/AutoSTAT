import sys, os
import tempfile
import streamlit as st
import time

from config import MODEL_CONFIGS
from utils.save_secrest import *
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

# Setting page configs
st.set_page_config(
    page_title="AutoSTAT",
    page_icon="🤖",
    layout="wide"
)


def init_session_state():

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "GPT-5"
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
        st.subheader("LLM Choosing")
        models = list(MODEL_CONFIGS.keys())
        st.selectbox(
            "Choose the LLM you want to use:",
            models,
            index=models.index(st.session_state.selected_model),
            key="model_selector",
            on_change=on_model_selector_change,
        )

        st.subheader("API Key Setting")
        selected = st.session_state.selected_model

        api_key_input = st.text_input(
            f"{selected} API key",
            value=st.session_state.api_keys.get(selected, ""),
            type="password",
            key="api_key_input",
        )
        st.session_state.api_keys[selected] = api_key_input

        if st.button("💾 Save Key", use_container_width=True, key="save_key"):
            # Saved at utils/.streamlit/secrets.toml.
            update_local_api_key(selected, api_key_input)
            st.session_state.api_keys[selected] = api_key_input
            st.success("Saved successfully!")
            time.sleep(1)
            st.rerun()

        if st.button("🧹 Clear Data", use_container_width=True, key="clear_data"):

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
                if st.button("🚗 Auto Mode", use_container_width=True):
                    st.session_state.auto_mode = True
                    planner.self_driving(st.session_state.data_loading_agent.load_df())
                    st.switch_page("workflow/dataloading/dataloading_render.py")
                    st.rerun()
            else:
                if st.button("❌ Exit Auto Mode", use_container_width=True):
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
        title="⚙️ Preferences",
    )
    data_loading = st.Page(
        "workflow/dataloading/dataloading_render.py",
        title="📥 Data Loading",
    )
    preprocessing = st.Page(
        "workflow/preprocessing/preprocessing_render.py",
        title="🛠️ Preprocessing",
    )
    visualization = st.Page(
        "workflow/visualization/viz_render.py",
        title="📊 Visualization",
    )
    coding_modeling = st.Page(
        "workflow/modeling/modeling_render.py",
        title="🧠 Modeling",
    )
    report = st.Page(
        "workflow/report/report_render.py",
        title="📝 Report Generation",
    )

    # Navigation
    pg = st.navigation(
        {
            "Functions": [data_loading, preprocessing, visualization, coding_modeling, report],
            "Setting": [preference]
        }
    )
    pg.run()
    
    
if __name__ == "__main__":
    run_app()

