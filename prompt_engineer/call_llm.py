import re
from openai import OpenAI, OpenAIError
from anthropic import Anthropic, AnthropicError
import requests
import json

import streamlit as st
import pandas as pd
import numpy as np
from config import MODEL_CONFIGS
from typing import IO, List, Dict
from zai import ZhipuAiClient

class LLMClient:
    def __init__(self, model_configs: dict, api_keys: dict, model: str):

        self.model = model
        self.model_configs = model_configs
        self.api_keys = api_keys
        self.memory = []
        self.df = None

    def call(self, prompt) -> str:

        model_name = st.session_state.selected_model
        config = self.model_configs.get(model_name, {})
        api_key = self.api_keys.get(model_name)

        if not api_key:
            return "Please enter the API key in \"API Key Setting\" first."
        
        system_msg = (
            "You are a professional data analysis assistant."
        )

        try:
            if model_name == "GPT-4o" or model_name == "GPT-5" or model_name == "DeepSeek" or model_name == "Qwen" or model_name == "Claude" or model_name == "Doubao":
                try:
                    client = OpenAI(
                        api_key=api_key,
                        base_url=config["api_base"]
                    )

                    resp = client.chat.completions.create(
                        model=config["model_name"],
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": prompt},
                        ],
                        stream = False
                    )
                    return resp.choices[0].message.content
                
                except OpenAIError as e:
                    # Catch all errors defined by OpenAI SDK
                    st.error(f"API call failed: {str(e)}")
                    # Remind user
                    return "Call failed, please check the API key or network connection."
                except Exception as e:
                    # Catch other unexpected errors, like net connection error
                    st.error(f"Unexpected Error: {str(e)}")
                    return "An unknown error occurred."

            elif model_name == "Zhipu":
                client = ZhipuAiClient(api_key=api_key)
                response = client.chat.completions.create(
                    model=config["model_name"],
                    messages=[{"role": "system", "content": "You are a professional data analysis assistant."},
                        {"role": "user", "content": prompt}],
                    thinking={
                        "type":"enabled"
                    }
                )
                if response:
                    print(response.choices[0].message)
                    desc = response.choices[0].message.content if hasattr(response.choices[0].message, "content") else str(response.choices[0].message)
                    return desc.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip()

                st.error(f"API Call failed: {response.text}")
                return "Call failed, please check the API key or network connection."

            else:
                return f"Not supported LLM: {model_name}"

        except Exception as e:
            st.error(f"{model_name} calling exception: {e}")
            return "Call failed, please check the API key or network connection."

    
    def add_memory(self, entry: Dict[str, str]) -> None:

        self.memory.append(entry)


    def load_memory(self) -> List[Dict[str, str]]:

        return self.memory


    def clear_memory(self) -> None:

        self.memory.clear()


    def add_df(self, input_df) -> None:

 
        
        self.df = input_df
        

    def load_df(self) -> pd.DataFrame:
        
        return self.df
    

    def clear_df(self) -> None:

        self.df = None


    def has_df(self) -> bool:

        return self.df == None