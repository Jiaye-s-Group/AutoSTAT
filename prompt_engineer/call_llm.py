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
            return "请先在设置中配置 API 密钥"
        
        system_msg = (
            "你是一个专业的数据分析助手。"
        )

        try:
            if model_name == "GPT-4o" or model_name == "GPT-5" or model_name == "DeepSeek" or model_name == "通义千问" or model_name == "Claude" or model_name == "豆包":
                try:
                    client = OpenAI(
                        api_key=api_key,
                        base_url=config["api_base"]
                    )
                    
                    # 使用新的 API 调用方式
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
                    # 这里可以捕获所有OpenAI SDK定义的错误
                    st.error(f"API调用失败: {str(e)}")
                    # 记录日志或提示用户
                    return "调用失败，请检查密钥或网络"
                except Exception as e:
                    # 捕获其他非预期的异常，如网络问题
                    st.error(f"发生未知错误: {str(e)}")
                    return "发生未知错误"

            elif model_name == "智谱AI":
                client = ZhipuAiClient(api_key=api_key)
                response = client.chat.completions.create(
                    model=config["model_name"],
                    messages=[{"role": "system", "content": "你是一个专业的数据分析助手。"},
                        {"role": "user", "content": prompt}],
                    thinking={
                        "type":"enabled"
                    }
                )
                if response:
                    print(response.choices[0].message)
                    desc = response.choices[0].message.content if hasattr(response.choices[0].message, "content") else str(response.choices[0].message)
                    return desc.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip()

                st.error(f"智谱调用失败：{response.text}")
                return "调用失败，请检查密钥或网络"

            else:
                return f"暂不支持模型：{model_name}"

        except Exception as e:
            st.error(f"{model_name} 调用异常：{e}")
            return "大模型调用失败，请检查 API 密钥或网络连接"

    
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