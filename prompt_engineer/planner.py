import re
import ast
import json
import traceback

import streamlit as st
from typing import IO, List

from prompt_engineer.call_llm import LLMClient


class PlannerAgent(LLMClient):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.loading_auto = False
        self.prep_auto = False
        self.vis_auto = False
        self.modeling_auto = False
        self.report_auto = False

        self.switched_loading = False
        self.switched_prep = False
        self.switched_vis = False
        self.switched_modeling = False
        self.switched_report = False


    def self_driving(self, df, user_input=None) -> str:

        prompt = (
            f"下面是一个数据集的基本信息，请你根据它和用户的需求，判断需要开启哪些分析步骤：\n\n"
            f"- 数据维度：{df.shape[0]} 行 × {df.shape[1]} 列\n"
            f"- 列名和数据类型：{dict(zip(df.columns.tolist(), df.dtypes.astype(str).tolist()))}\n"
            f"- 前 5 行样本：\n{df.head().to_dict(orient='list')}\n\n"
        )

        if st.session_state.preference_select:
            prompt += f"以下是用户的分析偏好设置：{st.session_state.preference_select}”。\n\n"
        if st.session_state.additional_preference:
            prompt += f"用户提供了以下建模目的与特殊需求：{st.session_state.additional_preference}”。\n\n"

        prompt += """
        你需要在以下 5 个步骤中，对每个步骤分别判断是否应该开启（True / False）：
        1. loading_auto —— 是否需要对数据列名进行初步分析？
        2. prep_auto —— 是否需要做数据预处理或清洗？
        3. vis_auto —— 是否需要做数据可视化？
        4. modeling_auto —— 是否需要建模或统计分析？
        5. report_auto —— 是否需要生成分析报告？

        必须以 **JSON 格式** 输出你的判断结果，如：
        {
            "loading_auto": true,
            "prep_auto": false,
            "vis_auto": true,
            "modeling_auto": true,
            "report_auto": true
        }

        不要输出其他内容。
        """

        plan_text = self.call(prompt)
        try:
            plan_dict = json.loads(plan_text)
        except json.JSONDecodeError:
            plan_text_fixed = plan_text.strip().strip('```json').strip('```')
            plan_dict = json.loads(plan_text_fixed)

        # self.loading_auto = bool(plan_dict.get("loading_auto", False))
        self.loading_auto = True
        self.prep_auto = bool(plan_dict.get("prep_auto", False))
        self.vis_auto = bool(plan_dict.get("vis_auto", False))
        self.modeling_auto = bool(plan_dict.get("modeling_auto", False))
        self.report_auto = bool(plan_dict.get("report_auto", False))


    def finish_loading_auto(self) -> str:

        self.switched_loading = True


    def finish_prep_auto(self) -> str:

        self.switched_prep = True


    def finish_vis_auto(self) -> str:

        self.switched_vis = True


    def finish_modeling_auto(self) -> str:

        self.switched_modeling = True


    def finish_report_auto(self) -> str:

        self.switched_report = True


def _extract_first_json(text: str):

    if not text:
        return None
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def _safe_parse_json(text: str):

    if not text or not text.strip():
        return None, text, "empty"

    try:
        return json.loads(text), text, None
    except Exception as e1:
        pass

    try:
        cleaned = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'```', '', cleaned)
        cleaned = cleaned.strip()
        return json.loads(cleaned), cleaned, None
    except Exception:
        pass

    try:
        sub = _extract_first_json(text)
        if sub:
            return json.loads(sub), sub, None
    except Exception:
        pass

    try:
        literal = ast.literal_eval(text)
        if isinstance(literal, dict):
            return literal, text, None
    except Exception:
        pass

    try:
        sub = _extract_first_json(text)
        if sub:
            literal = ast.literal_eval(sub)
            if isinstance(literal, dict):
                return literal, sub, None
    except Exception:
        pass

    return None, text, "unable_to_parse"