import re
import json
import ast
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
            f"Below is the basic information of a dataset. Please, based on it and the user's needs, determine which analysis steps need to be initiated:\n\n"
            f"- Data dimensions: {df.shape[0]} rows * {df.shape[1]} columns\n"
            f"- Column names and data types: {dict(zip(df.columns.tolist(), df.dtypes.astype(str).tolist()))}\n"
            f"- First 5 rows sample: \n{df.head().to_dict(orient='list')}\n\n"
        )

        if st.session_state.preference_select:
            prompt += f"The user's analysis preferences are as follows: {st.session_state.preference_select}.\n\n"
        if st.session_state.additional_preference:
            prompt += f"The user has provided the following modeling goals and special requirements: {st.session_state.additional_preference}.\n\n"

        prompt += """
        You need to determine for each of the following 5 steps whether it should be initiated (True / False):
        1. loading_auto —— Is a preliminary analysis of the data column names needed?
        2. prep_auto —— Is data preprocessing or cleaning required?
        3. vis_auto —— Is data visualization needed?
        4. modeling_auto —— Is modeling or statistical analysis required?
        5. report_auto —— Is generating an analysis report needed?

        You must output your judgment result in **JSON format**, for example:
        {
            "loading_auto": true,
            "prep_auto": false,
            "vis_auto": true,
            "modeling_auto": true,
            "report_auto": true
        }

        Do not output any other content.
        """

        plan_text = self.call(prompt)
        print(plan_text)
        try:
            plan_dict = json.loads(plan_text)
        except json.JSONDecodeError:
            plan_text_fixed = plan_text.strip().strip('```json').strip('```')
            plan_dict = json.loads(plan_text_fixed)

        print(plan_dict)
        self.loading_auto = bool(plan_dict.get("loading_auto", False))
        self.prep_auto = bool(plan_dict.get("prep_auto", False))
        self.vis_auto = bool(plan_dict.get("vis_auto", False))
        self.modeling_auto = bool(plan_dict.get("modeling_auto", False))
        # self.modeling_auto = False
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
    """Extract the first top-level curly brace JSON substring from the text (using the pairing counting method), if not found, return None."""
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
    """
    Attempt multiple strategies to parse LLM output as dict: 
    1) Direct json.loads
    2) Remove Markdown code fence then loads
    3) Extract first complete curly brace block then loads
    4) ast.literal_eval as last resort (accepts Python dict style)
    Returns (dict_or_None, used_text, error_message_or_None)
    """
    if not text or not text.strip():
        return None, text, "empty"
    # 1) Direct attempt
    try:
        return json.loads(text), text, None
    except Exception as e1:
        pass

    # 2) Remove ```json / ``` fence
    try:
        cleaned = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'```', '', cleaned)
        cleaned = cleaned.strip()
        return json.loads(cleaned), cleaned, None
    except Exception:
        pass

    # 3) Extract first matching top-level { ... } block
    try:
        sub = _extract_first_json(text)
        if sub:
            return json.loads(sub), sub, None
    except Exception:
        pass

    # 4) ast.literal_eval compatible with Python dict format (single quotes, etc.)
    try:
        literal = ast.literal_eval(text)
        if isinstance(literal, dict):
            return literal, text, None
    except Exception:
        pass

    # 5) Try again with extracted substring using literal_eval (to handle single quotes)
    try:
        sub = _extract_first_json(text)
        if sub:
            literal = ast.literal_eval(sub)
            if isinstance(literal, dict):
                return literal, sub, None
    except Exception:
        pass

    # Finally, return None with error message
    return None, text, "unable_to_parse"