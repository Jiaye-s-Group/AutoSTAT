import re

import streamlit as st
from typing import IO, List

from prompt_engineer.call_llm import LLMClient


class DataLoadingAgent(LLMClient):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.file_name = []
        self.user_input = None
        self.par_content = ""
        self.dfs = None
        self.abstract=None
        self.full = None
        self.finish_auto_task = False


    def finish_auto(self):

        self.finish_auto_task = True


    def save_file_name(self, file_name):

        self.file_name.append(file_name)


    def load_file_name(self):

        return self.file_name


    def save_dfs(self, dfs):

        self.dfs = (dfs)


    def load_dfs(self):

        return self.dfs


    def clear_file_name(self):
        
        self.file_name = []


    def read_names_from_file(self, uploaded_names_file, df_head):

        raw = uploaded_names_file.read().decode('utf-8', errors='ignore')
        try:
            uploaded_names_file.seek(0)
        except Exception:
            pass

        prompt = (
            "下面是上传的 names 和 df_head 文件内容，请仅以 Python 列表格式返回与df_head一一对应的所有属性(attribute)名称，"
            "并保持顺序，不要添加多余文字，请注意，你只需要返回一个列表，不要出现任何markdown语法：\n```\n"
            f"name文件：{raw}\n```"
            f"df_head：{df_head}\n```"
        )
        try:
            response = self.call(prompt)
            names_list = eval(response.strip())
            if isinstance(names_list, list) and all(isinstance(n, str) for n in names_list):
                col_names = names_list
            else:
                raise ValueError("LLM 输出格式不正确")
        except Exception:

            col_names = []
            attr_re = re.compile(
                r"""^@attribute\s+ 
                    ['"]?([^'"\s]+)['"]?
                    \s+.+
                """,
                re.IGNORECASE | re.VERBOSE
            )
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith('@data'):
                    break
                m = attr_re.match(line)
                if m:
                    col_names.append(m.group(1))

        counts: dict[str, int] = {}
        unique_names: List[str] = []
        for name in col_names:
            if name in counts:
                counts[name] += 1
                unique_names.append(f"{name}_{counts[name]}")
            else:
                counts[name] = 0
                unique_names.append(name)

        return unique_names


    def do_data_description(self, df, user_input=None, memory_limit=6):

        recent_memory = self.memory[-memory_limit:] if self.memory else []
        if recent_memory:
            formatted_memory = "\n".join(
                f"{m['role']}: {m['content']}" for m in recent_memory
            )
            memory_block = f"{formatted_memory}"
        else:
            memory_block = ""

        prompt = (
            "你是一名专业的数据分析助手，负责解释数据结构与业务含义。\n"
            f"- 数据维度：{df.shape[0]} 行 × {df.shape[1]} 列\n"
            f"- 列名和数据类型：{dict(zip(df.columns.tolist(), df.dtypes.astype(str).tolist()))}\n"
            f"- 前 5 行样本：\n{df.head().to_dict(orient='list')}\n\n"
            f"""- 数据解释聊天对话：
            --- 开始聊天记录 ---
            {memory_block}
            --- 结束聊天记录 ---"""
        )

        if user_input is not None:
            prompt += f"""
            请严格依据用户需求“{user_input}”，对当前数据进行深入、系统的分析。
            要求：
            1. 分析内容必须与该需求完全对应，不能添加无关推断。
            2. 结论要具体、清晰，可直接支持后续报告撰写或建模步骤。
            3. 分析语言应专业、简洁，不使用模糊或情绪化表述。
            """
        else:
            prompt += """
            以下是一个数据集的基本概览。请帮助我分析它的性质和结构，并回答以下问题：

            1. 该数据集可能来源于什么业务或研究场景？
            2. 各主要字段分别代表什么含义？若能判断，请说明其单位或数值含义。
            3. 数据中是否存在明显异常、异常分布或需要注意的特征？

            输出要求：
            - 使用自然、流畅的中文描述；
            - 采用清晰的分条结构（1、2、3）；
            - 语言客观简洁，不使用“可能”“也许”“似乎”等模糊词；
            - 重点突出数据结构、含义与潜在问题。
            """

        if st.session_state.preference_select:
            prompt += f"以下是用户的分析偏好设置：{st.session_state.preference_select}”。\n\n"
        if st.session_state.additional_preference:
            prompt += f"用户提供了以下建模目的与特殊需求：{st.session_state.additional_preference}”。\n\n"

        desc = self.call(prompt)

        return desc
    
    
    def summary_html(self):

        df = self.load_df()
        df_head = df.head()
        dtype_info = df.dtypes.astype(str)

        prompt = f"""
        你正在撰写一份数据分析报告的第一章——《数据概览与数据含义分析》。
        请根据以下输入内容，整理关键信息并进行分析说明：
        数据格式：
        {dtype_info}

        前五行数据：
        {df_head}
        
        数据解释聊天对话：
        --- 开始聊天记录 ---
        {self.memory}
        --- 结束聊天记录 ---

        额外要求：
        1. 要用流畅的自然语言
        2. 不要滥用形容词和副词，尽量用简单的动词和名词表达意思
        3. 不用"可能""也许""似乎""微妙"等模糊表述
        """.strip()

        desc = self.call(prompt)

        summary = {
                    "title": "数据导入",
                    "df": df_head,
                   "desc": desc,
                }

        return summary


    def summary_word(self):

        return self.summary_html()


    def check_abstract(self):

        if self.abstract is None:
            df = self.load_df()
            df_head = df.head()
            dtype_info = df.dtypes.astype(str)

            prompt = f"""
            这是数据分析的数据导入阶段
            数据格式：
            {dtype_info}

            前五行数据：
            {df_head}

            数据解释聊天对话：
            --- 开始聊天记录 ---
            {self.memory}
            --- 结束聊天记录 ---

            要求：
            请基于上述数据与对话内容，生成一段简洁、准确的综合摘要。
            摘要需完整呈现核心信息，便于后续自动判断该内容在报告撰写中是否需要被引用。
            """.strip()

            desc = self.call(prompt)
            self.abstract = desc

        return self.abstract


    def check_full(self):

        if self.full is None:
            df = self.load_df()
            df_head = df.head()
            dtype_info = df.dtypes.astype(str)

            self.full = (
                f"【阶段说明】这是数据分析流程中的数据导入阶段。\n"
                f"【数据格式】{dtype_info}\n"
                f"【样本预览】\n{df_head}\n"
                f"【分析对话记录】\n{self.memory}"
            )

        return self.full
