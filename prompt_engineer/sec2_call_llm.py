import numpy as np
import pandas as pd
import streamlit as st

from prompt_engineer.call_llm import LLMClient

class DataPreprocessAgent(LLMClient):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.processed_df = None
        self.code = None
        self.preprocessing_suggestions = None
        self.allowed_libs = [
            "numpy",
            "pandas",
            "sklearn.impute",
            "sklearn.preprocessing",
            "sklearn.compose",
            "sklearn.pipeline"
        ]
        self.par_content = ""
        self.error = None
        self.user_input = None
        self.refined_suggestions = ""
        self.abstract=None
        self.full = None
        self.finish_auto_task = False
        self.debug_num = 0


    def finish_auto(self):

        self.finish_auto_task = True


    def save_code(self, code):

        self.code = code


    def load_code(self):

        return self.code


    def save_user_input(self, user_input):

        self.user_input = user_input


    def load_user_input(self):

        return self.user_input


    def save_error(self, error):

        self.error = error


    def load_error(self):

        return self.error
    

    def save_preprocessing_suggestions(self, suggestions):
        
        self.preprocessing_suggestions = suggestions


    def load_preprocessing_suggestions(self):
        
        return self.preprocessing_suggestions
        

    def save_processed_df(self, processed_df):

        if not isinstance(processed_df, pd.DataFrame):
            if isinstance(processed_df, np.ndarray):
                processed_df = pd.DataFrame(processed_df)
            else:
                raise TypeError(f"期望 pandas.DataFrame 或 numpy.ndarray，收到 {type(processed_df)}")

        self.processed_df = processed_df


    def load_processed_df(self):

        return self.processed_df
    

    def load_refined_suggestions(self):
        return self.refined_suggestions
    

    def save_refined_suggestions(self, refined_suggestions):
        self.refined_suggestions = refined_suggestions


    def refine_suggestions(self, df_head):
        """将 LLM 返回的预处理推荐进行信息提取"""

        suggestion = self.load_preprocessing_suggestions()

        prompt = f"""
        请根据以下预处理建议，概括数据集中每一列的推荐预处理方法。

        数据示例:
        {df_head}

        详细预处理建议:
        {suggestion}

        输出要求（必须严格遵守）：
        1. 输出格式：列名：推荐预处理方法；每条独立换行。
        2. 每列最多给出三个推荐方法，多个方法用逗号分隔。
        3. 输出必须为纯文本，不使用任何 Markdown 标记。
        4. 每个方法的长度不得超过20个汉字，若包含英文则不超过10个单词。"""

        refined_suggestions = self.call(prompt)
        self.refined_suggestions = refined_suggestions

        return refined_suggestions
        

    def get_preprocessing_suggestions(
        self, 
        user_input=None,
        memory_limit=6,
    ):

        df = self.load_df()

        # 基本统计
        n_rows, n_cols = df.shape
        dtype_counts = df.dtypes.value_counts().to_dict()
        missing_total = int(df.isnull().sum().sum())
        missing_by_col = df.isnull().mean().mul(100).round(2).to_dict()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 整理 memory 片段
        recent_memory = self.memory[-memory_limit:] if self.memory else []
        if recent_memory:
            formatted_memory = "\n".join(
                f"{m['role']}: {m['content']}" for m in recent_memory
            )
            memory_block = f"{formatted_memory}"
        else:
            memory_block = ""

        prompt = f"""
        你是一名资深的数据预处理专家，负责为数据分析报告提供高质量的预处理建议。

        === 数据概览 ===
        - 数据规模：{n_rows} 行 × {n_cols} 列
        - 数据类型分布：{dtype_counts}
        - 缺失值总数：{missing_total}
        - 各列缺失率：{missing_by_col}
        - 数值型列：{num_cols}
        - 历史上下文（仅供参考）：{memory_block}
        """

        if user_input is None:
            prompt += """
            === 请对每一列进行逐项分析（注意，是逐列分析） ===
            请针对每一列依次说明以下四个方面：

            1. **数据类型**：明确该列的数据类型，若存在混合类型或异常值类型，请指出。
            2. **缺失值处理建议**：说明该列的缺失值处理策略；若建议调整，请指明具体“缺失值处理 策略”操作。
            3. **异常值处理建议**：说明该列的异常检测与处理方案；若需调整，请说明“异常值处理 策略或阈值”操作。
            4. **标准化建议**：说明是否建议标准化或缩放，并在需要时指出“标准化处理 策略”操作。

            输出格式要求：
            - 按“列名 + 分点说明（1–4）”的形式分段输出；
            - 每一列独立成段，并以换行分隔；
            - 使用清晰、简洁的专业语言。
            """
        else:
            prompt += f"""
            === 用户新需求 ===
            {user_input}

            请结合以上数据概览与历史上下文，针对该需求，给出下一步操作。
            可考虑的操作包括：缺失值处理、异常值检测与修正、标准化或归一化、特征类型调整等。
            输出应保持结构化与连贯性，避免重复说明。
            """

        if st.session_state.preference_select:
            prompt += f"以下是用户的分析偏好设置：{st.session_state.preference_select}”。\n\n"
        if st.session_state.additional_preference:
            prompt += f"用户提供了以下建模目的与特殊需求：{st.session_state.additional_preference}”。\n\n"

        suggestions = self.call(prompt)

        return suggestions
    

    def code_generation(self, df_head, user_prompt):

        allowed = ", ".join(self.allowed_libs)

        prompt = f"""
        请**严格只输出纯 Python 代码**，不得包含以下内容：
        - 解释性文字、注释、示例；
        - Markdown 代码块标记（禁止出现 ``` 或 ```python 等）；
        - 任何多余输出（如 print、全局变量赋值等）。

        === 运行环境说明 ===
        运行环境中已提供以下对象与库：
        - pandas DataFrame 变量：`df`
        - 库：numpy (np)、SimpleImputer、StandardScaler、MinMaxScaler、RobustScaler、
        OneHotEncoder、OrdinalEncoder、LabelEncoder、FunctionTransformer、
        ColumnTransformer、Pipeline。
        若所需功能在这些库中不存在，请自行写 Python code 实现。

        === 生成要求 ===
        1. 若有用户需求，请优先满足用户需求（优先级高于 LLM 返回的通用建议）。
        2. 若建议指出某列“无需处理”，则对该列不进行任何操作。
        3. 禁止导入其他库、禁止文件读写。
        4. 所有括号（圆括号、方括号、大括号）必须成对闭合，不得错位或遗漏。
        5. 对类别特征，可使用 OneHotEncoder 或 OrdinalEncoder；
        若为单列字符串／类别列，请使用 LabelEncoder 或 OrdinalEncoder，不得 passthrough。
        6. 在构建 ColumnTransformer 前，需检测并处理“混合型列”
        —— 即同时包含数值和字符串的列，
        使用 `FunctionTransformer(lambda x: x.astype(str))` 将其统一为字符串类型。
        7. ColumnTransformer 的 transformers 中仅包含经过上述处理的列。
        8. 使用 OneHotEncoder 时，若输出稀疏矩阵，请确保所有输入特征均为数值类型。
        9. 若 df 中存在重复表头（如第 0 行与 header 相同），需自动检测并删除重复表头行。
        10. 确保预处理后的 DataFrame 中每一列均有明确列名。
        11. 脚本最后仅保留一行结果：
            `process_df = ...`  
            不允许出现 print、显示语句或其他多余输出。

        === 输入数据示例 ===
        {df_head}

        === 用户指定需求 ===
        {user_prompt}

        请严格依据以上要求，输出完整且可直接执行的 Python 代码（纯代码块，无额外说明）。
        """.strip()

        if self.error is not None:
            if self.debug_num < 5 :
                self.debug_num += 1

                prompt += f"""
                上次生成的代码运行失败。
                【错误信息】：
                {self.error}

                【原始代码】：
                {self.code}

                请在不输出任何解释性文字的情况下，推理并理解导致错误的根本原因，

                要求：
                1. 不输出任何分析、解释或说明（包括文字、列表或注释段落）；
                2. 可在代码内部使用简短注释说明关键修改；
                3. 若错误源于逻辑、数据结构或函数使用不当，请自行调整；
                4. 若依赖库方法不适用，可自行实现替代函数；
                5. 生成的代码必须可独立运行，无语法错误；
                6. 保持整体逻辑与原代码意图一致，仅做必要修正。
                """

            else:
                self.debug_num = 0

        if self.user_input is not None:
            prompt += f"用户需求：{self.user_input}。\n请严格遵循并优先执行该需求，其优先级高于所有其他建议或规则。\n"

        if self.refined_suggestions is not None:
            prompt += f"LLM返回的预处理建议：{self.refined_suggestions}"

        if st.session_state.preference_select:
            prompt += f"以下是用户的分析偏好设置：{st.session_state.preference_select}”。\n\n"

        if st.session_state.additional_preference:
            prompt += f"用户提供了以下建模目的与特殊需求：{st.session_state.additional_preference}”。\n\n"

        raw = self.call(prompt)
        return raw


    def summary_html(self):

        if self.code is None:
            summary = None
            return summary

        else:
            processed_df = self.load_processed_df()
            prompt = f"""
            你正在撰写数据分析报告的第二章——《数据预处理与标准化》。
            请根据以下输入内容，提炼关键信息并撰写相应分析段落。

            - 预处理代码：
            {self.code}

            - 预处理结果（数据示例）：
            {processed_df.head()}

            {f"- 预处理建议对话记录：{self.load_memory}" if self.load_memory else ""}

            撰写要求：
            1. 使用流畅、自然的中文表达；
            2. 语言应简洁、准确，避免过多形容词或副词；
            3. 不使用“可能”“也许”“似乎”“微妙”等模糊表述；
            4. 不添加大标题，可使用自然段进行叙述；
            5. 内容需逻辑清晰，体现代码与结果之间的分析关联。

            """.strip()

            desc = self.call(prompt)

            summary = {
                        "title": "数据预处理",
                        "desc": desc,
                        "processed_df": self.processed_df.head(),
                        "code": self.code,
                    }

        return summary


    def summary_word(self):

        return self.summary_html()


    def check_abstract(self):

        if self.abstract is None:

            processed_df = self.load_processed_df()

            if self.code is None:
                self.abstract = None
            if processed_df is None:
                self.abstract = None

            else:

                memory = f"【预处理建议对话记录】\n{self.load_memory}\n" if self.load_memory else ""

                prompt = f"""
                这是数据分析流程中的“数据预处理与标准化”阶段。

                【预处理代码】
                {self.code}

                【预处理结果（前五行）】
                {processed_df.head()}

                {memory}
                请在确保信息准确完整的前提下，将上述内容概括为一段简洁的文字摘要。
                要求：
                1. 语言自然流畅，保持客观和专业；
                2. 内容应涵盖关键点（包括主要预处理步骤与结果特征）；
                3. 重点在于“说明核心信息”，而非逐行描述；
                4. 生成的摘要应可用于报告编写时判断该部分是否需要引用。
                """.strip()

                desc = self.call(prompt)
                self.abstract = desc

        return self.abstract


    def check_full(self):
        
        if self.full is None:
            processed_df = self.load_processed_df()
            if self.code is None:
                self.full = None
            else:
                content = f"""
                【阶段说明】这是数据分析流程中的数据预处理阶段。
                【预处理代码】{self.code}
                【预处理结果前五行】{processed_df.head()}
                """.strip()
                if self.load_memory is not None:
                    content += f"\n【预处理建议聊天对话】{self.load_memory}"

                self.full = content

        return self.full
