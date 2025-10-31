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
        """
        Extract attribute names from the uploaded .names/.arff files.
        Prefer to use LLM to identify attribute names in the @attribute lines; 
        if the LLM call fails, fall back to regular expression parsing.
        """
        
        raw = uploaded_names_file.read().decode('utf-8', errors='ignore')
        try:
            uploaded_names_file.seek(0)
        except Exception:
            pass

        prompt = (
            "Below is the content of the uploaded names and df_head files, please return all attribute names corresponding one-to-one with df_head in Python list format, "
            "and maintain the order, do not add any extra text. Please note, you only need to return a list, do not include any markdown syntax:\n```\n"
            f"name file: {raw}\n```"
            f"df_head: {df_head}\n```"
        )
        try:
            response = self.call(prompt)
            names_list = eval(response.strip())
            if isinstance(names_list, list) and all(isinstance(n, str) for n in names_list):
                col_names = names_list
            else:
                raise ValueError("LLM output format is incorrect.")
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
            "You are a professional data analysis assistant, responsible for explaining data structure and business meanings.\n"
            f"- Data dimensions: {df.shape[0]} rows × {df.shape[1]} columns\n"
            f"- Column names and data types: {dict(zip(df.columns.tolist(), df.dtypes.astype(str).tolist()))}\n"
            f"- First 5 rows sample: \n{df.head().to_dict(orient='list')}\n\n"
            f"""- Data explanation chat dialogue:
            --- Start of chat record ---
            {memory_block}
            --- End of chat record ---"""
        )

        if user_input is not None:
            prompt += f"""
            Please conduct a deep, systematic analysis of the current data strictly based on the user requirement "{user_input}".
            Requirements:
            1. The analysis must directly correspond to this requirement, refrain from adding extraneous inferences.
            2. Conclusions should be specific, clear, and able to directly support subsequent report writing or modeling steps.
            3. The analysis language should be professional, concise, and avoid vague or emotional expressions.
            """
        else:
            prompt += """
            Below is a basic overview of a dataset. Please help me analyze its nature and structure, and answer the following questions:

            1. Which business or research scenario is this dataset likely sourced from?
            2. What is the meaning of each major field? If discernible, please specify their units or the meaning of their values.
            3. Are there any obvious anomalies, unusual distributions, or features requiring attention in the data?

            Output requirements:
            - Use natural, fluent English descriptions;
            - Adopt a clear, numbered list structure (1, 2, 3);
            - Keep the language objective and concise, avoid vague terms like 'might', 'maybe', 'seems';
            - Focus on highlighting the data structure, meanings, and potential issues.
            """

        if st.session_state.preference_select:
            prompt += f"The user's analysis preferences are as follows: {st.session_state.preference_select}.\n\n"
        if st.session_state.additional_preference:
            prompt += f"The user has provided the following modeling goals and special requirements: {st.session_state.additional_preference}.\n\n"


        desc = self.call(prompt)

        return desc
    
    
    def summary_html(self):

        df = self.load_df()
        df_head = df.head()
        dtype_info = df.dtypes.astype(str)

        prompt = f"""
        You are writing the first chapter of a data analysis report -- 'Data Overview and Data Meaning Analysis'.
        Please, based on the following input content, organize key information and provide analytical explanations:
        Data format:
        {dtype_info}

        First five rows of data:
        {df_head}

        Data explanation chat dialogue:
        --- Start of chat record ---
        {self.memory}
        --- End of chat record ---

        Additional requirements:
        1. Use fluent natural language
        2. Do not overuse adjectives and adverbs; try to express meaning with simple verbs and nouns
        3. Do not use vague expressions such as 'might', 'maybe', 'seems', 'subtle'
        """.strip()

        desc = self.call(prompt)

        summary = {
                    "title": "Data Loading",
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
            This is the data import stage of data analysis.
            Data format:
            {dtype_info}

            First five rows of data:
            {df_head}

            Data explanation chat dialogue:
            --- Start of chat record ---
            {self.memory}
            --- End of chat record ---

            Requirements:
            Please, based on the above data and dialogue content, generate a concise and accurate comprehensive summary.
            The summary should completely present the core information, facilitating the subsequent automatic judgment of whether this content needs to be cited in the report writing.
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
                f"[Stage Description]This is the data import stage in the data analysis process.\n"
                f"[Data Format]{dtype_info}\n"
                f"[Sample Preview]\n{df_head}\n"
                f"[Analysis Dialogue Record]\n{self.memory}"
            )

        return self.full
