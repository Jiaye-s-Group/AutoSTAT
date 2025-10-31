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
                raise TypeError(f"Expecting pandas.DataFrame or numpy.ndarray, but received {type(processed_df)}")

        self.processed_df = processed_df


    def load_processed_df(self):

        return self.processed_df
    

    def load_refined_suggestions(self):
        return self.refined_suggestions
    

    def save_refined_suggestions(self, refined_suggestions):
        self.refined_suggestions = refined_suggestions


    def refine_suggestions(self, df_head):
        """refine suggestions returned by LLMs"""

        suggestion = self.load_preprocessing_suggestions()

        prompt = f"""
        Please summarize the recommended preprocessing methods for each column in the dataset based on the following preprocessing suggestions.

        Data sample:
        {df_head}

        Detailed preprocessing suggestions:
        {suggestion}

        Output requirements (must be strictly followed):
        1. Output format: Column name: Recommended preprocessing method; each entry on a new line.
        2. For each column, provide at most three recommended methods; multiple methods should be separated by commas.
        3. The output must be plain text, without any Markdown markup.
        4. The length of each method should not exceed 10 words."""

        refined_suggestions = self.call(prompt)
        self.refined_suggestions = refined_suggestions

        return refined_suggestions
        

    def get_preprocessing_suggestions(
        self, 
        user_input=None,
        memory_limit=6,
    ):

        df = self.load_df()

        # Basic statistics
        n_rows, n_cols = df.shape
        dtype_counts = df.dtypes.value_counts().to_dict()
        missing_total = int(df.isnull().sum().sum())
        missing_by_col = df.isnull().mean().mul(100).round(2).to_dict()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Organize memory fragments
        recent_memory = self.memory[-memory_limit:] if self.memory else []
        if recent_memory:
            formatted_memory = "\n".join(
                f"{m['role']}: {m['content']}" for m in recent_memory
            )
            memory_block = f"{formatted_memory}"
        else:
            memory_block = ""

        prompt = f"""
        You are a senior data preprocessing expert, responsible for providing high-quality preprocessing suggestions for data analysis reports.

        === Data Overview ===
        - Data scale: {n_rows} rows × {n_cols} columns
        - Data type distribution: {dtype_counts}
        - Total missing values: {missing_total}
        - Missing rate per column: {missing_by_col}
        - Numeric columns: {num_cols}
        - Historical context (for reference only): {memory_block}
        """

        if user_input is None:
            prompt += """
            === Please perform a column-by-column analysis (note: this is a column-by-column analysis) ===
            Please address the following four aspects for each column sequentially:

            1. **Data Type**: Clearly state the column's data type. If mixed types or outlier value types exist, please indicate.
            2. **Missing Value Handling Suggestions**: Explain the suggested strategy for handling missing values in this column; if an adjustment is recommended, specify the exact "Missing Value Handling Strategy" operation.
            3. **Outlier Handling Suggestions**: Explain the proposed outlier detection and handling method for this column; if an adjustment is needed, specify the "Outlier Handling Strategy or Threshold" operation.
            4. **Standardization Suggestions**: State whether standardization or scaling is recommended, and if needed, indicate the "Standardization Processing Strategy" operation.

            Output format requirements:
            - Output in segments following the format of "Column name + numbered points (1-4)";
            - Each column should be in an independent segment, separated by a line break;
            - Use clear, concise, professional language.
            """
        else:
            prompt += f"""
            === User's New Requirement ===
            {user_input}

            Please combine the above data overview and historical context to provide the next steps for this requirement.
            Considerable operations include: missing value handling, outlier detection and correction, standardization or normalization, feature type adjustment, etc.
            The output should maintain structure and coherence, avoiding repetitive explanations.
            """

        if st.session_state.preference_select:
            prompt += f"The user's analysis preferences are as follows: {st.session_state.preference_select}.\n\n"
        if st.session_state.additional_preference:
            prompt += f"The user has provided the following modeling goals and special requirements: {st.session_state.additional_preference}.\n\n"


        suggestions = self.call(prompt)

        return suggestions
    

    def code_generation(self, df_head, user_prompt):
        """Generate LLM prompt: Asking LLM to output process_df (pandas DataFrame)."""
        allowed = ", ".join(self.allowed_libs)

        prompt = f"""
        **Output pure Python code only**, strictly avoiding:
        - Explanatory text, comments, examples;
        - Markdown code block markers (do not use ``` or ```python);
        - Any extraneous output (e.g., print statements, global variable assignments).

        === Runtime Environment Specification ===
        The following objects and libraries are available:
        - pandas DataFrame variable: `df`
        - Libraries: numpy (np), SimpleImputer, StandardScaler, MinMaxScaler, RobustScaler,
        OneHotEncoder, OrdinalEncoder, LabelEncoder, FunctionTransformer,
        ColumnTransformer, Pipeline.
        If required functionality is unavailable in these libraries, implement it in Python code.

        === Generation Requirements ===
        1. Prioritize user requirements (higher priority than LLM's generic suggestions).
        2. If a column is marked "no processing needed", skip all operations for it.
        3. Do not import additional libraries; prohibit file I/O operations.
        4. All brackets (parentheses, square brackets, curly braces) must be correctly paired and closed.
        5. For categorical features:
        - Use OneHotEncoder or OrdinalEncoder.
        - For single string/categorical columns, use LabelEncoder or OrdinalEncoder (no passthrough).
        6. Before building ColumnTransformer, detect and handle "mixed-type columns"
        (containing both numbers and strings) using:
        `FunctionTransformer(lambda x: x.astype(str))` to convert them uniformly to string type.
        7. Include only processed columns in ColumnTransformer's transformers.
        8. When using OneHotEncoder, ensure all input features are numeric if outputting sparse matrices.
        9. Automatically detect and remove duplicate header rows (e.g., row 0 matching headers).
        10. Ensure preprocessed DataFrame columns have explicit names.
        11. Retain only one result line at the end:
            `process_df = ...`  
            No print/display statements or extraneous output allowed.

        === Input Data Sample ===
        {df_head}

        === User-Specified Requirement ===
        {user_prompt}

        Strictly adhere to all requirements above and output complete, executable Python code (pure code block, no additional explanations).
        """.strip()

        if self.error is not None:
            if self.debug_num < 5 :
                self.debug_num += 1

                prompt += f"""
                The previously generated code failed to run.
                [Error Message]:
                {self.error}

                [Original Code]:
                {self.code}

                Please infer and understand the root cause of the error without outputting any explanatory text.

                Requirements:
                1. Do not output any analysis, explanations, or clarifications (including text, lists, or comment paragraphs);
                2. Brief internal code comments may be used to explain key modifications;
                3. If the error stems from logic, data structure, or improper function usage, please adjust accordingly;
                4. If dependency library methods are unsuitable, implement alternative functions;
                5. The generated code must be independently executable without syntax errors;
                6. Maintain consistency with the original code's intent, making only necessary corrections.
                """

            else:
                self.debug_num = 0

        if self.user_input is not None:
            prompt += f"user requirements: {self.user_input}.\nPlease strictly adhere to and prioritize this requirement, as it takes precedence over all other suggestions or rules.\n"

        if self.refined_suggestions is not None:
            prompt += f"Preprocessing suggesions returned by LLM: {self.refined_suggestions}"

        raw = self.call(prompt)
        return raw


    def summary_html(self):

        if self.code is None:
            summary = None
            return summary

        else:
            processed_df = self.load_processed_df()
            prompt = f"""
            You are writing the second chapter of the data analysis report – 'Data Preprocessing and Standardization'.
            Please, based on the following input content, extract key information and write corresponding analysis paragraphs.

            - Preprocessing code:
            {self.code}

            - Preprocessing results (data sample):
            {processed_df.head()}

            {f"- Preprocessing suggestion dialogue record: {self.load_memory}" if self.load_memory else ""}

            Writing requirements:
            1. Use fluent, natural English expression;
            2. Language should be concise and accurate, avoid excessive adjectives or adverbs;
            3. Do not use vague expressions such as 'might', 'maybe', 'seems', 'subtle';
            4. Do not add large headings; use natural paragraphs for narration;
            5. Content must be logically clear, reflecting the analytical connection between the code and the results.

            """.strip()

            desc = self.call(prompt)

            summary = {
                        "title": "Preprocessing",
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

                memory = f"[Preprocessing Suggestion Dialogue Record]\n{self.load_memory}\n" if self.load_memory else ""

                prompt = f"""
                This is the 'Data Preprocessing and Standardization' stage in the data analysis process.

                [Preprocessing Code]
                {self.code}

                [Preprocessing Results (First Five Rows)]
                {processed_df.head()}

                {memory}
                Under the premise of ensuring accurate and complete information, summarize the above content into a concise text summary.
                Requirements:
                1. The language should be natural and fluent, maintaining objectivity and professionalism;
                2. The content should cover key points (including main preprocessing steps and result characteristics);
                3. The focus is on 'explaining the core information,' rather than describing line by line;
                4. The generated summary should be usable for determining whether this part needs to be cited in report writing.
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
                [Stage Description] This is the data preprocessing stage in the data analysis process.
                [Preprocessing Code] {self.code}
                [Preprocessing Results - First Five Rows] {processed_df.head()}
                """.strip()
                if self.load_memory is not None:
                    content += f"\n[Preprocessing suggestion chatting history]\n{self.load_memory}"

                self.full = content

        return self.full
