import streamlit as st
import base64
import plotly.graph_objs as go
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompt_engineer.call_llm import LLMClient

import numpy as np
np.set_printoptions(edgeitems=250, threshold=501)

class VisualizationAgent(LLMClient):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.cols_wo_id = None
        self.recommendations = None
        self.analysis = []
        self.quick_action = None
        self.data_meaning = ""
        self.allowed_libs = [
            "numpy", "plotly", "plotly.express", "plotly.graph_objects"
        ]
        self.code = None
        self.result = None
        self.suggestion = None
        self.user_input = None
        self.fig = []
        self.par_content = ""
        self.error = None
        self.abstract=None
        self.full = None
        self.color = None
        self.finish_auto_task = False
        self.debug_num = 0
        self.refined_suggestions = None


    def finish_auto(self):

        self.finish_auto_task = True


    def save_user_input(self, user_input):

        self.user_input = user_input


    def load_user_input(self):

        return self.user_input
    

    def save_color(self, color):

        self.color = color


    def load_color(self):

        return self.color
    
    
    def add_fig(self, fig, desc):

        entry = {"fig": fig, "desc": desc}
        self.fig.append(entry)


    def load_fig(self):

        return self.fig
    
    
    def save_cols_wo_id(self, col):

        self.cols_wo_id = col


    def load_cols_wo_id(self):

        return self.cols_wo_id


    def save_code(self, code):

        self.code = code


    def load_code(self):

        return self.code


    def save_recommendations(self, recommendations):

        self.recommendations = recommendations


    def load_recommendations(self):

        return self.recommendations


    def save_suggestion(self, suggestion):

        self.suggestion = suggestion


    def load_suggestion(self):

        return self.suggestion


    def load_data_meaning(self):

        return self.data_meaning


    def save_error(self, error):

        self.error = error


    def load_error(self):

        return self.error


    def refine_suggestions(self, rec):

        prompt = f"""
        Please extract the recommended visualization methods for each column and each variable group based on the following detailed visualization suggestions.

        Detailed visualization suggestions:
        {rec}

        Output requirements (must be strictly followed):
        1. Output as plain text, with each entry on a new line, and no extra explanations.
        2. Single variable format: Column name: Chart1, Chart2.
        3. Multi-variable format: Relationship group: ColumnA,ColumnB: Chart1, Chart2.
        4. Overall variable format: Overall: Chart1, Chart2.
        5. Strictly do not add titles, numbering, examples, or extra explanations.
        6. Extract visualization methods accurately.
        """

        refined_suggestions = self.call(prompt)
        self.refined_suggestions = refined_suggestions

        return refined_suggestions


    def get_visualization_recommendations(
        self,
        cols,
        user_input=None,
        memory_limit: int = 6,
    ) -> str:

        dim_info = f"{self.df.shape[0]} 行 x {self.df.shape[1]} 列"

        recent_memory = self.memory[-memory_limit:] if getattr(self, "memory", None) else []
        if recent_memory:
            formatted_memory = "\n".join(
                f"{m['role']}: {m['content']}" for m in recent_memory
            )
            memory_block = f"{formatted_memory}"
        else:
            memory_block = ""

        if user_input is None:
            prompt = f"""
            You are a senior data visualization expert. Please provide systematic and professional suggestions for the "Visualization Design" chapter of a data analysis report based on the following information.

            [Dataset Information]
            - Numerical variables: {cols}
            - Data dimensions: {dim_info}
            - Historical context (for reference only): {memory_block}

            [Output Format]
            Please strictly follow the structure below (maintain the headings and hierarchy, do not add or remove):

            I. Univariate Visualization
            1. For each numerical variable, recommend 1-2 most suitable visualization methods and briefly explain the reasoning.
            Example:
            - `Column1`: Recommend "Histogram" and "Box Plot", reason: ...

            II. Multivariate Relationship Visualization
            1. Select 1-3 variable combinations (each containing 2-3 variables) from the above variables that are worth focusing on, and explain the reason for selection.
            Example:
            - Relationship group 1: `[Column1, Column2]`, reason: ...
            2. For each variable group, recommend the most suitable visualization method and briefly explain.
            Example:
            - Relationship group 1: Scatter Plot + Regression Line, reason: ...

            III. Overall Distribution Visualization
            1. Recommend 1-2 global visualization methods for the overall distribution characteristics of the entire dataset, and explain their purpose.
            Example:
            - Recommend "Violin Plot Matrix", purpose: ...
            - Recommend "Heatmap", purpose: ...

            [Execution Requirements]
            1. Automatically filter out column names without practical meaning (e.g., index, redundant ID);
            2. The output content should be well-organized, concise, and professional.
            """.strip()

        else:
            prompt = f"""
            You are a senior data visualization expert. Please respond to the user's requirements and fulfill them based on the following information:

            [User Requirements]
            {user_input}

            [Dataset Information]
            - Numerical variables: {cols}
            - Data dimensions: {dim_info}
            - Data overview (first few rows):
            {self.df.head().to_string(index=False)}
            - Historical context (for reference only): {memory_block}

            [Execution Requirements]
            1. If the user explicitly specifies visualization columns, only provide suggestions for these columns;
            2. If the user proposes specific requirements (such as graph size, axis log scaling, etc.), they must be reflected in the output;
            3. Only respond to the user's requirements, do not output irrelevant content;
            4. If the user requests partial modifications to previous content, retain the unchanged parts and only update the relevant suggestions;
            5. The output content should be well-structured, logically coherent, and concise in language.
            6. Do not output code.
            """.strip()

        if st.session_state.preference_select:
            prompt += f"The user's analysis preferences are as follows: {st.session_state.preference_select}.\n\n"
        if st.session_state.additional_preference:
            prompt += f"The user has provided the following modeling goals and special requirements: {st.session_state.additional_preference}.\n\n"

        recommendations = self.call(prompt)
        return recommendations


    def desc_fig(self, fig, dtype_info):

        selected = st.session_state.selected_model

        if selected == "Zhipu" or selected == "Qwen" or selected == "GPT-4o" or selected == "GPT-5":
            img_bytes = fig.to_image(format="jpg")
            fig_info = extract_plotly_info(fig)
            base64_bytes = base64.b64encode(img_bytes)
            base64_string = base64_bytes.decode('utf-8')

            prompt_payload = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpg;base64,{base64_string}"}
                },
                {
                    "type": "text",
                    "text": f"""
                    Please synthesize the visualization chart and variable information below to conduct a **concise yet in-depth analysis**.
                    From the five perspectives of distribution shape, trend characteristics, relationships between variables, potential anomalies, and real-world implications, extract key insights.
                    Output a natural language analysis conclusion not exceeding 120 words (not a summary).

                    [Variable Information]
                    {dtype_info}

                    [Chart Structure Information]
                    {fig_info}

                    Writing requirements:
                    1. The analysis must include the identification and explanation of data anomalies:
                    - If there are obvious anomaly points, anomaly segments, or abrupt change trends, please point out their characteristics and potential impact;
                    - If no anomalies are found, clearly state that the overall distribution is stable or there are no significant anomalies;
                    2. The content should reflect reasoning and explanatory thinking, rather than superficial description;
                    3. Use logical, clear, objective, and professional language;
                    4. Use verb-driven sentence patterns (such as "shows", "reflects", "reveals", "illustrates", etc.);
                    5. Do not use vague words (such as "might", "seems", "subtle", etc.);
                    6. Do not use titles, lists, or formatting symbols;
                    7. If there is noise or repetitive information in the variable meanings, please ignore it automatically;
                    8. Maintain a concise and powerful tone, emphasizing data characteristics and analysis conclusions.
                    """.strip()
                }
            ]

            desc_fig = self.call(prompt_payload)

        else:
            prompt = f"""  
            Please synthesize the visualization chart and variable information below, and analyze from perspectives such as data distribution, trend characteristics, and potential relationships.
            Summarize key findings in natural language not exceeding 100 words, highlighting the variable's significance or anomalies in the overall data structure.

            [Variable Information]
            {dtype_info}

            [Chart Information]
            {fig.to_dict()}

            Writing requirements:
            1. Language should be fluent and natural, maintaining objectivity and professionalism;
            2. Use concise verbs and nouns, avoid overusing adjectives or adverbs;
            3. Avoid vague words such as 'might', 'maybe', 'seems', 'subtle', etc.;
            4. Do not add titles or list structures;
            5. Combine the meaning of the data and the characteristics of the chart to provide insightful brief conclusions;
            6. If there is messy or repetitive information in the variable meanings, please ignore it automatically.
            """.strip()

            desc_fig = self.call(prompt)
            
        return desc_fig


    def summary_html(self) -> str:
        
        analysis = self.summary_fig_analysis_list()

        if analysis is None:
            
            return None

        else:
            analysis = {i: item for i, item in enumerate(analysis)}

            summary = {
                        "title": "Visualization",
                        "fig_analysis": analysis,
                    }

            return summary


    def summary_word(self) -> str:
        
        analysis = self.summary_fig_analysis_list()

        if analysis is None:
            
            return None

        else:

            summary = {
                        "title": "Visualization",
                        "fig_analysis": analysis,
                    }

            return summary


    def summary_fig_analysis_list(self) -> str:

        if not self.code:
            return self.analysis
        
        if self.analysis:
            return self.analysis

        # state_copy = dict(st.session_state)
        selected = st.session_state.get("selected_model", "default")
        # selected = state_copy.get("selected_model", "default")

        # --- Define single mission ---
        def analyze_one(item, offset):
            fig = item["fig"]
            desc = item["desc"]

            # Recover state（If need to access st.session_state）
            # st.session_state.update(state_copy)
            selected = st.session_state.get("selected_model", "default")
            if isinstance(fig, go.Figure):
                if selected == "Zhipu" or selected == "Qwen" or selected == "GPT-5" or selected == "GPT-4o":
                    img_bytes = fig.to_image(format="jpg")
                    base64_string = base64.b64encode(img_bytes).decode("utf-8")
                
                    fig_info = extract_plotly_info(fig)

                    prompt_payload = [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpg;base64,{base64_string}"}
                        },
                        {
                            "type": "text",
                            "text": f"""
                            You are writing Chapter 3 of the data analysis report – 'Data Visualization'.  
                            Please, for the variables below, combine their **business meaning, statistical features** and **visualization performance** to write a professional, logically rigorous analysis content that can be directly used in the report body.

                            [Variable Information]
                            {self.cols_wo_id}

                            [Plotly Chart Structure]
                            {fig_info}

                            [Basic Statistical Overview]
                            {desc}

                            [Analysis Task]
                            Please first complete the following reasoning steps mentally, then output the structured text:
                            1. Identify core patterns from the chart: overall trends, peaks, distribution shapes, anomaly points, or clusters;
                            2. Consider the relationship between these patterns and the variable's business meaning;
                            3. Judge whether anomalies exist (single-point anomalies, phase anomalies, or structural changes) and explain their potential impact;
                            4. If the chart includes other variables, analyze their statistical or logical relationships;
                            5. Integrate the above insights into a logically complete and naturally worded paragraph.

                            [Output Format (Strictly Adhere)]
                            Output as plain text, sequentially containing the following three parts (do not use Markdown or symbols):

                            1. Overview  
                            - Briefly describe the variable's definition, business role, and overall trend in data performance;  
                            - Suggest the variable's potential importance in the overall data structure.

                            2. Distribution and Feature Analysis  
                            - Analyze its distribution characteristics from statistical and graphical perspectives (central tendency, dispersion, skewness, kurtosis, periodicity, etc.);  
                            - If anomalies or changes are found, specify their manifestations and potential mechanisms;  
                            - If there are association trends with other variables, indicate the direction and strength.

                            3. Practical Implications and Inferences  
                            - Explain the observed phenomena in the context of business or research background;  
                            - Analyze the real-world rules, risks, or optimization directions they may reveal;  
                            - If appropriate, propose reasonable speculations or follow-up analysis suggestions (maintain objectivity and logical self-consistency).

                            [Writing Requirements]  
                            1. Maintain formal, professional, and tightly logical language;  
                            2. Use diverse sentence structures and natural expression, avoid templated phrasing;  
                            3. Forbid vague vocabulary (such as "might", "seems", "probably", etc.);  
                            4. Do not use any heading symbols (e.g., #, **, etc.);  
                            5. Do not output terms like "AI", "model", "assistant", etc.;  
                            6. Output as continuous text, without explanatory sentences or additional notes.  
                            """.strip()
                                }
                            ]

                    analysis_text = self.call(prompt_payload)

                else:

                    prompt = f"""
                            You are writing Chapter 3 of the data analysis report – 'Data Visualization'.
                            Please, for the variables below, combine their business meanings and corresponding visualizations to write a structured, professional analysis text.

                            [Variable Information]
                            {self.cols_wo_id}

                            [Plotly Chart Information]
                            {fig.to_dict()}

                            [Basic Statistical Overview]
                            {desc}

                            Please strictly follow the format below to write the content (use plain text, do not use Markdown syntax or symbols):

                            1. Overview
                            - Explain the meaning of the variable and its role in the data or business;  
                            - Briefly describe the overall distribution characteristics or main association trends between variables.

                            2. Distribution / Association Features
                            - Explain the distribution characteristics or correlation relationships of the variable from a statistical perspective;  
                            - Key statistics (mean, median, quartiles, correlation coefficient, etc.) can be cited to support the analysis.

                            3. Practical Implications
                            - Combine the variable's meaning in practical contexts to explain the observed distributions or relationships;  
                            - Point out the real-world phenomena or potential impacts these patterns may reflect (e.g., a high value in a variable indicates increased risk or group characteristic differences).

                            [Writing Requirements]  
                            1. Use fluent, natural, and formal English expression;  
                            2. The language should be objective and concise, avoiding redundant rhetoric;  
                            3. Do not use vague words such as "might", "maybe", "seems", "subtle", etc.;  
                            4. Do not use heading symbols (#, **, etc.);  
                            5. Maintain logical coherence and clear analytical layers.
                            """.strip()

                    analysis_text = self.call(prompt)
                    print(prompt)
                return offset, {"figure": fig, "analysis": analysis_text}

        # --- Parallel Execution ---
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(analyze_one, item, i) for i, item in enumerate(self.fig)]
            for f in as_completed(futures):
                result = f.result()
                if result:
                    results.append(result)

        # --- Sort by previous order ---
        results.sort(key=lambda x: x[0])
        self.analysis = [r[1] for r in results]

        return self.analysis


    def code_generation(self, df_head: str, user_prompt: str) -> str:
        """Generate LLM prompt: Require the LLM to output result_dict (which can be JSON serializable)."""
        allowed = ", ".join(self.allowed_libs)

        prompt = (
            "Output pure Python code only, do not output any explanatory text, comments, examples, or markdown code fences (no orpython allowed). "
            "The environment provides pandas DataFrame variable df, numpy (np), "
            "plotly.express (px), and plotly.graph_objects (go).\n\n"
            "## Strict Requirements ##:\n"
            "1) Strictly adhere to user requirements: If the user specifies columns to visualize (either exact names or approximate inputs like 'ordera' when the actual column is 'ordertypea'), DO NOT fabricate column names!!! "
            f"At the start of the script, use LLM understanding to map user input to the most appropriate actual column name from {df_head}, or adopt a more conservative approach using indices (e.g., column 0, column 1 - recommended!). Only plot charts for these columns;\n"
            """2) Count and Rename: For all categorical distribution plots, strictly follow this template—NEVER use indexdirectly as a column name—
            # === Template: Count and Plot Bar Chart ===
            for col in categorical_cols:
                df_counts = df[col] \\
                    .value_counts() \\
                    .rename_axis(col) \\
                    .reset_index(name='count')
                fig = px.bar(
                    df_counts,
                    x=col,
                    y='count',
                    title=f'Bar Chart of {col}',
                    labels={col: col, 'count': 'Count'}
                )
                fig_dict[f'{col}_bar'] = fig

            3) Intelligent Chart Selection: Automatically choose appropriate charts based on data type (numerical/categorical).
            4) Auto-detect Coloring: If a categorical column exists and requires continuous mapping, encode it into numerical codes first. For discrete mapping, use parallel_categories.
            5) If no suitable Plotly Express chart exists, use `go.Figure` for customization.
            6) The script must end with only `fig_dict = {...}`; no `print` statements or extra global variables.
            7) Under NO circumstances fabricate column names or directly write `'index'`; if using the index, explicitly reference `df.index`.
            8) Do not use file I/O or external operations.
            9) Provide Python code ONLY—no non-code identifiers like '''python.\n"""
            f"Sample data head:\n{df_head}\n\n"
            f"Colors for each chart MUST be selected from {self.color}\n\n"
            f"Plotting suggestions: {self.refined_suggestions}\n\n"
            "Return: Complete Python code block (pure code only)."
        )

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

        if st.session_state.preference_select:
            prompt += f"The user's analysis preferences are as follows: {st.session_state.preference_select}.\n\n"
        if st.session_state.additional_preference:
            prompt += f"The user has provided the following modeling goals and special requirements: {st.session_state.additional_preference}.\n\n"

        raw = self.call(prompt)

        return raw


    def check_abstract(self):
        if self.abstract is None:
            # Get all analysis content
            analysis_list = self.summary_fig_analysis_list()

            if not analysis_list :
                self.abstract = "No visualization analysis content available."
                return self.abstract

            # Merge all analytical content into a unified text.
            all_analyses = "\n\n".join([
                f"[variable analysis {i+1}]\n{item['analysis']}"
                for i, item in enumerate(analysis_list)
            ])

            prompt = f"""
            Please read and synthesize the analysis content of the following multiple variables:
            {all_analyses}

            Task:
            Integrate these analyses into a structured, information-rich **comprehensive semantic summary** for subsequent automatic generation of report outlines by large language models.

            Objectives:
            - The output should help subsequent models understand the themes, variables, dimensions, relationships, and logical sequence contained in the analysis;
            - It will serve as input for the "outline generation model," so it must enable the model to identify the chapters and subchapters that should be included in the report.

            Writing Requirements:
            1. **Information Retention**:
            - Preserve key conclusions, trends, characteristics, and significant differences for each variable;
            - Clearly indicate connections, comparisons, or influences between variables;
            - Do not omit any facts valuable to the analysis theme.

            2. **Structure-Oriented**:
            - Organize logically: overall characteristics → analysis of individual variables → inter-variable relationships → underlying patterns;
            - If different themes exist (e.g., meteorological factors, pollutant indicators, model results), naturally reflect hierarchy;
            - Imply section boundaries semantically (e.g., "first... then... finally...", "regarding meteorological variables...", "in the modeling section...").

            3. **Language Style**:
            - Professional, clear, objective;
            - Use complete sentences, avoid lists or numbering;
            - Can be slightly detailed; brevity is not the primary goal.

            4. **Output Format**:
            - Output only a single continuous paragraph;
            - Do not add titles, comments, JSON, or code blocks;
            - This text will be directly fed to the outline generation model, not displayed to humans.

            Please generate a comprehensive semantic summary that meets the above requirements.
            """.strip()
            self.abstract = self.call(prompt)

        return self.abstract


    def check_full(self):
        """
        Return structured content, adhering to the image insertion protocol:
        - Prefix each analysis content with an index
        - Use [FIG:index] to indicate image insertion positions
        - During subsequent processing, actual images can be replaced according to this protocol
        """
        if self.full is None:
            analysis_list = self.summary_fig_analysis_list()

            if not analysis_list :
                self.full = "No visualization analysis content available."
                return self.full

            # Construct structured text: with image insertion markers
            full_parts = ["""[Stage Description] This is the data visualization stage in the data analysis process."""]
            for i, item in enumerate(analysis_list):
                desc = item["analysis"]
                part = f"""
                [Analysis of Figure {i}]
                {desc}
                [FIG:{i}]  # Image insertion position marker
                """.strip()
                full_parts.append(part)

            self.full = "\n\n".join(full_parts)

            # Add protocol note
            protocol_note = """
            ---
            # Image Insertion Processing Protocol Description:
            #  [FIG:index] indicates the image insertion position
            #  index corresponds to the index in the analysis content
            #  You can use [FIG:index] where you need to place an image
            """.strip()

            self.full = f"{self.full}\n\n{protocol_note}"

        return self.full


def extract_plotly_info(fig):
    """
    Extract key information from a Plotly Figure (object / dict / string):
    - Chart title
    - X/Y axis titles
    - Chart type
    - Color information
    - Number of traces
    """
    import ast
    import plotly.graph_objects as go

    if isinstance(fig, go.Figure):
        fig = fig.to_dict()
    elif isinstance(fig, dict):
        pass
    elif isinstance(fig, str):
        clean_str = fig.strip()
        if clean_str.startswith("Figure("):
            clean_str = clean_str[len("Figure("):-1]
        try:
            fig = ast.literal_eval(clean_str)
        except Exception as e:
            raise ValueError(f"Unable to parse string-formatted Figure: {e}")
    else:
        raise TypeError(f"Not supported fig type: {type(fig)}")

    layout = fig.get("layout", {})
    title = layout.get("title", {}).get("text", "")
    xaxis_title = layout.get("xaxis", {}).get("title", {}).get("text", "")
    yaxis_title = layout.get("yaxis", {}).get("title", {}).get("text", "")

    data_list = fig.get("data", [])
    types = list({d.get("type", "") for d in data_list})


    return {
        "title": title or "(No title)",
        "xaxis": xaxis_title or "(No x-axis title)",
        "yaxis": yaxis_title or "(No y-axix title)",
        "types": types,

    }
