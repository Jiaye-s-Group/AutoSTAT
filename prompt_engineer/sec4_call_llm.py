import streamlit as st

from prompt_engineer.call_llm import LLMClient


class ModelingCodingAgent(LLMClient):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.allowed_libs = [
            "numpy", "sklearn.model_selection", "sklearn.preprocessing", "sklearn.ensemble", 'torch', 'torchvision', 'torchaudio', 'xgboost', 'lightgbm'
        ]
        self.code = None
        self.result = None
        self.suggestion = None
        self.user_selection = None
        self.par_content = ""
        self.inference_code = None
        self.best_model = None
        self.inference_data = None
        self.inference_processed_df = None
        self.abstract=None
        self.full = None
        self.error = None
        self.inference_error = None
        self.target = None
        self.finish_auto_task = False
        self.best_model_gz_bytes = None
        self.debug_num = 0
        self.refined_suggestions = None

    def finish_auto(self):

        self.finish_auto_task = True


    def save_best_model_gz_bytes(self, best_model_gz_bytes):

        self.best_model_gz_bytes = best_model_gz_bytes


    def load_best_model_gz_bytes(self):

        return self.best_model_gz_bytes


    def save_target(self, target):

        self.target = target


    def load_target(self):

        return self.target


    def save_error(self, error):

        self.error = error


    def load_error(self):

        return self.error

    
    def save_inference_error(self, inference_error):

        self.inference_error = inference_error


    def load_inference_error(self):

        return self.inference_error


    def save_inference_data(self, inference_data):
        
        self.inference_data = inference_data
        
        
    def load_inference_data(self):
        
        return self.inference_data


    def save_inference_processed_df(self, inference_processed_df):
        
        self.inference_processed_df = inference_processed_df
        
        
    def load_inference_processed_df(self):
        
        return self.inference_processed_df


    def save_inference_code(self, code):
        
        self.inference_code = code
        
        
    def load_inference_code(self):
        
        return self.inference_code


    def save_best_model(self, best_model):
        
        self.best_model = best_model
        
        
    def load_best_model(self):
        
        return self.best_model


    def save_code(self, code):

        self.code = code


    def load_code(self):

        return self.code


    def save_suggestion(self, suggestion):

        self.suggestion = suggestion


    def load_suggestion(self):

        return self.suggestion


    def save_modeling_result(self, result):

        self.result = result


    def load_modeling_result(self):

        return self.result
    
    
    def save_user_selection(self, user_selection):

        self.user_selection = user_selection


    def load_user_selection(self):

        return self.user_selection


    def refine_suggestions(self):
        """Extract information from the preprocessing recommendations returned by the LLM"""

        prompt = f"""
        Please read the following modeling suggestions and transform them into clear modeling task instructions for the next coding agent.

        === Modeling Suggestions ===
        {self.suggestion}

        === Output Requirements (Strictly Adhere) ===
        1. Output must be plain text, without using any Markdown, numbering, or symbols;
        2. Instructions should be concise and clear, enabling the coding agent to directly understand and execute them;
        3. Content should focus on the specific tasks of model building, training, or evaluation;
        4. Avoid explanatory or analytical language, only describe the "operations that need to be performed";
        5. The output should cover all key steps, allowing the coding agent to independently complete the modeling workflow.
        """.strip()

        refined_suggestions = self.call(prompt)
        self.refined_suggestions = refined_suggestions

        print(refined_suggestions)

        return refined_suggestions


    def code_generation(self, df_head: str, user_prompt: str) -> str:
        """Generate an LLM prompt: require the LLM to output a result_dict (JSON serializable)."""
        allowed = ", ".join(self.allowed_libs)

        if self.refined_suggestions is None:
            suggestion = user_prompt
        else:
            suggestion = self.refined_suggestions

        prompt = (
            f"""**Output pure Python code only**, **do not** output any explanatory text, comments, examples, or markdown code fences (no ``` or ```python allowed). The environment provides pandas DataFrame variable `df`, numpy (np), train_test_split, StandardScaler, and any model classes mentioned in the Requirement (e.g., RandomForestRegressor, GradientBoostingRegressor, LinearRegression, XGBRegressor, LogisticRegression, SVC, etc.).

            Requirements:

            1) Use an 80/20 split (random_state=42). Standardize numerical features (StandardScaler) based on user requirements. If standardization is applied, use it ONLY on numerical columns and perform fit_transform on the training set and transform on the test set.
            2) **Train and evaluate ALL models listed in the Requirement sequentially**. Do not select only random forest; if multiple model names are specified in the Requirement, the script MUST iterate through these models, training, predicting, and calculating metrics for each.
            3) Do not import any evaluation libraries (e.g., sklearn.metrics). Implement common metrics manually using numpy (regression: MAE, MSE, R2; classification: accuracy, precision, recall, f1).
            4) **The script MUST output and assign ONLY one variable `result_dict` at the end, which must be a JSON-serializable Python dict.**
            Recommended schema (must include the following keys):
            {{
                "dataset": "<optional descriptive string>",
                "models": [
                {{
                    "name": "<model class name>",
                    "type": "<regression or classification>",
                    "metrics": {{ "<metric name>": <float>, ... }}
                }},
                ...
                ],
                "best_model": {{
                "name": "<best-performing model class name>",
                "score": <float>
                }},
                "artifacts": {{
                "best_model_b64": "<base64 string>",
                "best_model_format": "pickle+gzip"
                }},
                // Optional "artifact_warning": <int byte size> if model is too large
                // Plus any additional fields requested in the Requirement
            }}
            5) Ensure all values are native Python types (float, int). Field names must strictly be models, best_model, artifacts. If the user has additional requirements (e.g., recording training time, feature importance), include them in result_dict.
            6) Model export: After training, serialize the selected best_model using pickle, compress with gzip, then base64 encode. Place the encoded string and format info into result_dict["artifacts"], ensuring the final result_dict is JSON-serializable.
            7) The script must end with ONLY the line `result_dict = {{...}}`. No print statements, no other global variables, no file I/O.
            8) If the serialized model size exceeds a reasonable limit, add `"artifact_warning": <byte size>` to result_dict.
            9) Do not use any external I/O or file operations.
            10) Accurately implement models specified in the Requirement. Do not add models outside the Requirement. If a model cannot be directly called from provided libraries, implement it manually!

            Sample data head:
            {df_head}

            Requirement (Based on the following modeling task instructions, train and evaluate ALL listed models sequentially. If a model is unavailable in the current environment, manually implement its algorithm or class to ensure reproducible results):
            {suggestion}

            Allowed libraries: {allowed}.

            Return: Complete Python code block (pure code only)."""
        )

        if self.error is not None:
            if self.debug_num < 5 :
                self.debug_num += 1
                prompt += f"""
                The previously generated code failed to run.
                【Error Message】:
                {self.error}

                【Original Code】:
                {self.code}

                Please reason about and understand the root cause of the error without outputting any explanatory text.

                Requirements:
                1. Do not output any analysis, explanations, or clarifications (including text, lists, or comment paragraphs);
                2. Brief in-code comments may be used to explain critical modifications;
                3. If the error stems from logic, data structure, or improper function usage, make adjustments accordingly;
                4. If dependency library methods are unsuitable, implement alternative functions independently;
                5. The generated code must be independently runnable and syntax error-free;
                6. Maintain consistency with the original code's intent, making only necessary fixes.
                """
            else:
                self.debug_num = 0

        if st.session_state.preference_select:
            prompt += f"The user's analysis preferences are as follows: {st.session_state.preference_select}.\n\n"
        if st.session_state.additional_preference:
            prompt += f"The user has provided the following modeling goals and special requirements: {st.session_state.additional_preference}.\n\n"

        raw = self.call(prompt)

        return raw


    def result_format_prompt(self, result_json: str) -> str:
        """Generate an LLM prompt: require the LLM to output a result_dict (JSON serializable)."""

        prompt = f"""
        Below is a JSON object (containing the model evaluation result structure). Please convert it into a human-friendly Markdown report. Output requirements are as follows:

        === Output Requirements ===
        1. The report must begin with a brief "Dataset Description".
        2. For each model, display the following:
        - Model name;
        - Model type (classification / regression);
        - Main performance metrics (e.g., accuracy, R², MAE, MSE, etc.), each metric rounded to 4 decimal places;
        - It is recommended to use tables or bullet lists for clear presentation.
        3. Clearly mark the **best_model** (highlight its name and optimal metrics in bold).
        4. If the JSON includes feature engineering related information, describe the specific methods and their roles in the "Feature Engineering Description" section.
        5. Output format:
        - Return only Markdown text;
        - Do not use any code block markers (e.g., ```, ```markdown, etc.);
        - Do not output explanatory text, only the final report content (for direct rendering in Streamlit).

        === Input JSON ===
        {result_json}
        """.strip()

        if st.session_state.preference_select:
            prompt += f"The user's analysis preferences are as follows: {st.session_state.preference_select}.\n\n"
        if st.session_state.additional_preference:
            prompt += f"The user has provided the following modeling goals and special requirements: {st.session_state.additional_preference}.\n\n"

        raw = self.call(prompt)

        return raw


    def get_model_suggestion(
        self,
        user_input=None,
        memory_limit: int = 6,  # Control the number of memory rounds introduced
    ) -> str:
        """
        Based on the dataset and historical context, generate intelligent suggestions for the modeling phase. 
        Automatically integrate memory (recent rounds of dialogue) as auxiliary context.
        """

        # === Load basic data ===
        df = self.load_df()
        df_head = df.head().to_string(index=False)
        columns = df.columns.tolist()
        data_info = f"Column names: {columns}\n\nFirst 5 rows:\n{df_head}"

        # === Organize memory fragments ===
        recent_memory = self.memory[-memory_limit:] if getattr(self, "memory", None) else []
        if recent_memory:
            formatted_memory = "\n".join(
                f"{m['role']}: {m['content']}" for m in recent_memory
            )
            memory_block = f"\n=== Historical Context (for reference only) ===\n{formatted_memory}\n"
        else:
            memory_block = ""
        # === Main prompt assembly ===
        prompt = f"""
        You are a senior machine learning modeling expert. Please analyze and reason based on the following information, and output targeted modeling suggestions or improvement plans.

        === Data Information ===
        {data_info}

        === Historical Context (For Reference Only) ===
        {memory_block}
        """.strip()

        # If user has explicit modeling target
        if getattr(self, "target", None):
            prompt += f"""
            
            === Modeling Target ===
            {self.target}
            (Must satisfy this target and explicitly restate the modeling intent in the response.)
            """

        # If user provided additional requirements
        if user_input:
            prompt += f"""
            
            === Current User Requirements ===
            {user_input}
            (Strictly satisfy these requirements. If it's a partial modification, preserve the original logic and only update the specified parts.)
            """

        # If there is previously generated training code
        train_code = self.load_code()
        if train_code:
            prompt += f"""

            === Historical Training Code ===
            {train_code}

            Based on a thorough understanding of the above code, propose **1-2 high-quality model improvement suggestions**.
            Consider the following aspects, but not limited to:
            - Model structure optimization (e.g., adding layers, adjusting activation functions, replacing model types);
            - Feature engineering improvements (e.g., variable selection, feature engineering, normalization strategies);
            - Training process optimization (e.g., regularization, learning rate scheduling, loss function adjustment);
            - Hyperparameter tuning (e.g., tree depth, learning rate, batch size).
            When giving suggestions, briefly explain the "why" and "expected improvement effect".
            """
        else:
            prompt += """
            
            === Modeling Suggestion Task ===
            Based on data characteristics and context, recommend 2-3 suitable model solutions.
            Requirements:
            1. Each model should include the model name, main principles, and applicable scenarios;
            2. Point out its advantages and potential limitations in the current task;
            3. Maintain professional and concise language, do not output code.
            """

        if st.session_state.preference_select:
            prompt += f"The user's analysis preferences are as follows: {st.session_state.preference_select}.\n\n"
        if st.session_state.additional_preference:
            prompt += f"The user has provided the following modeling goals and special requirements: {st.session_state.additional_preference}.\n\n"

        raw = self.call(prompt)
        return raw

    
    
    def summary_html(self) -> str:

        if self.code is None:
            
            summary = None

            return summary

        else:

            prompt = f"""
            You are writing **Chapter 4: Data Modeling** for a data analysis report.
            Please comprehensively analyze the following input content and generate the complete chapter text.
            The content should be logically rigorous, naturally expressed, and demonstrate professional analytical and summary capabilities.

            === Output Structure ===
            Strictly organize the content according to the following five sections:

            1. Overview
            - Explain the objectives of this modeling effort, research background, and data source context.

            2. Methodology Description
            - Introduce the core concepts and implementation process of the adopted models or algorithms;
            - If involving feature engineering, hyperparameter selection, or data preprocessing, explain them accordingly;
            - May appropriately include mathematical principles or optimization mechanisms of the models to demonstrate technical depth.

            3. Key Code Interpretation
            - Focus on core functions and modules, explaining their roles in the modeling workflow;
            - May mention model structure definition, training loops, loss functions, and evaluation logic;
            - Language should be clear and concise, avoiding line-by-line explanations.

            4. Results and Evaluation
            - Summarize main performance metrics (e.g., Accuracy, AUC, MSE, etc.) and result performance;
            - Analyze whether model effectiveness meets expectations, and identify main advantages/disadvantages and bottlenecks.

            5. Improvement Suggestions
            - Propose specific, feasible optimization directions based on model performance and experimental findings;
            - Suggestions can be provided from perspectives such as model structure, feature selection, training strategies, or regularization.

            === Writing Requirements ===
            1. Use natural, fluent, and formal written expression;
            2. Avoid vague or subjective vocabulary (e.g., "maybe," "seems," "subtle," etc.);
            3. Emphasize logical coherence and professionalism;
            4. Do not output titles, list markers, or additional explanations—only generate the main text content.
            """.strip()

            if self.code is not None:
                prompt += f"=== Data Modeling Code ===\n\n{self.code}"
            if self.target is not None:
                prompt += f"=== User Modeling Target ===\n\n{self.target}"
            if self.load_memory is not None:
                prompt += f"=== Data Modeling Chat Dialogue ===\n\n{self.load_memory}"
            if self.result is not None:
                prompt += f"=== Modeling Run Results ===\n\n{self.result}"
            
            desc = self.call(prompt)

            summary = {
                        "title": "Modeling Analysis",
                        "code": self.code,
                        "desc": desc,
                        "result": self.result,
                    }

            return summary


    def summary_word(self) -> str:

        return self.summary_html()


    def code_generation_for_inference(self, code, inference_df_head) -> str:
        """Generate an LLM prompt: require the LLM to output inference analysis code."""

        prompt = (
        f"""Generate a complete Python inference analysis script (return only code, no explanatory text). The runtime environment provides pandas DataFrame variable `inference_df`, pre-trained model `model_obj`, numpy (np), StandardScaler library, and helper function `align_features`. Implement any other required libraries manually. Requirements:

        Sample data information:
        {code}, first five rows of inference_df: {inference_df_head} (DO NOT introduce variables not present in inference_df)

        1) **Available Variables Explanation:**
        - `inference_df`: Inference dataset (Pandas DataFrame)
        - `model_obj`: Pre-trained model object (loaded from best_model.joblib)
        - `np`: NumPy library
        - `pd`: Pandas library
        - `StandardScaler`: sklearn tool for data standardization

        2) **Mandatory Script Functions:**
        a) Apply identical preprocessing to inference data as during training (e.g., missing value handling, encoding conversion, standardization)
        b) **Critical Step: Before prediction, MUST process feature data using align_features function to ensure feature count and order match training**
        c) Use model_obj to predict on preprocessed and aligned feature data
        d) Generate detailed inference report including preprocessing steps and prediction result analysis

        3) **Prediction Result Processing Requirements:**
        - Convert model output to human-readable form (e.g., probability values, class labels, numerical results)
        - **MUST generate DataFrame with predictions**: Merge original/processed `inference_df` with predictions, named `inference_df_with_predictions`
        - Merged DataFrame must contain original feature columns and a prediction column named `'prediction'` (expand to `prediction_0`, `prediction_1`, ... for multi-dimensional outputs)

        4) **Serialization Requirements (for frontend download):**
        - Convert `inference_df_with_predictions` to index-free CSV format
        - Compress CSV data with gzip, then encode as base64 string
        - Create `result_dict['artifacts']` dictionary with these keys:
        * `'predictions_df_b64'`: base64-encoded compressed data
        * `'predictions_df_format'`: fixed value 'csv+gzip'
        * `'predictions_df_size_bytes'`: compressed byte size (integer)
        - Add `'predictions_df_records'` key to `result_dict` with value `inference_df_with_predictions.to_dict(orient='records')`
        - Ensure all numpy/pandas types converted to native Python types (int/float/str) for JSON serialization

        5) **Code Structure & Output Constraints:**
        - Script must end with ONLY `result_dict = {...}` statement
        - `result_dict` must be fully JSON-serializable Python dictionary
        - Prohibit external I/O operations (no file read/write)
        - Prohibit print statements or additional global variables

        6) **Code Quality Requirements:**
        - Ensure all variable names strictly follow above specifications
        - Clear logic, complete steps, strictly use provided data and best model file
        - Handle potential exceptions to improve stability and reliability

        Return: Complete Python code (only code itself, no explanatory text)."""
        )
        
        raw = self.call(prompt)
        
        return raw


    def check_abstract(self):
        if self.abstract is None:
            if self.code is None:
                self.abstract = None
            else:
                prompt = f"""
                This is the "Modeling Phase" within the data analysis workflow.

                Based on the information below, organize the content into a concise, coherent text summary while preserving all key information, to be used as a preview of the modeling section in report writing.

                === Input Information ===
                - User's initial requirement: {self.target}
                - Modeling code: {self.code}
                - Interaction records from modeling phase: {self.load_memory}
                - Modeling execution results: {self.result}

                === Output Requirements ===
                1. Write a summary in natural, fluent language that comprehensively covers the core information from the above content;
                2. Focus on explaining the modeling objectives, methods used, main implementation logic, and result characteristics;
                3. Avoid line-by-line code description, only extract the core concepts;
                4. Language should be professional and objective, avoiding vague expressions like "maybe," "seems," "perhaps," etc.;
                5. Output should be only one complete paragraph (no titles, numbering, or lists);
                6. The summary should enable readers to determine whether this section should be included in the final report.
                """.strip()

                desc = self.call(prompt)
                self.abstract = desc

        return self.abstract


    def check_full(self):
        if self.full is None:
            if self.code is None:
                self.full = None
            else:
                self.full = f"""
                [Phase Description]This is the data modeling phase within the data analysis workflow.
                [User's Initial Requirement]{self.target}
                [Data Modeling Code]{self.code}
                [Modeling Chat Dialogue]{self.load_memory}
                [Modeling Execution Results]{self.result}
                """.strip()

        return self.full
