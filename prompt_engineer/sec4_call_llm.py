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

        prompt = f"""
        请阅读以下建模建议，并将其转化为对下一个 coding agent 的清晰建模任务指令。

        === 建模建议 ===
        {self.suggestion}

        === 输出要求（必须严格遵守） ===
        1. 输出为纯文本，不使用任何 Markdown、编号或符号；
        2. 指令应简洁明确，便于 coding agent 直接理解并执行；
        3. 内容应聚焦于模型构建、训练或评估的具体任务；
        4. 避免解释性或分析性语言，仅描述“需要执行的操作”；
        5. 输出应覆盖所有关键步骤，使 coding agent 能独立完成建模流程。
        """.strip()

        refined_suggestions = self.call(prompt)
        self.refined_suggestions = refined_suggestions

        print(refined_suggestions)

        return refined_suggestions


    def code_generation(self, df_head: str, user_prompt: str) -> str:
        
        allowed = ", ".join(self.allowed_libs)

        if self.refined_suggestions is None:
            suggestion = user_prompt
        else:
            suggestion = self.refined_suggestions

        prompt = (
        f"""请**严格只输出纯 Python 代码**，**不要**输出任何解释性文字、注释、示例、markdown code fence（禁止出现 ``` 或 ```python 等）。运行环境已提供 pandas DataFrame 变量 `df`、numpy（np）、train_test_split、StandardScaler、以及用户在 Requirement 中可能提到的任意模型类（例如 RandomForestRegressor、GradientBoostingRegressor、LinearRegression、XGBRegressor、LogisticRegression、SVC 等）。

        要求：

        1) 使用 80/20 切分（random_state=42），根据用户需求决定是否对数值特征标准化（StandardScaler），如果标准化，务必只应用于数值列并在训练/测试集上分别执行 fit_transform/transform。
        2) **对 Requirement 中列出的所有模型都依次训练和评估**，不得只选随机森林；如果用户在 Requirement 中指定了多个模型名称，脚本必须循环遍历这些模型并分别训练、预测、计算指标。
        3) 不要导入任何评价库（如 sklearn.metrics），如需评价请用 numpy 手写实现常见指标（回归：MAE、MSE、R2；分类：accuracy、precision、recall、f1）。
        4) **脚本最后必须只输出并赋值一个变量 `result_dict`，且它是一个可以 JSON 序列化的 Python dict。**
        推荐 schema（必须包含以下键）：
        {{
            "dataset": "<可选描述字符串>",
            "models": [
            {{
                "name": "<模型类名>",
                "type": "<regression 或 classification>",
                "metrics": {{ "<指标名>": <float>, ... }}
            }},
            ...
            ],
            "best_model": {{
            "name": "<得分最优的模型类名>",
            "score": <float>
            }},
            "artifacts": {{
            "best_model_b64": "<base64 字符串>",
            "best_model_format": "pickle+gzip"
            }},
            // 如模型过大，可选 "artifact_warning": <int 字节大小>
            // 以及用户在 Requirement 中提出的其他字段
        }}
        5) 确保所有数值均为 Python 原生类型（float、int），字段名严格为 models、best_model、artifacts；如果用户有额外需求，如记录训练时间、特征重要性等，也请加入 result_dict。
        6) 模型导出：训练完毕后，将选定的 best_model 用 pickle 序列化并 gzip 压缩，再 base64 编码；把编码字符串和格式信息填入 result_dict["artifacts"]，并确保最终 result_dict 可 JSON 序列化。
        7) 脚本末尾仅包含一行 `result_dict = {{...}}`，不要 print、不创建其他全局变量、不读写文件。
        8) 如果模型序列化后的字节数超过合理大小，请在 result_dict 中添加 `"artifact_warning": <字节数>`。
        9) 不要使用任何外部 IO 或文件操作。
        10) 请准确实现Requirement中要求的模型，不许添加Requirement之外的模型，若先提供的库中无法直接调用对应模型，请手动实现！

        示例数据头部：
        {df_head}

        Requirement（请根据以下建模任务指令，对所有列出的模型依次执行训练与评估。若某模型在当前环境不可用，请手动实现对应算法或类，使结果完整可复现）：
        {suggestion}

        Allowed libraries: {allowed}。

        返回：完整 Python 代码（纯代码块）。"""
        )

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

        if st.session_state.preference_select:
            prompt += f"以下是用户的分析偏好设置：{st.session_state.preference_select}”。\n\n"
        if st.session_state.additional_preference:
            prompt += f"用户提供了以下建模目的与特殊需求：{st.session_state.additional_preference}”。\n\n"

        raw = self.call(prompt)

        return raw


    def result_format_prompt(self, result_json: str) -> str:

        prompt = f"""
                下面给出一个 JSON 对象（包含模型评估结果结构）。请将其转换为一份对人类友好的 Markdown 报告，输出要求如下：

                === 输出要求 ===
                1. 报告开头需有一行简短的“数据集说明”。
                2. 对每个模型，展示以下内容：
                - 模型名称；
                - 模型类型（分类 / 回归）；
                - 主要性能指标（如准确率、R²、MAE、MSE 等），每个指标保留 4 位小数；
                - 建议使用表格或分点列表清晰呈现。
                3. 明确标出 **best_model**（以粗体高亮显示其名称和最优指标）。
                4. 若 JSON 中包含特征工程相关信息，请在“特征工程说明”部分详细描述其具体方法与作用。
                5. 输出格式：
                - 只返回 Markdown 文本；
                - 不得使用任何代码块标记（如 ```、```markdown 等）；
                - 不输出解释性文字，仅输出最终报告内容（便于直接渲染于 Streamlit）。

                === 输入 JSON ===
                {result_json}
                """.strip()

        if st.session_state.preference_select:
            prompt += f"以下是用户的分析偏好设置：{st.session_state.preference_select}”。\n\n"
        if st.session_state.additional_preference:
            prompt += f"用户提供了以下建模目的与特殊需求：{st.session_state.additional_preference}”。\n\n"

        raw = self.call(prompt)

        return raw


    def get_model_suggestion(
        self,
        user_input=None,
        memory_limit: int = 6,
    ) -> str:

        df = self.load_df()
        df_head = df.head().to_string(index=False)
        columns = df.columns.tolist()
        data_info = f"数据列名: {columns}\n\n数据前5行:\n{df_head}"

        recent_memory = self.memory[-memory_limit:] if getattr(self, "memory", None) else []
        if recent_memory:
            formatted_memory = "\n".join(
                f"{m['role']}: {m['content']}" for m in recent_memory
            )
            memory_block = f"\n=== 历史上下文（仅供参考） ===\n{formatted_memory}\n"
        else:
            memory_block = ""

        prompt = f"""
        你是一位资深的机器学习建模专家，请基于以下信息进行分析与推理，输出针对性建模建议或改进方案。

        === 数据信息 ===
        {data_info}

        === 历史上下文（仅供参考） ===
        {memory_block}
        """.strip()

        if getattr(self, "target", None):
            prompt += f"""
            
            === 建模目标 ===
            {self.target}
            （请务必满足该目标，并在回答中明确复述建模意图。）
            """

        if user_input:
            prompt += f"""
            
            === 用户当前需求 ===
            {user_input}
            （请严格满足该需求。若为局部修改，请保留原逻辑，仅更新指定部分。）
            """

        train_code = self.load_code()
        if train_code:
            prompt += f"""

            === 历史训练代码 ===
            {train_code}

            请在充分理解上述代码的基础上，提出 **1–2 条高质量的模型改进建议**。
            可从以下角度思考，但不限于此：
            - 模型结构优化（如增加层数、调整激活函数、替换模型类型等）；
            - 特征工程改进（如变量选择、特征交互、归一化策略等）；
            - 训练流程优化（如正则化、学习率调度、损失函数调整等）；
            - 超参数调整（如树深度、学习率、batch size 等）。
            在给出建议时，请简要说明“为什么”与“预期改进效果”。
            """
        else:
            prompt += """
            
            === 建模建议任务 ===
            请根据数据特征和上下文，推荐 2–3 个适合的模型方案。
            要求：
            1. 每个模型需包含模型名称、主要原理、适用场景；
            2. 指出其在当前任务中的优势与潜在局限；
            3. 保持语言专业、简洁，不输出代码。
            """

        if st.session_state.preference_select:
            prompt += f"以下是用户的分析偏好设置：{st.session_state.preference_select}”。\n\n"
        if st.session_state.additional_preference:
            prompt += f"用户提供了以下建模目的与特殊需求：{st.session_state.additional_preference}”。\n\n"

        raw = self.call(prompt)
        return raw

    
    
    def summary_html(self) -> str:

        if self.code is None:
            
            summary = None

            return summary

        else:

            prompt = f"""
            你正在撰写数据分析报告的**第四章：数据建模**。
            请根据以下输入内容，综合分析并生成完整的章节正文。
            内容需逻辑严谨、表达自然，体现专业的分析与总结能力。

            === 输出结构 ===
            请严格按照以下五个小节组织内容：

            1. 概述
            - 说明本次建模的目标、研究背景及数据来源的上下文。

            2. 方法说明
            - 介绍所采用的模型或算法的核心思想与实现流程；
            - 若涉及特征工程、超参数选择或数据预处理，请一并说明；
            - 可适当涉及模型的数学原理或优化机制，以体现技术深度。

            3. 关键代码解读
            - 聚焦核心函数与模块，说明其在建模流程中的作用；
            - 可提及模型结构定义、训练循环、损失函数与评估逻辑；
            - 语言应清晰简练，避免逐行解释。

            4. 结果与评估
            - 概述主要性能指标（如 Accuracy、AUC、MSE 等）及结果表现；
            - 分析模型效果是否符合预期，并指出主要优劣与瓶颈。

            5. 改进建议
            - 针对模型性能与实验发现，提出具体可行的优化方向；
            - 可从模型结构、特征选择、训练策略或正则化等角度给出建议。

            === 写作要求 ===
            1. 使用自然流畅、正式的书面表达；
            2. 避免使用模糊或主观词汇（如“可能”“似乎”“微妙”等）；
            3. 注重逻辑连贯与专业性；
            4. 不输出标题、列表标记或额外说明，只生成正文内容。
            """.strip()

            if self.code is not None:
                prompt += f"=== 数据建模代码 ===\n\n{self.code}"
            if self.target is not None:
                prompt += f"=== 用户建模目标 ===\n\n{self.target}"
            if self.load_memory is not None:
                prompt += f"=== 数据建模聊天对话 ===\n\n{self.load_memory}"
            if self.result is not None:
                prompt += f"=== 建模运行结果 ===\n\n{self.result}"
            
            desc = self.call(prompt)

            summary = {
                        "title": "建模分析",
                        "code": self.code,
                        "desc": desc,
                        "result": self.result,
                    }

            return summary


    def summary_word(self) -> str:

        return self.summary_html()


    def code_generation_for_inference(self, code, inference_df_head) -> str:
        """生成 LLM prompt：要求 LLM 输出推断分析代码。"""
        
        prompt = (
        f"""请生成完整的 Python 推断分析脚本（仅返回代码，不要任何解释文字）。运行环境已提供 pandas DataFrame 变量 `inference_df`、已经 train 好的模型 `model_obj`、numpy（np）、StandardScaler 库、align_features 辅助函数，其余未提及的库请手写实现。要求：

        示例数据信息：
        {code}, inference_df 前五行: {inference_df_head}（请勿引入不存在 inference_df 中的变量）

        1) **可用变量说明：**
        - `inference_df`：推断数据集（Pandas DataFrame）
        - `model_obj`：已训练好的模型对象（从best_model.joblib加载）
        - `np`：NumPy库
        - `pd`：Pandas库
        - `StandardScaler`：用于数据标准化的sklearn工具

        2) **脚本必须实现的功能：**
        a) 对推断数据进行与训练时完全一致的预处理（例如，缺失值处理、编码转换、标准化等）
        b) **关键步骤：在预测前，必须使用align_features函数处理特征数据，确保特征数量和顺序与训练时一致**
        c) 使用model_obj对预处理并对齐后的特征数据进行预测
        d) 生成详细的推断报告，包含预处理步骤、预测结果分析等

        3) **预测结果处理要求：**
        - 将模型输出转换为人类可理解的形式（如概率值、类别标签、数值结果等）
        - **必须生成带预测结果的DataFrame**：将原始或处理后的`inference_df`与预测结果合并，命名为`inference_df_with_predictions`
        - 合并后的DataFrame必须包含原始特征列和一列名为`'prediction'`的预测结果列（模型输出多维时扩展为`prediction_0`, `prediction_1`, ...）

        4) **序列化要求（用于前端下载）：**
        - 将`inference_df_with_predictions`转换为无索引的CSV格式
        - 对CSV数据进行gzip压缩，然后编码为base64字符串
        - 创建包含以下键的`result_dict['artifacts']`字典：
          * `'predictions_df_b64'`：base64编码的压缩数据
          * `'predictions_df_format'`：固定值'csv+gzip'
          * `'predictions_df_size_bytes'`：压缩后的字节大小（整数）
        - 在`result_dict`中添加`'predictions_df_records'`键，值为`inference_df_with_predictions.to_dict(orient='records')`
        - 确保所有numpy/pandas类型转换为原生Python类型（int/float/str）以保证JSON可序列化

        5) **代码结构与输出约束：**
        - 脚本最后**仅**包含一行`result_dict = {...}`语句
        - `result_dict`必须是完全JSON可序列化的Python字典
        - 禁止任何外部IO操作（不读写文件）
        - 禁止使用print语句或创建额外的全局变量

        8) **生成代码质量要求：**
        - 确保所有变量名称与上述规范严格一致
        - 逻辑清晰，步骤完整，严格按照用户提供的数据和最佳模型文件生成代码
        - 处理可能出现的各种异常情况，提高代码的稳定性和可靠性

        返回：完整的Python代码（仅包含代码本身，不要任何解释性文字）。"""
        )
        
        raw = self.call(prompt)
        
        return raw


    def check_abstract(self):
        if self.abstract is None:
            if self.code is None:
                self.abstract = None
            else:
                prompt = f"""
                这是数据分析流程中的“建模阶段”。

                请基于以下信息，在保留所有关键信息的前提下，将内容整理成一段简洁、连贯的文字摘要，用于报告撰写中的建模小节预览。

                === 输入信息 ===
                - 用户初始需求：{self.target}
                - 建模代码：{self.code}
                - 建模阶段的交互记录：{self.load_memory}
                - 建模运行结果：{self.result}

                === 输出要求 ===
                1. 以自然流畅的语言撰写一段总结，全面涵盖上述内容中的核心信息；
                2. 重点说明建模目标、所用方法、主要实现逻辑与结果特征；
                3. 避免逐行描述代码，仅提炼核心思路；
                4. 语言应专业、客观，不使用“可能”“似乎”“也许”等模糊表达；
                5. 输出仅为一段完整文字（不要标题、编号或列表）；
                6. 摘要应能让人据此判断该部分是否需要纳入最终报告。
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
                【阶段说明】这是数据分析流程中的数据建模阶段。
                【用户初始需求】{self.target}
                【数据建模代码】{self.code}
                【建模聊天对话】{self.load_memory}
                【建模运行结果】{self.result}
                """.strip()

        return self.full
