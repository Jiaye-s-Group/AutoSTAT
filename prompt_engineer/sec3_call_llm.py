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
        请根据以下详细的可视化建议，提取每一列与每个变量组的推荐可视化方法。

        详细可视化建议:
        {rec}

        输出要求（必须严格遵守）：
        1. 输出为纯文本，每条独立换行，且不得有多余说明。
        2. 单变量格式：列名：图表1, 图表2。
        3. 多变量格式：关系组：列A,列B：图表1, 图表2。
        4. 总体变量格式：总体：图表1, 图表2。
        5. 严格不要添加标题、编号、示例或额外解释。
        6. 提取可视化方法精准。
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
            你是一位资深数据可视化专家，请根据以下信息，为数据分析报告的“可视化设计”章节提供系统、专业的建议。

            【数据集信息】
            - 数值型变量：{cols}
            - 数据维度：{dim_info}
            - 历史上下文（仅供参考）：{memory_block}

            【输出格式】
            请严格按照以下结构输出（保持标题和层级一致，不得增减）：

            一、单变量可视化（Univariate）
            1. 针对每个数值型变量，推荐 1–2 种最合适的可视化方法，并简要说明理由。
            例如：
            - `列1`：推荐“直方图（Histogram）”和“盒须图（Box Plot）”，理由：……

            二、多变量关系可视化（Multivariate）
            1. 从上述变量中选择 1–3 组值得重点分析的变量组合（每组包含 2–3 个变量），并说明选择理由。
            例如：
            - 关系组 1：`[列1, 列2]`，理由：……
            2. 对每一组变量，推荐最合适的可视化方法，并简要说明。
            例如：
            - 关系组 1：散点图（Scatter Plot）+ 回归线（Regression Line），理由：……

            三、整体分布可视化（Distribution Overview）
            1. 针对全数据的总体分布特征，推荐 1–2 种全局可视化方法，并说明用途。
            例如：
            - 推荐“小提琴图矩阵（Violin Plot Matrix）”，用途：……
            - 推荐“热力图（Heatmap）”，用途：……

            【执行要求】
            1. 若列名无实际意义（如索引、冗余 ID），应自动过滤；
            2. 输出内容需保持条理清晰、语言简洁、专业。
            """.strip()

        else:
            prompt = f"""
            你是一位资深数据可视化专家，请根据以下信息，请回应用户需求，实现用户需求：

            【用户需求】
            {user_input}

            【数据集信息】
            - 数值型变量：{cols}
            - 数据维度：{dim_info}
            - 数据概览（前几行）：
            {self.df.head().to_string(index=False)}
            - 历史上下文（仅供参考）：{memory_block}

            【执行要求】
            1. 若用户明确指定可视化列，仅针对这些列给出建议；
            2. 若用户提出特定要求（如图形大小、坐标轴 log 缩放等），必须在输出中体现；
            3. 仅响应用户需求，不输出无关内容；
            4. 若用户要求对先前内容进行局部修改，应保留未更动部分，仅更新相关建议；
            5. 输出内容应结构清晰、逻辑连贯、语言简洁。
            6. 禁止输出代码。
            """.strip()

        if st.session_state.preference_select:
            prompt += f"以下是用户的分析偏好设置：{st.session_state.preference_select}”。\n\n"
        if st.session_state.additional_preference:
            prompt += f"用户提供了以下建模目的与特殊需求：{st.session_state.additional_preference}”。\n\n"

        recommendations = self.call(prompt)
        return recommendations


    def desc_fig(self, fig, dtype_info):

        selected = st.session_state.selected_model

        if selected == "智谱AI" or selected == "通义千问" or selected == "GPT-4o" or selected == "GPT-5" or selected == "豆包" or selected == "Claude":
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
                    请综合下方可视化图与变量信息，进行**简洁但深入的分析**。
                    从分布形态、趋势特征、变量间关系、潜在异常现象、现实含义五个角度，提炼关键洞察。
                    输出一段不超过 120 字的自然语言分析结论（非摘要）。

                    【变量信息】
                    {dtype_info}

                    【图表结构信息】
                    {fig_info}

                    写作要求：
                    1. 分析需包含对数据异常的识别与说明：
                    - 若存在明显异常点、异常段或突变趋势，请指出其特征与潜在影响；
                    - 若未发现异常，也需明确说明整体分布稳定或无显著异常；
                    2. 内容需体现推理与解释性思考，而非表面描述；
                    3. 使用逻辑清晰、客观专业的语言；
                    4. 使用动词驱动句式（如“呈现出”“反映出”“揭示出”“说明了”等）；
                    5. 不使用模糊词（如“可能”“似乎”“微妙”等）；
                    6. 不使用标题、列表或格式符号；
                    7. 若变量含义中存在噪声或重复信息，请自动忽略；
                    8. 保持语气简洁有力，强调数据特征与分析结论。
                    """.strip()
                }
            ]

            desc_fig = self.call(prompt_payload)

        else:
            prompt = f"""  
            请综合下方可视化图与变量信息，从数据分布、趋势特征及潜在关系等角度进行分析。
            以不超过 100 字的自然语言总结关键发现，突出该变量在整体数据结构中的意义或异常现象。
            
            【变量信息】
            {dtype_info}

            【图表信息】
            {fig.to_dict()}

            写作要求：
            1. 语言应流畅自然，保持客观、专业；
            2. 使用简洁的动词和名词，不滥用形容词或副词；
            3. 避免“可能”“也许”“似乎”“微妙”等模糊词；
            4. 不添加标题或列表结构；
            5. 结合数据含义和图表特征，给出具有洞察力的简要结论；
            6. 若变量含义中存在杂乱或重复信息，请自动忽略。
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
                        "title": "数据可视化",
                        "fig_analysis": analysis,
                    }

            return summary


    def summary_word(self) -> str:
        
        analysis = self.summary_fig_analysis_list()

        if analysis is None:
            
            return None

        else:

            summary = {
                        "title": "数据可视化",
                        "fig_analysis": analysis,
                    }

            return summary


    def summary_fig_analysis_list(self) -> str:

        if not self.code:
            return self.analysis
        
        if self.analysis:
            return self.analysis

        selected = st.session_state.get("selected_model", "default")

        # --- 定义单个任务 ---
        def analyze_one(item, offset):
            fig = item["fig"]
            desc = item["desc"]

            selected = st.session_state.get("selected_model", "default")
            if isinstance(fig, go.Figure):
                if selected == "智谱AI" or selected == "通义千问" or selected == "GPT-4o" or selected == "GPT-5" or selected == "豆包" or selected == "Claude":
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
                            你正在撰写数据分析报告的第三章——《数据可视化》。  
                            请针对下方变量，结合其**业务含义、统计特征**与**可视化图表现**，撰写一段专业、逻辑严谨、可直接用于报告正文的分析内容。

                            【变量信息】
                            {self.cols_wo_id}

                            【Plotly 图表结构】
                            {fig_info}

                            【基础统计概览】
                            {desc}

                            【分析任务】
                            请在脑中先完成以下推理步骤，然后输出结构化正文：
                            1. 从图表识别核心模式：整体趋势、峰值、分布形态、异常点或聚集区；
                            2. 思考该模式与变量业务含义的关系；
                            3. 判断是否存在异常现象（单点异常、阶段性异常或结构性突变），并说明其潜在影响；
                            4. 若图中包含其他变量，请分析它们之间的统计或逻辑关联；
                            5. 将上述洞察整合成逻辑完整、语言自然的段落。

                            【输出格式（严格遵守）】
                            输出为纯文本，依次包含以下三部分（不使用 Markdown 或符号）：

                            1. 概述  
                            - 简述变量的定义、业务角色及数据表现的总体趋势；  
                            - 提出该变量在整体数据结构中可能的重要性。

                            2. 分布与特征分析  
                            - 从统计与图形角度分析其分布特征（集中趋势、离散程度、偏态、峰度、周期性等）；  
                            - 若发现异常或突变，请具体说明其表现形式与潜在机制；  
                            - 若与其他变量有关联趋势，指出方向与强度。

                            3. 实际含义与推论  
                            - 结合业务或研究背景，解释观察到的现象；  
                            - 分析其可能揭示的现实规律、风险或优化方向；  
                            - 若合适，可提出合理推测或后续分析建议（保持客观与逻辑自洽）。

                            【写作要求】  
                            1. 保持语言正式、专业、逻辑紧密；  
                            2. 句式多样、表达自然，避免模板化表述；  
                            3. 禁用模糊词汇（如“可能”“似乎”“大概”等）；  
                            4. 不使用任何标题符号（如 #、** 等）；  
                            5. 不输出“AI”“模型”“助手”等字样；  
                            6. 输出为连续正文，不包含解释性语句或附加说明。  
                            """.strip()
                                }
                            ]

                    analysis_text = self.call(prompt_payload)

                else:

                    prompt = f"""
                            你正在撰写数据分析报告的第三章——《数据可视化》。
                            请针对下方变量，结合其业务含义与对应的可视化图，撰写一段结构化、专业的分析文字。

                            【变量信息】
                            {self.cols_wo_id}

                            【Plotly 图表信息】
                            {fig.to_dict()}

                            【基础统计概览】
                            {desc}

                            请严格按照以下格式撰写内容（使用纯文本，不使用 Markdown 语法或符号）：

                            1. 概述
                            - 说明该变量的含义及其在数据或业务中的作用；  
                            - 简要描述整体分布特征或变量间的主要关联趋势。

                            2. 分布 / 关联特征
                            - 从统计角度说明变量的分布特征或相关关系；  
                            - 可引用关键统计量（均值、中位数、四分位数、相关系数等）支持分析。

                            3. 现实含义
                            - 结合变量在实际情境中的意义，解释所观察到的分布或关系；  
                            - 指出这些模式可能反映的现实现象或潜在影响（例如：某变量偏高代表风险上升或群体特征差异）。

                            【写作要求】  
                            1. 使用流畅、自然且正式的中文表达；  
                            2. 语言应客观、简洁，避免冗余修辞；  
                            3. 禁止使用“可能”“也许”“似乎”“微妙”等模糊词；  
                            4. 不使用标题符号（#、** 等）；  
                            5. 保持逻辑连贯，分析层次清晰。
                            """.strip()

                    analysis_text = self.call(prompt)
                    print(prompt)
                return offset, {"figure": fig, "analysis": analysis_text}

        # --- 并行执行 ---
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(analyze_one, item, i) for i, item in enumerate(self.fig)]
            for f in as_completed(futures):
                result = f.result()
                if result:
                    results.append(result)

        # --- 按原顺序排序 ---
        results.sort(key=lambda x: x[0])
        self.analysis = [r[1] for r in results]

        return self.analysis


    def code_generation(self, df_head: str, user_prompt: str) -> str:
        """生成 LLM prompt：要求 LLM 输出 result_dict（可 JSON 序列化）。"""
        allowed = ", ".join(self.allowed_libs)

        prompt = (
            "请**严格只输出纯 Python 代码**，**不要**输出任何解释性文字、注释、示例、markdown code fence（禁止出现 ``` 或 ```python 等）"
            "运行环境已提供 pandas DataFrame 变量 `df`、numpy（np）、"
            "plotly.express（px）、plotly.graph_objects（go）。\n\n"
            "##严格要求##：\n"
            "1) **严格执行用户需求**：若用户指定了要可视化的列，可能是精确列名，也可能是模糊输入"
            "（如输入 “ordera” 但实际列名为 “ordertypea”），不要凭空产生虚假列名！！！"
            f"请在脚本开头使用 LLM 理解将用户输入映射到 {df_head} 中最合适的真正列名，或采用更保守的索引（如第0列，第1列 推荐！），再仅对这些列绘制图表；\n"
            """2) **统计并重命名**：所有类别分布图请按下面模板写，**绝不直接用** `index` 作为列名——
            # === 模板：统计并绘制 Bar Chart ===
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

            3) 智能选图：根据数据类型（数值/类别）自动选择合适的图表。
            4) 自动检测是否需要按分类列着色，并做两种处理：若存在指定的分类列且想连续映射，先编码为数值 codes;如要离散映射，使用 parallel_categories
            5) 如 Plotly Express 中无合适图表，使用 `go.Figure` 自定义。
            6) 脚本末尾仅包含 `fig_dict = {...}`，不要 `print`、不要额外全局变量。
            7) 任何情况下不得“造”列名或直接写 `'index'`；若要使用索引，必须显式使用 `df.index`。
            8) 不要使用文件读写或其他外部 IO。
            9) 请只给我python代码，不要给我任何'''python等非代码内容的标识符。"""
            f"示例数据头部：\n{df_head}\n\n"
            f"每一张图的颜色必须从{self.color}中，选择\n\n"
            f"画图建议: {self.refined_suggestions}\n\n"
            "返回：完整 Python 代码（纯代码块）。"
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


    def check_abstract(self):
        if self.abstract is None:
            analysis_list = self.summary_fig_analysis_list()

            if not analysis_list :
                self.abstract = "暂无可视化分析内容。"
                return self.abstract

            all_analyses = "\n\n".join([
                f"【变量分析 {i+1}】\n{item['analysis']}"
                for i, item in enumerate(analysis_list)
            ])

            prompt = f"""
            请阅读并综合以下多个变量的分析内容：
            {all_analyses}

            任务：
            将这些分析整合为一段结构化、信息充分的**综合语义总结**，供后续大模型自动生成报告目录使用。

            目标：
            - 输出内容应帮助后续模型理解分析中包含的主题、变量、维度、关系与逻辑顺序；
            - 它将作为“目录生成模型”的输入，因此必须让模型能看出报告中应有哪些章节与子章节。

            写作要求：
            1. **信息保留**：
            - 保留每个变量的关键结论、趋势、特征、显著差异；
            - 明确变量间的联系、对比或影响；
            - 不得省略任何对分析主题有价值的事实。

            2. **结构导向**：
            - 按逻辑顺序组织：总体特征 → 各变量分析 → 变量间关系 → 潜在规律；
            - 若存在不同主题（如气象因素、污染物指标、模型结果），应自然体现层次；
            - 语义中隐含章节边界信号（如“首先…其次…最后…”、“在气象变量方面…”、“在建模部分…”等）。

            3. **语言风格**：
            - 专业、清晰、客观；
            - 使用完整句表达，不使用列表或编号；
            - 可以稍微详细，不追求简短。

            4. **输出格式**：
            - 输出仅为一段完整文字；
            - 不得加入标题、注释、JSON、代码块；
            - 该文字将被直接送入目录生成模型，不对人类展示。

            请生成符合上述要求的综合语义总结。
            """.strip()

            self.abstract = self.call(prompt)

        return self.abstract


    def check_full(self):
        """
        返回结构化的内容，遵守图片插入协议：
        - 每个分析内容前标注索引
        - 图片插入位置用 [FIG:index] 表示
        - 后续处理时可根据此协议替换为实际图像
        """
        if self.full is None:
            analysis_list = self.summary_fig_analysis_list()

            if not analysis_list :
                self.full = "暂无可视化分析内容。"
                return self.full

            # 构造结构化文本：带图片插入标记
            full_parts = ["""【阶段说明】这是数据分析流程中的数据可视化阶段。"""]
            for i, item in enumerate(analysis_list):
                desc = item["analysis"]
                part = f"""
                【对图 {i}的分析】
                {desc}
                [FIG:{i}]  # 图片插入位置标记
                """.strip()
                full_parts.append(part)

            self.full = "\n\n".join(full_parts)

            # 添加协议说明
            protocol_note = """
            ---
            # 图片插入处理协议说明：
            #  [FIG:index] 表示图片插入位置
            #  index 对应分析内容中的索引
            #  你在需要放图的地方用 [FIG:index] 代替即可
            """.strip()

            self.full = f"{self.full}\n\n{protocol_note}"

        return self.full


def extract_plotly_info(fig):

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
            raise ValueError(f"无法解析字符串形式的 Figure: {e}")
    else:
        raise TypeError(f"不支持的 fig 类型: {type(fig)}")

    layout = fig.get("layout", {})
    title = layout.get("title", {}).get("text", "")
    xaxis_title = layout.get("xaxis", {}).get("title", {}).get("text", "")
    yaxis_title = layout.get("yaxis", {}).get("title", {}).get("text", "")

    data_list = fig.get("data", [])
    types = list({d.get("type", "") for d in data_list})


    return {
        "title": title or "(无标题)",
        "xaxis": xaxis_title or "(无X轴标题)",
        "yaxis": yaxis_title or "(无Y轴标题)",
        "types": types,

    }
