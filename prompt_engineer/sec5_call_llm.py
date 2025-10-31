import streamlit as st
import openai
import requests
import json
import re
import pandas as pd
import numpy as np

from config import MODEL_CONFIGS
from prompt_engineer.call_llm import LLMClient


class ReportAgent(LLMClient):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.template = None
        self.name = None
        self.date = None
        self.report_format = None
        self.html = None
        self.word = None
        self.markdown = None
        self.user_input = None
        self.outline = None
        self.outline_length = None
        self.report= None
        self.finish_auto_task = False
        self.gen_mode = None


    def save_gen_mode(self, gen_mode):

        self.gen_mode = gen_mode


    def load_gen_mode(self):

        return self.gen_mode 


    def finish_auto(self):

        self.finish_auto_task = True


    def save_user_input(self, user_input):

        self.user_input = user_input


    def load_user_input(self):

        return self.user_input
    
    
    def save_outline_length(self, outline_length):

        self.outline_length = outline_length


    def load_outline_length(self):

        return self.outline_length


    def save_outline(self, outline):

        self.outline = outline


    def load_outline(self):

        return self.outline


    def save_template(self, template):

        self.template = template


    def load_template(self):

        return self.template


    def save_word(self, word):

        self.word = word


    def load_word(self):

        return self.word


    def save_html(self, html):

        self.html = html


    def load_html(self):

        return self.html


    def save_markdown(self, markdown):

        self.markdown = markdown


    def load_markdown(self):

        return self.markdown
    

    def save_report(self, report):

        self.report = report


    def load_report(self):

        return self.report


    def save_report_format(self, report_format):

        self.report_format = report_format


    def load_report_format(self):

        return self.report_format


    def save_date(self, date):

        self.date = date


    def load_date(self):

        return self.date
    
    
    def save_name(self, name):

        self.name = name


    def load_name(self):

        return self.name
    

    def generate_template(self, user_input = None) -> str:

        prompt = (
        """
        我希望你输出一个现代、简洁且美观的 HTML 章节模板，请满足以下要求：

        1. 整体配色采用“蓝 – 白”主题：
        - 背景为白色，标题与边框使用深蓝（#1E3A8A）和浅蓝（#3B82F6）；
        2. 最外层用 `<section class="chapter" id="chapter-{{ num }}">` 包裹；
        3. 标题使用 `<h2>{{ title }}</h2>`：
        - 文字颜色：#1E3A8A；
        - 下方装饰性下划线：高度 3px，颜色 #3B82F6，宽度 30%；
        4. 正文内容区 `<div class="content">{{ body }}</div>`，支持任意 HTML；
        - **仅对“重点摘录”或“引用”段落加用圆角矩形**，其余普通段落保持标准 `<p>` 样式；
        - 圆角矩形样式：背景 #EFF6FF，padding 12px，border-radius 8px，margin-bottom 16px；
        5. 如果有图片列表 `images`：
        - ≤3 张时水平并排；>3 张时自动换行，每行最多 3 张；
        - `<img>` 带 6px 圆角、轻微阴影 `box-shadow:0 2px 6px rgba(0,0,0,0.1)`；
        6. 在 `<style>` 中内联基础样式：
        - `.chapter` 外层间距、内边距、最大宽度、白底阴影；
        - `.chapter h2` 字体、颜色、下划线；
        - `.content p` 和 `.content .highlight`（重点段落）样式区分；
        - `.images` 的 flex 布局与 gap；  
        7. 使用 Jinja2 占位符：
        - 普通段落：`{% for p in paragraphs %}<p>{{ p }}</p>{% endfor %}`；
        - 重点段落数组 `highlights`：`{% for h in highlights %}<div class="highlight">{{ h }}</div>{% endfor %}`；
        8. **只输出完整的 `<section>…</section>` 片段**，不要任何解释文字或其他标签。
        9. 在模板的 .content 区域加入一个 DataFrame 占位并用 Jinja2 渲染变量 df_html（{{ df_html | safe }}），要求输出为响应式 HTML 表格（显示表头、支持横向滚动并在窄屏下自动换行），以便在导出为 PDF 时正确排版。

        请直接给出最终的 HTML 模板代码。
        """
        )

        if user_input is not None:
            prompt += f"请根据用户需求进行调整{user_input}"

        return self.call(prompt)


    def fill_report(self, template: str, content: str) -> str:

        prompt = (f"""  
            下面是章节结构模板：
            {template}
            请仅输出 `<section>` 里完整的 HTML（包括标题、正文、图片区块），请将重点内容用highlight凸显，
            对于内容的分析具有一下要求：
            1. 要用流畅的自然语言
            2. 不要滥用形容词和副词，尽量用简单的动词和名词表达意思
            3. 不用"可能""也许""似乎""微妙"等模糊表述
            请根据一下提供的信息对文章进行深入分析：
            """)

        if content.get("title") is not None:
            prompt += f"- title={content['title']}\n"
        if content.get("fig_analysis") is not None:
            prompt += f"- images及其分析（请将image也放入报告中）：{content['fig_analysis']}\n"
        if content.get("df") is not None:
            prompt += f"- 表格预览（请将表格也放入报告中，输出美观完整）：{content['df']}\n"  
        if content.get("code") is not None:
            prompt += f"- 对应部分代码（请将代码中的重点公式与内容进行讲解与分析）：{content['code']}\n"  
        if content.get("processed_df") is not None:
            prompt += f"- 预处理后的数据预览：{content['processed_df']}\n"  
        if content.get("desc") is not None:
            prompt += f"- 具体内容分析：{content['desc']}\n"  
        if content.get("header") is not None:
            prompt = f"""
            下面是章节结构模板：
            {template}
            要求：header单独占一页
            - 请为我生成封面header：{content['header']}
            """
        if content.get("footer") is not None:
            prompt = f"""
            下面是章节结构模板：
            {template}
            要求：footer单独占一页
            - 请为我生成最后一页footer：{content['footer']}
            """

        prompt += "请仅返回提供html"

        return self.call(prompt)


    def fill_report_word(self, content: str) -> str:


        prompt = (f"""  
            你是一个资深的数据分析专家，
            请仅输出每一章节的完整的word内容（包括标题、正文、图片区块），
            对于内容的分析具有一下要求：
            1. 要用流畅的自然语言
            2. 不要滥用形容词和副词，尽量用简单的动词和名词表达意思
            3. 不用"可能""也许""似乎""微妙"等模糊表述
            请根据一下提供的信息对文章进行深入分析：
            """)

        if content.get("title") is not None:
            prompt += f"- title={content['title']}\n"
        if content.get("fig_analysis") is not None:
            prompt += f"- images及其分析（请将image也放入报告中）：{content['fig_analysis']}\n"
        if content.get("df") is not None:
            prompt += f"- 表格预览（请将表格也放入报告中，输出美观完整）：{content['df']}\n"  
        if content.get("code") is not None:
            prompt += f"- 对应部分代码（请将代码中的重点公式与内容进行讲解与分析）：{content['code']}\n"  
        if content.get("processed_df") is not None:
            prompt += f"- 预处理后的数据预览：{content['processed_df']}\n"  
        if content.get("desc") is not None:
            prompt += f"- 具体内容分析：{content['desc']}\n"  
        if content.get("header") is not None:
            prompt = f"""
            下面是章节结构模板：
            {template}
            要求：header单独占一页
            - 请为我生成封面header：{content['header']}
            """
        if content.get("footer") is not None:
            prompt = f"""
            下面是章节结构模板：
            {template}
            要求：footer单独占一页
            - 请为我生成最后一页footer：{content['footer']}
            """

        prompt += "请仅返回提供html"

        return self.call(prompt)


    def get_content(self, agent):

        content = agent.summary()

        return content

    def generate_toc_from_summary(self, full_summary) -> str:
        """
        调用大模型，根据已有 summary 内容自动生成带有分级结构与内容大纲的目录（最多 2 级标题）
        """

        prompt = f"""
        你是一位资深数据分析报告结构设计专家。

        请你根据以下报告摘要内容，为该数据分析报告生成**层次清晰、内容具体、贴合数据本身**的目录结构。

        【输出要求】
        1. 格式：
        - 纯文本输出（不得使用 Markdown、代码块、Python 列表或符号标记）
        - 每行一个目录项，无缩进或前缀符号
        - 示例格式：
            1.概述（说明报告背景与目标）
            2.数据导入（说明数据来源与结构）
            2.1 数据概览（展示核心字段与样本规模）
            2.1.1 租赁数量趋势（分析租赁随时间的变化）
        2. 编号规则：
        - 一级标题：1, 2, 3...
        - 二级标题：2.1, 2.2...
        - 三级标题：2.1.1, 2.1.2...
        3. 内容说明：
        - 所有标题与说明应以摘要为基础，可在保持主题一致的前提下，适度补充逻辑性或结构性内容。
        - 每个标题后附一句说明，用于指导后续大模型撰写章节内容；
        - 说明须以中文括号“（ ）”包裹；
        - 每条说明需精准、具体，**明确指示该部分的写作任务、分析角度、数据焦点或方法方向**；
        - 字数不超过 50 字；
        - 上下级说明应保持语义连贯，避免重复；
        - 说明可涉及：
            - 要分析的变量或主题（如“气温”“租赁数量”“污染物浓度”）；
            - 要执行的任务（如“展示分布”“分析趋势”“比较模型性能”）；
        4. 禁止输出任何解释、前言、说明、提示、或多余空行，仅输出目录正文。

        【生成逻辑】
        1. 依据摘要内容中出现的主题（如数据特征、指标、变量名、任务目标）生成章节标题。
        - 若摘要中提及 “租赁数量”“气温”“湿度”“时间”等，请将其体现在相关标题中。
        - 避免使用模糊标题（如“数据分析”“关系探索”“模型评估”等）。
        2. 报告可能包含模块：
        “数据导入”、“数据预处理”、“数据可视化”、“建模分析”。
        - 仅生成摘要中实际涉及的模块。
        3. 确保章节间语义互斥（正交），避免内容重叠。
        4. 根据详细程度动态调整层级：
        - 简要：生成两级标题；
        - 标准：生成三级标题；
        - 详细：生成四级标题。
        5. 若摘要涉及具体变量（如“Temperature”、“Rented Bike Count”），
        请在目录中直接引用中文变量名（如“气温”、“租赁数量”），
        以体现报告的“数据感知性”。

        用户选择的目录详细程度为：{self.outline_length}

        报告摘要如下：
        {full_summary}
        """

        if st.session_state.preference_select:
            prompt += f"以下是用户的分析偏好设置：{st.session_state.preference_select}”。\n\n"
        if st.session_state.additional_preference:
            prompt += f"用户提供了以下建模目的与特殊需求：{st.session_state.additional_preference}”。\n\n"

        toc_response = self.call(prompt)
        return toc_response.strip()


    def selected_photo_update_toc(self, toc, selected_full_contents_vis: str) -> list:
 
        print(selected_full_contents_vis)

        prompt = f"""
        你是一位专业的数据分析报告结构与图文匹配专家。

        任务：请你根据报告的目录结构和，正文内容和阶段说明，判断每个 [FIG:x] 图像最合适归属的章节。

        【输入内容】
        1. 目录结构（含标题、层级、内容大纲）：
        {toc}

        2. 报告完整正文（带有 [FIG:x] 图片标记）：
        {selected_full_contents_vis}

        【任务说明】
        请你逐一分析每个 [FIG:x] 图像的出现上下文，并结合目录内容，判断该图应归属于哪个章节。  
        要求同时考虑：
        - **语义匹配**：图像内容的主题（如污染物趋势、气象变化、时间分布、模型结果）与章节描述的一致性；
        - **上下文位置**：图像在正文中出现时，其前后段落通常属于哪个章节；
        - **粒度优先**：若图像语义符合多个章节（如“气象参数”与“气象参数图形分析”），优先归入更具体的章节（层级数字更大）；
        - **禁止误归**：禁止将图像分配到“概述”“结论”“摘要”等非分析或与图像不相关的章节！
        - **全部使用**：所有 [FIG:x] 必须被使用一次，不得遗漏或重复。

        【输出格式】
        请以 Python 列表形式输出，每项为：
        (标题, 层级, 内容大纲, 图编号列表)
        要求：
        - 图编号按出现顺序排列；
        - 若无图片则为空列表 [];
        - 层级仅用整数表示（1, 2, 3...）；
        - 不输出任何解释、注释、Markdown标记。

        【示例格式】
        [
        ('概述',1,'说明报告背景与目标',[]),
        ('数据导入',1,'说明数据来源与结构',[]),
        ('数据可视化',1,'展示变量特征与关系',[4,5]),
        ('气象参数分析',2,'研究温度与湿度对污染的影响',[2,3]),
        ('模型评估',2,'展示预测结果与误差',[6,7])
        ]

        【提示与约束】
        1. 若章节间存在嵌套关系，优先分配给最具体的子章节（如 3.1.2 比 3.1 更优）。
        """

        if st.session_state.preference_select:
            prompt += f"以下是用户的分析偏好设置：{st.session_state.preference_select}”。\n\n"
        if st.session_state.additional_preference:
            prompt += f"用户提供了以下建模目的与特殊需求：{st.session_state.additional_preference}”。\n\n"

        toc_with_figs = self.call(prompt)
        return toc_with_figs.strip()


    def summarize_all_sections(
        self,
        toc_md: str,
        load_summary: str,
        preproc_summary: str,
        visual_summary: str,
        coding_summary: str
    ) -> str:


        # Step 1：拼接所有 agent 的摘要
        section_summaries = {
            "加载阶段": load_summary,
            "预处理阶段": preproc_summary,
            "可视化分析": visual_summary,
            "模型建构": coding_summary,
        }

        # Step 2：构建大模型 prompt
        prompt = f"""你现在是一个经验丰富的数据分析报告撰写助手。

        我已经完成了一个数据分析项目的初稿，结构目录如下：
        {toc_md}

        现在我将为你提供各个章节的内容摘要，请你根据这些内容，用流畅的中文撰写一段总结性描述（可用于报告的导语或结语），要求包括但不限于：

        1. 报告分析的主题方向  
        2. 各章节的核心处理逻辑和大致作用  
        3. 报告内容的整体风格与结构特性（例如是否包含图表、是否强调建模等）  
        4. 使用自然语言、风格正式，避免主观判断词汇（如“也许”、“不错”、“感觉”）  
        5. 最终输出 150~300 字中文总结段落，不需要标题  

        每个阶段摘要如下：\n\n"""

        for title, content in section_summaries.items():
            if content:
                prompt += f"\n【{title}】\n{content}\n"

        # 调用大模型总结
        overall_summary = self.call(prompt)

        return overall_summary


    def update_toc_with_relevant_sections(self, toc, agent_abstracts):

        prompt = f"""
        你是一个专业的数据分析报告规划助手。
        我将提供报告目录和各分析模块的摘要，请为每个章节确定应参考的模块编号列表。
        
        报告目录（每个元素为四元组：标题、层级、内容大纲、图编号列表）：
        {toc}

        各数据分析模块摘要如下：
        {agent_abstracts}
        请根据：
        1. 各章节的标题、层级与内容大纲；
        2. 各数据处理板块摘要；
        3. 各章节的图编号分配情况（报告目录第四项）；

        合理判断各章节在生成报告时应参考哪些数据处理板块的信息。
        输出要求：

        - 对每个章节生成一个五元组 (标题, 层级, 内容大纲, 图编号列表, 模块编号列表)
            - 标题, 层级, 内容大纲, 图编号列表一定不能改变，只在原有基础上添加第五项
        - 模块编号列表为 Python list，例如 [0, 2]
        - 若无需参考任何模块，返回 []
        - 输出为 Python 列表，不含任何额外说明
        示例：
        输入：
        [
          ('概述',1,'介绍报告背景与目标',[1]),
          ('数据可视化',1,'分析空气质量和相关环境变量的可视化图表',[2,3]),
          ('xxxx关联性分析',2,'分析相对湿度与其他污染物关系',[4,5])
        ]
        输出：
        [
          ('概述',1,'介绍报告背景与目标',[1],[1,2]),
          ('数据可视化',1,'分析空气质量和相关环境变量的可视化图表',[2,3],[0,1]),
          ('xxxx关联性分析',2,'分析相对湿度与其他污染物关系',[4,5],[2,3])
        ]
        """
        toc_with_sections = self.call(prompt)
        print(toc_with_sections)
        return toc_with_sections.strip()


    def write_section_body(self, toc, t, selected_full_contents, history_content):

        prompt = f"""
        你是一个专业的数据分析报告撰写助手。你的任务是基于我提供的参考信息，生成逻辑清晰、结构严谨、内容专业的报告章节。

        当前章节信息（四元组：标题、层级、内容大纲、图编号列表）：
        {t}

        报告目录结构（包含所有章节的四元组信息）：
        {toc}

        可参考的分析内容如下：
        {selected_full_contents}

        此前已生成的章节内容如下（用于保持整体风格一致、避免重复）：
        {history_content}

        写作要求：

        一、写作目标
        1. 仅撰写当前章节“{t[0]}”的正文内容；
        2. 内容必须以“参考信息”为核心依据，可在其逻辑框架内**进行适度拓展与归纳总结**；
        3. 允许进行合理的专业性补充（如统计学解释、方法原理、结果含义），但**禁止编造具体数据、图表结果、实验场景或样本特征**；
        4. 若参考信息不足，可补充一般性分析思路，但需保持内容通用、客观、抽象，不得具体化为假想数据。

        二、语言与结构
        1. 文风应正式、专业、学术化；
        2. 论述应符合数据分析逻辑：先描述、后解释、再总结；
        3. 每一自然段应围绕一个逻辑核心展开（如趋势、对比、相关性、分布特征等）。

        三、图表使用规范
        1. 正文中仅可使用本章节的图编号 {t[3]}；
        2. 使用占位符 [FIG:index] 标注图表位置；
        3. 在每个占位符下方添加图片标题：
            图：图片标题（简要说明图片内容及分析要点）
        4. 图片位置与语义保持自然衔接：
        - 若图片引出分析 → 放在段落开头；
        - 若图片支撑论点 → 放在相关描述句之后；
        - 若图片总结结果 → 放在段落结尾；
        5. 不得增删或重排图片编号。

        四、输出要求
        - 仅输出正文内容；
        - 不得输出标题、编号、解释文字、Markdown；
        - 不使用加粗、斜体、符号修饰或非正文语句；
        - 不得出现“我认为”、“请继续”、“综上可见”等主观表达。

        五、写作模式
        当前模式：{self.outline_length}
        - 简要：仅写结论；
        - 标准：含逻辑与结论；
        - 详细：包含推理与方法，但仍应基于参考信息，不得自由创作。

        请严格在以上范围内撰写本章节正文。
        """

        if st.session_state.preference_select:
            prompt += f"以下是用户的分析偏好设置：{st.session_state.preference_select}”。\n\n"
        if st.session_state.additional_preference:
            prompt += f"用户提供了以下建模目的与特殊需求：{st.session_state.additional_preference}”。\n\n"

        content = self.call(prompt)

        return content