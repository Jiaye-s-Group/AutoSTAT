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
        """
        Call an LLM to generate an HTML report template with placeholders,
        including sections for title, summary, tables, charts, etc.
        """
        prompt = (
        """
        I want you to output a modern, clean, and aesthetically pleasing HTML chapter template. Please meet the following requirements:

        1. Overall color scheme uses a "Blue – White" theme:
        - Background: white; titles and borders use dark blue (#1E3A8A) and light blue (#3B82F6);
        2. Wrap the outermost layer with `<section class="chapter" id="chapter-{{ num }}">`;
        3. Title uses `<h2>{{ title }}</h2>`:
        - Text color: #1E3A8A;
        - Decorative underline below: height 3px, color #3B82F6, width 30%;
        4. Main content area `<div class="content">{{ body }}</div>`, supporting any HTML;
        - **Apply rounded rectangles only to "key excerpts" or "quoted" paragraphs**; keep standard `<p>` styling for other regular paragraphs;
        - Rounded rectangle style: background #EFF6FF, padding 12px, border-radius 8px, margin-bottom 16px;
        5. If there is an image list `images`:
        - Display horizontally side-by-side when ≤3 images; automatically wrap lines when >3 images, with a maximum of 3 images per row;
        - `<img>` with 6px rounded corners and subtle shadow `box-shadow:0 2px 6px rgba(0,0,0,0.1)`;
        6. Inline basic styles in `<style>`:
        - `.chapter` outer spacing, padding, max-width, white background shadow;
        - `.chapter h2` font, color, underline;
        - Style differentiation for `.content p` and `.content .highlight` (key paragraphs);
        - Flex layout and gap for `.images`;
        7. Use Jinja2 placeholders:
        - Regular paragraphs: `{% for p in paragraphs %}<p>{{ p }}</p>{% endfor %}`;
        - Key paragraph array `highlights`: `{% for h in highlights %}<div class="highlight">{{ h }}</div>{% endfor %}`;
        8. **Output only the complete `<section>…</section>` fragment**, no explanatory text or other tags.
        9. Add a DataFrame placeholder in the template's .content area and render the variable df_html using Jinja2 ({{ df_html | safe }}). Require output as a responsive HTML table (display headers, support horizontal scrolling, and wrap automatically on narrow screens) to ensure correct layout when exported to PDF.

        Please provide the final HTML template code directly.
        """
        )

        if user_input is not None:
            prompt += f"Please adjust according to user requirements: {user_input}"

        return self.call(prompt)


    def fill_report(self, template: str, content: str) -> str:
        """
        Convert the DataFrame to an HTML table, splice it into the template,
        and have the LLM polish the report and supplement explanatory text.
        """

        prompt = (f"""  
            Below is the chapter structure template:
            {template}
            Please output only the complete HTML within `<section>` (including title, main content, image blocks), and highlight key content with highlight.
            For content analysis, the following requirements must be met:
            1. Use fluent natural language
            2. Do not overuse adjectives and adverbs; try to express meaning with simple verbs and nouns
            3. Do not use vague expressions such as "might", "perhaps", "seems", "subtle", etc.
            Please conduct an in-depth analysis of the article based on the following provided information:
            """)

        if content.get("title") is not None:
            prompt += f"- title: {content['title']}\n"
        if content.get("fig_analysis") is not None:
            prompt += f"- Images and their analysis (please include the images in the report): {content['fig_analysis']}\n"
        if content.get("df") is not None:
            prompt += f"- Table preview (please include the table in the report, output beautifully and completely): {content['df']}\n"  
        if content.get("code") is not None:
            prompt += f"- Corresponding code section (please explain and analyze the key formulas and content in the code): {content['code']}\n"  
        if content.get("processed_df") is not None:
            prompt += f"- Preprocessed data preview: {content['processed_df']}\n"  
        if content.get("desc") is not None:
            prompt += f"- Detailed content analysis: {content['desc']}\n"  
        if content.get("header") is not None:
            prompt = f"""
            Below is the chapter structure template:
            {template}
            Requirement: header should occupy a separate page
            - Please generate a cover header for me: {content['header']}
            """
        if content.get("footer") is not None:
            prompt = f"""
            Below is the chapter structure template:
            {template}
            Requirement: footer should occupy a separate page
            - Please generate a footer for the last page for me: {content['footer']}
            """

        prompt += "Please only return the provided HTML"

        return self.call(prompt)


    def fill_report_word(self, content: str) -> str:

        prompt = (f"""  
            You are a senior data analysis expert,
            Please output only the complete Word content for each chapter (including title, main body, image blocks),
            The analysis of the content has the following requirements:
            1. Use fluent natural language
            2. Do not overuse adjectives and adverbs; try to express meaning with simple verbs and nouns
            3. Do not use vague expressions such as "might", "perhaps", "seems", "subtle", etc.
            Please conduct an in-depth analysis of the article based on the following provided information:
            """)

        if content.get("title") is not None:
            prompt += f"- title: {content['title']}\n"
        if content.get("fig_analysis") is not None:
            prompt += f"- Images and their analysis (please include the images in the report): {content['fig_analysis']}\n"
        if content.get("df") is not None:
            prompt += f"- Table preview (please include the table in the report, output beautifully and completely): {content['df']}\n"  
        if content.get("code") is not None:
            prompt += f"- Corresponding code section (please explain and analyze the key formulas and content in the code): {content['code']}\n"  
        if content.get("processed_df") is not None:
            prompt += f"- Preprocessed data preview: {content['processed_df']}\n"  
        if content.get("desc") is not None:
            prompt += f"- Detailed content analysis: {content['desc']}\n"  
        if content.get("header") is not None:
            prompt = f"""
            Below is the chapter structure template:
            {template}
            Requirement: header should occupy a separate page
            - Please generate a cover header for me: {content['header']}
            """
        if content.get("footer") is not None:
            prompt = f"""
            Below is the chapter structure template:
            {template}
            Requirement: footer should occupy a separate page
            - Please generate a footer for the last page for me: {content['footer']}
            """

        prompt += "Please only return the provided HTML"

        return self.call(prompt)


    def get_content(self, agent):

        content = agent.summary()

        return content

    def generate_toc_from_summary(self, full_summary) -> str:
        """
        Call the large model to automatically generate a table of contents with a hierarchical structure and content outline based on the existing summary content (up to 2 levels of headings)
        """

        prompt = f"""
        You are a senior data analysis report structure design expert.

        Please generate a table of contents for this data analysis report based on the following report summary content, which should be **hierarchically clear, content-specific, and closely aligned with the data itself**.

        [Output Requirements]
        1. Format:
        - Plain text output (must not use Markdown, code blocks, Python lists, or symbol markers)
        - One directory item per line, without indentation or prefix symbols
        - Example format:
            1.Overview (Explain the report background and objectives)
            2.Data Import (Describe the data source and structure)
            2.1 Data Overview (Display core fields and sample size)
            2.1.1 Rental Quantity Trend (Analyze rental changes over time)
        2. Numbering rules:
        - Level 1 headings: 1, 2, 3...
        - Level 2 headings: 2.1, 2.2...
        - Level 3 headings: 2.1.1, 2.1.2...
        3. Content description:
        - All headings and descriptions should be based on the summary, and logical or structural content can be appropriately supplemented while maintaining thematic consistency.
        - Each heading should be followed by a description sentence to guide the subsequent large model in writing chapter content;
        - Descriptions must be enclosed in parentheses;
        - Each description must be precise and specific, **clearly indicating the writing task, analysis angle, data focus, or method direction for that section**;
        - Word count should not exceed 50 characters;
        - Descriptions between upper and lower levels should maintain semantic coherence and avoid repetition;
        - Descriptions may involve:
            - Variables or themes to be analyzed (such as "temperature", "rental quantity", "pollutant concentration");
            - Tasks to be performed (such as "display distribution", "analyze trends", "compare model performance");
        4. It is forbidden to output any explanations, prefaces, descriptions, prompts, or extra blank lines; only output the main text of the table of contents.

        [Generation Logic]
        1. Generate chapter headings based on themes present in the summary content (such as data features, indicators, variable names, task objectives).
        - If the summary mentions "rental quantity", "temperature", "humidity", "time", etc., reflect them in the relevant headings.
        - Avoid using vague headings (such as "Data Analysis", "Relationship Exploration", "Model Evaluation", etc.).
        2. The report may include modules:
        "Data Import", "Data Preprocessing", "Data Visualization", "Modeling Analysis".
        - Only generate modules that are actually involved in the summary.
        3. Ensure semantic mutual exclusion (orthogonality) between chapters to avoid content overlap.
        4. Dynamically adjust the hierarchy based on the level of detail:
        - Brief: Generate two levels of headings;
        - Standard: Generate three levels of headings;
        - Detailed: Generate four levels of headings.
        5. If the summary involves specific variables (such as "Temperature", "Rented Bike Count"),
        Please directly reference the variable names in the table of contents (such as "temperature", "rental quantity"),
        To reflect the "data awareness" of the report.

        The directory detail level selected by the user is: {self.outline_length}

        The report summary is as follows:
        {full_summary}
        """
        
        if st.session_state.preference_select:
            prompt += f"The user's analysis preferences are as follows: {st.session_state.preference_select}.\n\n"
        if st.session_state.additional_preference:
            prompt += f"The user has provided the following modeling goals and special requirements: {st.session_state.additional_preference}.\n\n"

        toc_response = self.call(prompt)
        return toc_response.strip()


    def selected_photo_update_toc(self, toc, selected_full_contents_vis: str) -> list:
        """
        Based on the full report content selected_full_contents_vis, update the toc, adding a fourth item to each section: the corresponding list of image numbers.
        """
        print(selected_full_contents_vis)

        prompt = f"""
        You are a professional data analysis report structure and image-text matching expert.

        Task: Please judge the most appropriate chapter for each [FIG:x] image to belong to, based on the report's table of contents structure, main text content, and phase descriptions.

        [Input Content]
        1. Table of contents structure (including titles, levels, content outlines): {toc}

        2. Full report text (with [FIG:x] image tags): {selected_full_contents_vis}

        [Task Description]
        Please analyze the context of each [FIG:x] image's appearance one by one, and combine it with the table of contents to judge which chapter the image should belong to.  
        Requirements to consider simultaneously:
        - **Semantic Matching**: Consistency between the theme of the image content (such as pollutant trends, meteorological changes, time distribution, model results) and the chapter description;
        - **Contextual Position**: When the image appears in the main text, which chapter the paragraphs before and after it usually belong to;
        - **Granularity Priority**: If the image semantics fit multiple chapters (e.g., 'Meteorological Parameters' and 'Meteorological Parameter Graphical Analysis'), prioritize assigning it to the more specific chapter (with a larger level number);
        - **Prohibit Misassignment**: It is forbidden to assign images to non-analytical or image-unrelated chapters such as 'Overview', 'Conclusion', 'Summary', etc.!;
        - **Use All**: All [FIG:x] must be used once, with no omissions or repetitions.

        [Output Format]
        Please output in the form of a Python list, each item as:
        (title, level, content outline, list of image numbers)
        Requirements:
        - Image numbers should be arranged in the order of appearance;
        - If there are no images, use an empty list [];
        - Levels are represented only by integers (1, 2, 3...);
        - Do not output any explanations, comments, or Markdown markers.

        [Example Format]
        [
        ('Overview',1,'Explain the report background and objectives',[]),
        ('Data Import',1,'Describe the data source and structure',[]),
        ('Data Visualization',1,'Display variable characteristics and relationships',[4,5]),
        ('Meteorological Parameter Analysis',2,'Study the impact of temperature and humidity on pollution',[2,3]),
        ('Model Evaluation',2,'Display prediction results and errors',[6,7])
        ]

        [Tips and Constraints]
        1. If there are nested relationships between chapters, prioritize assigning to the most specific subchapter (e.g., 3.1.2 is better than 3.1).
        """

        if st.session_state.preference_select:
            prompt += f"The user's analysis preferences are as follows: {st.session_state.preference_select}.\n\n"
        if st.session_state.additional_preference:
            prompt += f"The user has provided the following modeling goals and special requirements: {st.session_state.additional_preference}.\n\n"

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
        """
        Summarize all agents' summaries and perform a textual summary based on the toc_md structure.
        """

        # Step 1: Concatenate all agents' summaries
        section_summaries = {
            "Loading Phase": load_summary,
            "Preprocessing Phase": preproc_summary,
            "Visualization Analysis": visual_summary,
            "Model Construction": coding_summary,
        }

        # Step 2: Construct the large model prompt
        prompt = f"""You are now an experienced data analysis report writing assistant.

        I have completed a draft of a data analysis project, with the structure directory as follows:
        {toc_md}

        Now I will provide you with the content summaries of each chapter. Please write a summary description in fluent English based on this content (which can be used as the introduction or conclusion of the report). The requirements include but are not limited to:
 
        1. The thematic direction of the report analysis  
        2. The core processing logic and general role of each chapter  
        3. The overall style and structural characteristics of the report content (e.g., whether it includes charts, emphasizes modeling, etc.)  
        4. Use natural language, formal style, avoid subjective judgment words (such as 'maybe', 'good', 'feel')  
        5. Finally output a 150-300 word summary paragraph, no title needed  

        The summaries for each phase are as follows:\n\n"""

        for title, content in section_summaries.items():
            if content:
                prompt += f"\n[{title}]\n{content}\n"

        # Call the large model for summary
        overall_summary = self.call(prompt)

        return overall_summary


    def update_toc_with_relevant_sections(self, toc, agent_abstracts):
        """
        Based on the toc and the summaries of each module, generate a list of module numbers that each chapter should reference,
        and add the result as the fifth item.
        """
        prompt = f"""
        You are a professional data analysis report planning assistant.
        I will provide the report table of contents and the summaries of each analysis module. Please determine the list of module numbers that each chapter should reference.

        Report table of contents (each element is a quadruple: title, level, content outline, list of figure numbers):
        {toc}

        Summaries of each data analysis module are as follows:
        {agent_abstracts}
        Please base on:
        1. The title, level, and content outline of each chapter;
        2. The summaries of each data processing section;
        3. The assignment of figure numbers for each chapter (the fourth item in the report table of contents);

        Reasonably judge which data processing sections' information each chapter should reference when generating the report.
        Output requirements:

        - For each chapter, generate a quintuple (title, level, content outline, list of figure numbers, list of module numbers)
            - Title, level, content outline, list of figure numbers must not be changed; only add the fifth item based on the original
        - The list of module numbers is a Python list, e.g., [0, 2]
        - If no module needs to be referenced, return []
        - Output as a Python list, without any additional explanation
        Example:
        Input:
        [
          ('Overview',1,'Introduce the report background and objectives',[1]),
          ('Data Visualization',1,'Analyze visualization charts of air quality and related environmental variables',[2,3]),
          ('xxx Correlation Analysis',2,'Analyze the relationship between relative humidity and other pollutants',[4,5])
        ]
        Output:
        [
          ('Overview',1,'Introduce the report background and objectives',[1],[1,2]),
          ('Data Visualization',1,'Analyze visualization charts of air quality and related environmental variables',[2,3],[0,1]),
          ('xxx Correlation Analysis',2,'Analyze the relationship between relative humidity and other pollutants',[4,5],[2,3])
        ]
        """
        toc_with_sections = self.call(prompt)
        print(toc_with_sections)
        return toc_with_sections.strip()


    def write_section_body(self, toc, t, selected_full_contents, history_content):

        prompt = f"""
        You are a professional data analysis report writing assistant. Your task is to generate logically clear, structurally rigorous, and professionally content-rich report chapters based on the reference information I provide.

        Current chapter information (quadruple: title, level, content outline, list of figure numbers):
        {t}

        Report table of contents structure (contains quadruple information for all chapters):
        {toc}

        Reference analysis content is as follows:
        {selected_full_contents}

        Previously generated chapter content (used to maintain overall style consistency and avoid repetition):
        {history_content}

        Writing requirements:

        1. Writing Objectives
        1. Write only the main text content of the current chapter "{t[0]}";
        2. The content must be based on the "reference information" as the core basis, and can **appropriately expand and summarize** within its logical framework;
        3. Reasonable professional supplements are allowed (such as statistical explanations, method principles, result meanings), but **it is forbidden to fabricate specific data, chart results, experimental scenarios, or sample features**;
        4. If the reference information is insufficient, general analysis ideas can be supplemented, but the content must remain universal, objective, and abstract, and cannot be concretized into hypothetical data.

        2. Language and Structure
        1. The writing style should be formal, professional, and academic;
        2. The discussion should follow data analysis logic: first describe, then explain, and finally summarize;
        3. Each natural paragraph should revolve around one logical core (such as trends, comparisons, correlations, distribution characteristics, etc.).

        3. Chart Usage Specifications
        1. Only the figure numbers of this chapter {t[3]} can be used in the main text;
        2. Use placeholders [FIG:index] to mark the chart positions;
        3. Add a image title below each placeholder:
            Figure: Image title (briefly explain the image content and analysis points)
        4. The image position should naturally connect with the semantics:
        - If the image introduces analysis → place it at the beginning of the paragraph;
        - If the image supports an argument → place it after the relevant descriptive sentence;
        - If the image summarizes results → place it at the end of the paragraph;
        5. Do not add, delete, or rearrange figure numbers.

        4. Output Requirements
        - Output only the main text content;
        - Do not output titles, numbers, explanatory text, or Markdown;
        - Do not use bold, italics, symbolic decorations, or non-main text statements;
        - Do not use subjective expressions such as "I think", "please continue", "in summary, it can be seen", etc.

        5. Writing Mode
        Current mode: {self.outline_length}
        - Brief: Write only conclusions;
        - Standard: Include logic and conclusions;
        - Detailed: Include reasoning and methods, but still based on reference information, and do not freely create.

        Please strictly write the main text of this chapter within the above scope.
        """
        
        if st.session_state.preference_select:
            prompt += f"The user's analysis preferences are as follows: {st.session_state.preference_select}.\n\n"
        if st.session_state.additional_preference:
            prompt += f"The user has provided the following modeling goals and special requirements: {st.session_state.additional_preference}.\n\n"


        content = self.call(prompt)

        return content