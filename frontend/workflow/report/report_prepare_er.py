import ast

import streamlit as st
from stqdm import stqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.sanitize_code import sanitize_code
from workflow.report.report_core import *


def report_prepare(agents, parallel=True, max_workers=4):
    report_agent = agents[-1]
    toc = report_agent.load_outline()
    if toc is None:
        st.error("请先生成目录")
        return

    # Reuse the generated report while outline and request settings stay unchanged.
    current_length = getattr(report_agent, "load_outline_length", lambda: getattr(report_agent, "outline_length", "Standard"))()
    current_user_input = getattr(report_agent, "load_user_input", lambda: getattr(report_agent, "user_input", "Default"))()
    current_cache_key = (str(toc), str(current_length), str(current_user_input))

    last_cache_key = getattr(report_agent, '_last_gen_cache_key', None)
    existing_report = report_agent.load_report()

    if last_cache_key == current_cache_key and existing_report is not None:
        if hasattr(existing_report, 'root') and existing_report.root.children:
            st.success(f"⚡ 检测到报告要求与大纲未变更，直接复用现有文档结构进行格式转换。")
            return

    toc = sanitize_code(toc)

    # Collect stage abstracts for section-to-stage matching.
    agent_abstracts = {}
    with st.spinner("正在汇总各分析模块的结果..."):
        for i in stqdm(range(len(agents) - 1)):
            agent_abstracts[i] = agents[i].check_abstract()

    # Figure placement is handled by the report-writing workflow.
    toc = sanitize_code(toc)
    try:
        toc = ast.literal_eval(toc)
    except Exception:
        pass

    # Attach relevant analysis stages to each outline item.
    with st.spinner("正在匹配各章节所需的分析模块..."):
        toc_with_choice = report_agent.update_toc_with_relevant_sections(toc, agent_abstracts)
        toc_with_choice = sanitize_code(toc_with_choice)
        try:
            toc_with_choice = ast.literal_eval(toc_with_choice)
        except Exception:
            pass

    doc = Reportcore()
    doc.add_heading('数据分析报告', 0)

    # Keep the selected model available inside worker threads.
    if 'selected_model' in st.session_state:
        selected_model = st.session_state.selected_model
    else:
        selected_model = "default"

    def process_section(idx, t, t_w_c, history_content=""):
        st.session_state.selected_model = selected_model
        _, _, _, _, choice_list = t_w_c
        selected_full_contents = {i: agents[i].check_full() for i in choice_list if i < len(agents) - 1}
        content = report_agent.write_section_body(toc, t, selected_full_contents, history_content)
        return (idx, t, content)

    results = []

    if not parallel:
        with st.spinner("正在串行生成各章节内容（带上下文）..."):
            history_content = ""
            for idx, t in stqdm(enumerate(toc)):
                t_w_c = toc_with_choice[idx]
                _, _, content = process_section(idx, t, t_w_c, history_content)
                results.append((idx, t, content))
                history_content += f"\n\n{t[0]}\n{content}"
    else:
        with st.spinner(f"正在并行生成各章节内容（{max_workers}线程）..."):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_section, idx, t, toc_with_choice[idx], ""): idx
                    for idx, t in enumerate(toc)
                }
                for future in stqdm(as_completed(futures), total=len(futures)):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        print(f"章节生成失败: {e}")

    # Write sections in outline order.
    results.sort(key=lambda x: x[0])
    for _, t, content in results:
        doc.add_heading(t[0], level=t[1])
        doc.add_paragraph(content)

    report_agent.save_report(doc)
    report_agent._last_gen_cache_key = current_cache_key
