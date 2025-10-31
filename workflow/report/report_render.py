import datetime
import io
from io import BytesIO
import os

from stqdm import stqdm
import mammoth
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompt_engineer.sec5_call_llm import *
from workflow.report.report_utils import html_dowmload
from workflow.report.report_html import write_html
from workflow.report.report_word import write_word
from workflow.report.report_markdown import write_markdown
from workflow.report.report_prepare_er import report_prepare


def report_save(agents, auto): 

    report_agent = agents[-1]
    action = report_agent.load_report_format()
    
    if report_agent.load_report_format() == 'HTML':
        not_generate = report_agent.html == None
    if report_agent.load_report_format() == 'Word':
        not_generate = report_agent.word == None
    if report_agent.load_report_format() == 'Markdown':
        not_generate = report_agent.markdown == None

    mode = report_agent.load_gen_mode()
    parallel = (mode == "Parallel")
    
    if st.button(f"📝 Generate {action} Report") or (auto and not_generate):
        with st.spinner(f"Generating {action} Report..."):

            report_prepare(agents, parallel=parallel)

            if report_agent.load_report_format() == 'Word':
                write_word(agents)
            elif report_agent.load_report_format() == 'HTML':
                write_html(agents)
            elif report_agent.load_report_format() == 'Markdown':
                write_markdown(agents)


def report_basic_info(agent, auto) -> None:
    outline_length = sac.segmented(
        items=[
            sac.SegmentedItem(label='Brief'),
            sac.SegmentedItem(label='Standard'),
            sac.SegmentedItem(label='Detailed'),
        ],
        label='Detail Level', index=1, align='center',
        size='sm', radius='sm', use_container_width=True
    )
    agent.save_outline_length(outline_length)

    c1, c2 = st.columns([3, 1])
    with c1:
        report_format = sac.chip(
            items=[
                sac.ChipItem(label='Word', icon=sac.BsIcon(name='file-earmark-word', size=16)),
                sac.ChipItem(label='HTML', icon=sac.BsIcon(name='filetype-html', size=16)),
                sac.ChipItem(label='Markdown', icon=sac.BsIcon(name='file-earmark-code', size=16)),
            ],
            label='Select Report Format', index=[0, 2],
            align='start', radius='md', multiple=False,
        )
        agent.save_report_format(report_format)

    with c2:
        mode = sac.segmented(
            items=[
                sac.SegmentedItem(label='Parallel'),
                sac.SegmentedItem(label='Serial'),
            ],
            label='Generation Mode', align='end', size='sm',
            use_container_width=True, radius='md'
        )
        agent.save_gen_mode(mode)

    user_input = st.text_input("Report Generation Requirements", "Default")
    agent.save_user_input(user_input)

    not_generated = report_agent.load_outline() is None

    # === Generate table of contents in parallel ===
    if st.button("🗒️ Generate Table of Contents") or (auto and not_generated):
        with st.spinner("⏳ Automatically generating table of contents..."):
            summaries = []

            # === Save current Streamlit state snapshot ===
            session_snapshot = dict(st.session_state)

            def process_summary(idx, sub_agent, session_snapshot):
                """Execute summary_html/summary_word in parallel (with state replication)"""
                # Restore session_state
                for k, v in session_snapshot.items():
                    st.session_state[k] = v

                # Actual generation logic
                if hasattr(sub_agent, "summary_html"):
                    summary = sub_agent.summary_html()
                else:
                    summary = sub_agent.summary_word()

                return idx, summary

            max_workers = min(6, len(agents) - 1)
            results = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_summary, i, sub_agent, session_snapshot): i
                    for i, sub_agent in enumerate(agents[:-1])
                }

                for future in stqdm(as_completed(futures), total=len(futures)):
                    try:
                        idx, summary = future.result()
                        if summary:
                            results.append((idx, summary))
                    except Exception as e:
                        print(f"Submodule summary generation failed: {e}")

            # === Restore original section order ===
            results.sort(key=lambda x: x[0])
            summaries = [summary for _, summary in results if summary]

            # === Generate table of contents ===
            default_toc = report_agent.generate_toc_from_summary(summaries)
            report_agent.save_outline(default_toc)


def report_outline(agents):

    st.subheader("Table of Contents Preview and Editing")
    load_agent, preproc_agent, visual_agent, coding_agent, report_agent = agents[0], agents[1], agents[2], agents[3], agents[4]

    default_toc = report_agent.load_outline()

    toc_md = st.text_area(
        "You can edit the table of contents structure here", 
        value=default_toc, 
        height=260
    )

    report_agent.save_outline(toc_md)


def report_execution(report_agent):

    if report_agent.load_report_format() == 'Word':

        full_report = report_agent.load_word()
        if full_report is not None:
            st.download_button(
                label="⬇️ Download Word Report",
                data=full_report,
                file_name="report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

    elif report_agent.load_report_format() == 'HTML':

        full_report = report_agent.load_html()

        if full_report is not None:
            st.download_button(
                label="⬇️ Download HTML Report",
                data=full_report.encode("utf-8"),
                file_name="report.html",
                mime="text/html",
            )

            if st.button("⬇️ Download PDF Report"):
                html_dowmload(full_report)
    elif report_agent.load_report_format() == 'Markdown':

        full_report = report_agent.load_markdown()
        if full_report is not None:

            # Provide download button
            st.download_button(
                label="⬇️ Download Markdown Report",
                data=full_report,
                file_name="report.md",
                mime="text/markdown"
            )


if __name__ == "__main__":

    st.title("Report Generation")

    st.markdown("---")

    load_agent   = st.session_state.data_loading_agent
    preproc_agent = st.session_state.data_preprocess_agent
    visual_agent = st.session_state.visualization_agent
    coding_agent = st.session_state.modeling_coding_agent
    planner = st.session_state.planner_agent
    auto = planner.report_auto

    processed_df = preproc_agent.load_processed_df()
    if processed_df is None:
        df = load_agent.load_df()
    else:
        df = processed_df

    if df is None:
        st.warning("⚠️ Please load data on the data import page first")
        st.stop()

    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    report_agent = st.session_state.report_agent
    report_agent.add_df(df_shuffled)

    agents = [load_agent, preproc_agent, visual_agent, coding_agent, report_agent]

    c = st.columns(2)
    with c[0].expander('Report Settings', True):
        report_basic_info(report_agent, auto)

    with c[1].expander('Report Outline', True):
        report_outline(agents)
        report_save(agents, auto)
        report_execution(report_agent)