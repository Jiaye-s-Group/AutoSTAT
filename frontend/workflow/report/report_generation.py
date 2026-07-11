"""Background-process lifecycle for report generation."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from workflow.report.report_constants import (
    REPORT_GENERATION_JOB_KEY,
    REPORT_GENERATION_PROCESS_KEY,
    REPORT_GENERATION_RUNNING_KEY,
    REPORT_PENDING_PREVIEW_KEY,
)
from workflow.report.report_content_utils import stringify_string
from workflow.report.report_inputs import build_report_worker_payload, report_repo_root
from workflow.report.report_state import (
    begin_report_generation,
    finish_report_generation,
    is_current_report_generation,
    is_report_generation_cancelled,
)


def cleanup_report_job_files(job: dict[str, Any] | None) -> None:
    if not isinstance(job, dict):
        return

    work_dir = job.get("work_dir")
    if not work_dir:
        return

    try:
        work_path = Path(str(work_dir)).resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if temp_root not in (work_path, *work_path.parents):
            print("[REPORT][JOB] skip cleanup outside temp dir:", work_path)
            return
        if not work_path.name.startswith("autostat_report_"):
            print("[REPORT][JOB] skip cleanup for unexpected temp dir:", work_path)
            return
        shutil.rmtree(work_path, ignore_errors=True)
    except Exception as exc:
        print("[REPORT][JOB] cleanup failed:", repr(exc))


def terminate_report_generation_process() -> None:
    job = st.session_state.get(REPORT_GENERATION_JOB_KEY)
    process = st.session_state.get(REPORT_GENERATION_PROCESS_KEY)
    if process is None and isinstance(job, dict):
        process = job.get("process")

    poll = getattr(process, "poll", None)
    if callable(poll):
        try:
            if process.poll() is None:
                print(f"[REPORT][JOB] terminate previous report process pid={getattr(process, 'pid', None)}")
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        except Exception as exc:
            print("[REPORT][JOB] terminate failed:", repr(exc))

    if isinstance(job, dict):
        cleanup_report_job_files(job)
        token = job.get("token")
        if token and is_current_report_generation(token):
            st.session_state[REPORT_GENERATION_RUNNING_KEY] = False

    st.session_state.pop(REPORT_GENERATION_PROCESS_KEY, None)
    st.session_state.pop(REPORT_GENERATION_JOB_KEY, None)


def start_report_generation_process(report_agent, action: str) -> bool:
    terminate_report_generation_process()
    generation_token = begin_report_generation(report_agent)

    work_dir = tempfile.mkdtemp(prefix="autostat_report_")
    input_path = os.path.join(work_dir, "input.json")
    output_path = os.path.join(work_dir, "output.json")
    progress_path = os.path.join(work_dir, "progress.json")
    payload = build_report_worker_payload(report_agent)

    try:
        with open(input_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

        env = os.environ.copy()
        llm_config = payload.get("llm_config") if isinstance(payload.get("llm_config"), dict) else {}
        if llm_config.get("api_key"):
            env["OPENAI_API_KEY"] = str(llm_config["api_key"])
        if llm_config.get("base_url"):
            env["OPENAI_BASE_URL"] = str(llm_config["base_url"])
        if llm_config.get("model"):
            env["OPENAI_MODEL"] = str(llm_config["model"])

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "workflows.reporting_partly_worker",
                input_path,
                output_path,
                progress_path,
            ],
            cwd=str(report_repo_root()),
            env=env,
        )
    except Exception as exc:
        cleanup_report_job_files({"work_dir": work_dir})
        finish_report_generation(generation_token)
        st.error(f"报告生成进程启动失败：{exc}")
        return False

    job = {
        "token": generation_token,
        "action": action,
        "work_dir": work_dir,
        "input_path": input_path,
        "output_path": output_path,
        "progress_path": progress_path,
        "pid": process.pid,
        "started_at": time.time(),
        "process": process,
    }
    st.session_state[REPORT_GENERATION_PROCESS_KEY] = process
    st.session_state[REPORT_GENERATION_JOB_KEY] = job
    st.session_state[REPORT_GENERATION_RUNNING_KEY] = True
    print(f"[REPORT][JOB] started report process pid={process.pid}")
    return True


def read_report_worker_output(job: dict[str, Any]) -> dict[str, Any] | None:
    output_path = job.get("output_path")
    if not output_path or not os.path.exists(str(output_path)):
        return None
    try:
        with open(str(output_path), "r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        return {"ok": False, "error": f"报告生成结果读取失败：{exc}"}


def read_report_worker_progress(job: dict[str, Any]) -> dict[str, Any] | None:
    progress_path = job.get("progress_path")
    if not progress_path or not os.path.exists(str(progress_path)):
        return None

    try:
        with open(str(progress_path), "r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None
    except Exception as exc:
        print("[REPORT][PROGRESS] read failed:", repr(exc))
        return None


def format_report_progress_status(action: str, progress: dict[str, Any] | None) -> str:
    if not isinstance(progress, dict):
        return f"正在生成{action}报告，请耐心等待"

    phase = stringify_string(progress.get("phase")).strip()
    section_title = _format_progress_section_title(progress)
    total_sections = _safe_progress_int(progress, "total_sections")
    section_index = _safe_progress_int(progress, "section_index")
    completed_sections = _safe_progress_int(progress, "completed_sections")

    if progress.get("status") == "finalizing" or phase == "body_completed":
        return f"报告正文已完成，正在整理标题并准备生成{action}文件。"
    if phase == "section_draft" and section_title:
        return f"“{section_title}”草稿已生成，正在整理为正式段落。（{section_index}/{total_sections}）"
    if phase == "section_completed" and section_title:
        return f"“{section_title}”已完成，继续生成后续内容。（已完成 {completed_sections}/{total_sections} 节）"
    if phase == "section_started" and section_title:
        return f"正在撰写“{section_title}”。（{section_index}/{total_sections}）"
    if total_sections:
        return f"正在生成{action}报告正文。（已完成 {completed_sections}/{total_sections} 节）"
    return f"正在生成{action}报告，请耐心等待"


def remember_live_report_progress(progress: dict[str, Any] | None) -> None:
    if not isinstance(progress, dict):
        return

    markdown_content = stringify_string(progress.get("markdown")).strip()
    existing_preview = st.session_state.get(REPORT_PENDING_PREVIEW_KEY)
    preview = dict(existing_preview) if isinstance(existing_preview, dict) else {}
    if markdown_content:
        preview = {
            "html": "",
            "markdown": markdown_content,
            "progress": progress,
            "live": True,
        }
    else:
        preview["progress"] = progress
    st.session_state[REPORT_PENDING_PREVIEW_KEY] = preview


def poll_report_generation_job(
    report_agent,
    action: str,
    *,
    merge_report_workflow_results: Callable[[list[Any]], dict[str, Any] | None],
    save_formatted_report_result: Callable[[Any, str, dict[str, Any], str | None], bool],
) -> str:
    job = st.session_state.get(REPORT_GENERATION_JOB_KEY)
    if not isinstance(job, dict):
        return "idle"

    process = st.session_state.get(REPORT_GENERATION_PROCESS_KEY) or job.get("process")
    poll = getattr(process, "poll", None)
    token = job.get("token")
    if not token or not is_current_report_generation(token):
        terminate_report_generation_process()
        return "idle"

    if not callable(poll):
        cleanup_report_job_files(job)
        st.session_state.pop(REPORT_GENERATION_PROCESS_KEY, None)
        st.session_state.pop(REPORT_GENERATION_JOB_KEY, None)
        finish_report_generation(token)
        st.error("报告生成进程状态丢失，请重新生成。")
        return "failed"

    return_code = process.poll()
    if return_code is None:
        progress = read_report_worker_progress(job)
        remember_live_report_progress(progress)
        st.info(format_report_progress_status(action, progress))
        st.session_state[REPORT_GENERATION_RUNNING_KEY] = True
        return "running"

    worker_payload = read_report_worker_output(job)
    cleanup_report_job_files(job)
    st.session_state.pop(REPORT_GENERATION_PROCESS_KEY, None)
    st.session_state.pop(REPORT_GENERATION_JOB_KEY, None)

    if is_report_generation_cancelled(token):
        return "idle"

    if not worker_payload:
        finish_report_generation(token)
        st.error(f"报告生成进程已退出（code={return_code}），但没有返回可用结果。")
        return "failed"

    if not worker_payload.get("ok"):
        finish_report_generation(token)
        error_message = worker_payload.get("error") or "未知错误"
        st.error(f"报告生成失败：{error_message}")
        traceback_text = stringify_string(worker_payload.get("traceback"))
        if traceback_text:
            print("[REPORT][JOB] worker traceback:\n", traceback_text)
        return "failed"

    workflow_result = merge_report_workflow_results([worker_payload.get("result")])
    if workflow_result is None:
        finish_report_generation(token)
        st.error("Word 报告生成失败，未解析到有效输出，请重新生成。")
        return "failed"

    success = save_formatted_report_result(report_agent, action, workflow_result, token)
    finish_report_generation(token)
    if success:
        st.success(f"{action} 报告已生成，已在下方展示。")
        return "complete"
    return "failed"


def is_report_generation_job_running() -> bool:
    job = st.session_state.get(REPORT_GENERATION_JOB_KEY)
    process = st.session_state.get(REPORT_GENERATION_PROCESS_KEY)
    poll = getattr(process, "poll", None)
    return isinstance(job, dict) and callable(poll) and process.poll() is None


def _format_progress_section_title(progress: dict[str, Any]) -> str:
    section_title = stringify_string(progress.get("section_title")).strip()
    section_num = stringify_string(progress.get("section_num")).strip()
    if section_num:
        section_title = re.sub(
            rf"^\s*{re.escape(section_num)}(?:[\.．、]|\s+)?\s*",
            "",
            section_title,
        ).strip()
    return section_title


def _safe_progress_int(progress: dict[str, Any], key: str) -> int:
    try:
        return int(progress.get(key) or 0)
    except Exception:
        return 0
