"""Bounded, process-isolated execution for code launched from Streamlit pages.

The workflow validators already run generated code in subprocesses.  The page
buttons used to call ``safe_exec`` in the Streamlit process itself, which meant
a costly but syntactically safe operation could keep the whole application
busy indefinitely.  This module keeps the same execution namespaces and full
in-memory DataFrame semantics while making each manual run terminable.
"""

from __future__ import annotations

import contextlib
import io
import pickle
import random
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.generated_code_context import (
    ExecutionKind,
    build_execution_namespace,
    execution_output_name,
)
from core.safe_code import UnsafeCodeError, safe_subprocess_env, validate_code


PREPROCESSING_TIMEOUT_SECONDS = 60
VISUALIZATION_TIMEOUT_SECONDS = 120
MODELING_TIMEOUT_SECONDS = 300
INFERENCE_TIMEOUT_SECONDS = 120
MODELING_TRANSPORT_MAX_RECORDS = 1_000
MODELING_TRANSPORT_PREVIEW_ROWS = 20


def _looks_like_record_list(value: list[Any]) -> bool:
    if not value:
        return False
    sample = value[: min(len(value), 5)]
    return all(isinstance(item, dict) for item in sample)


def _clean_for_modeling_transport(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return str(type(value).__name__)
    if isinstance(value, pd.DataFrame):
        return {
            "row_count": int(len(value.index)),
            "columns": [str(column) for column in value.columns],
            "preview": value.head(MODELING_TRANSPORT_PREVIEW_ROWS).to_dict(orient="records"),
            "transport_note": "DataFrame was summarized before returning from the execution worker.",
        }
    if isinstance(value, pd.Series):
        return {
            "row_count": int(len(value.index)),
            "name": str(value.name or ""),
            "preview": value.head(MODELING_TRANSPORT_PREVIEW_ROWS).tolist(),
            "transport_note": "Series was summarized before returning from the execution worker.",
        }
    if isinstance(value, dict):
        return {
            str(key): _clean_for_modeling_transport(child, depth=depth + 1)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) > MODELING_TRANSPORT_MAX_RECORDS and _looks_like_record_list(values):
            return {
                "row_count": len(values),
                "preview": [
                    _clean_for_modeling_transport(child, depth=depth + 1)
                    for child in values[:MODELING_TRANSPORT_PREVIEW_ROWS]
                ],
                "omitted_count": len(values) - MODELING_TRANSPORT_PREVIEW_ROWS,
                "transport_note": "Large record list was summarized before returning from the execution worker.",
            }
        return [_clean_for_modeling_transport(child, depth=depth + 1) for child in values]
    return value

def _emit_worker_payload(payload: dict[str, Any]) -> None:
    sys.stdout.buffer.write(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    sys.stdout.buffer.flush()


def _worker_main() -> None:
    """Execute an internally supplied payload and emit exactly one pickle result."""
    try:
        request = pickle.loads(sys.stdin.buffer.read())
        kind = str(request["kind"])
        output_name = execution_output_name(kind)
        code = str(request["code"])
        dataframe = request["dataframe"]
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("The execution dataframe must be a pandas.DataFrame.")
        extra_values = request.get("extra_values") or {}
        if not isinstance(extra_values, dict):
            raise TypeError("extra_values must be a dictionary.")
        random_state = request.get("random_state") or {}

        # All user print/warning output is captured so stdout remains a reliable
        # binary transport channel.  The captured text is returned for diagnosis.
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        namespace: dict[str, Any] = {}
        execution_error = ""
        try:
            with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
                from core.safe_code import safe_exec

                namespace = build_execution_namespace(
                    kind=kind,
                    dataframe=dataframe,
                    code=code,
                    extra_values=extra_values,
                )
                if isinstance(random_state, dict):
                    if random_state.get("numpy") is not None:
                        np.random.set_state(random_state["numpy"])
                    if random_state.get("python") is not None:
                        random.setstate(random_state["python"])
                safe_exec(code, namespace)
        except BaseException:
            execution_error = traceback.format_exc()

        partial_figures = namespace.get("fig_dict")
        partial_solo_figure = namespace.get("fig")
        has_partial_visualization = (
            isinstance(partial_figures, dict)
            and bool(partial_figures)
        ) or partial_solo_figure is not None
        if execution_error and not (kind == "visualization" and has_partial_visualization):
            raise RuntimeError(execution_error)
        warnings: list[Any] = []
        if execution_error:
            warnings.append({"scope": "script", "error": execution_error[-2000:]})
        if kind == "visualization" and isinstance(namespace.get("figure_errors"), list):
            warnings.extend(
                item for item in namespace["figure_errors"] if isinstance(item, (dict, str))
            )
        output_value = namespace.get(output_name)
        if kind == "visualization" and not isinstance(output_value, dict) and partial_solo_figure is not None:
            output_value = {"default": partial_solo_figure}
        if kind == "modeling":
            output_value = _clean_for_modeling_transport(output_value)
        _emit_worker_payload(
            {
                "is_success": True,
                "value": output_value,
                "metadata": (
                    {"qc_summary": namespace.get("qc_summary")}
                    if kind == "preprocessing" and isinstance(namespace.get("qc_summary"), dict)
                    else {}
                ),
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "warnings": warnings,
                "random_state": {
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                },
            }
        )
    except BaseException:
        _emit_worker_payload(
            {
                "is_success": False,
                "error": traceback.format_exc(),
                "stdout": "",
                "stderr": "",
            }
        )


def run_bounded_safe_exec(
    *,
    kind: ExecutionKind,
    code: str,
    dataframe: pd.DataFrame,
    timeout_seconds: int,
    extra_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run generated code in an isolated, terminable Python process.

    The full dataframe is transported with Python's in-memory pickle protocol,
    rather than JSON records, so column dtypes, indexes, categoricals, and
    timezone-aware values keep the same semantics as the former in-process
    execution.  The payload is created locally and never originates from a
    user-provided file or network source.
    """
    source = str(code or "").strip()
    if not source:
        return {"is_success": False, "error": "Generated code is empty.", "value": None}
    if not isinstance(dataframe, pd.DataFrame):
        return {
            "is_success": False,
            "error": "The execution dataframe must be a pandas.DataFrame.",
            "value": None,
        }
    try:
        execution_output_name(kind)
    except ValueError as exc:
        return {"is_success": False, "error": str(exc), "value": None}

    try:
        validate_code(source)
        request = pickle.dumps(
            {
                "kind": kind,
                "code": source,
                "dataframe": dataframe,
                "extra_values": dict(extra_values or {}),
                "random_state": {
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                },
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    except (UnsafeCodeError, pickle.PickleError, TypeError, ValueError, AttributeError) as exc:
        return {"is_success": False, "error": str(exc), "value": None}

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "core.bounded_code_execution"],
            input=request,
            capture_output=True,
            timeout=max(1, int(timeout_seconds)),
            env=safe_subprocess_env(),
            cwd=str(Path(__file__).resolve().parents[1]),
        )
    except subprocess.TimeoutExpired:
        return {
            "is_success": False,
            "error": f"代码执行超时（>{int(timeout_seconds)}s）；已终止本次执行。",
            "value": None,
        }
    except OSError as exc:
        return {"is_success": False, "error": f"执行子进程启动失败：{exc}", "value": None}

    try:
        response = pickle.loads(completed.stdout)
    except (pickle.PickleError, EOFError, ValueError, TypeError) as exc:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-6000:]
        return {
            "is_success": False,
            "error": f"执行子进程未返回有效结果：{exc}\n{stderr}".strip(),
            "value": None,
        }
    if not isinstance(response, dict):
        return {"is_success": False, "error": "执行子进程返回了无效结果。", "value": None}
    if not response.get("is_success"):
        return {
            "is_success": False,
            "error": str(response.get("error") or "代码执行失败。"),
            "value": None,
        }
    random_state = response.get("random_state")
    if isinstance(random_state, dict):
        try:
            if random_state.get("numpy") is not None:
                np.random.set_state(random_state["numpy"])
            if random_state.get("python") is not None:
                random.setstate(random_state["python"])
        except (TypeError, ValueError):
            # The returned analysis result remains valid even if an optional
            # library changed its random-state representation.
            pass
    return {
        "is_success": True,
        "error": "",
        "value": response.get("value"),
        "metadata": dict(response.get("metadata") or {}),
        "stdout": str(response.get("stdout") or ""),
        "stderr": str(response.get("stderr") or ""),
        "warnings": list(response.get("warnings") or []),
    }


if __name__ == "__main__":
    _worker_main()
