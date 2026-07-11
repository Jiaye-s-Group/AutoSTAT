"""
Local Python helpers used by AutoSTAT workflows.

The functions in this module replace small hosted workflow/plugin nodes with
plain Python implementations. They focus on data loading, code execution,
visualization extraction, and summary composition.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from typing import Any

import pandas as pd

from core.workflow_runner import to_json_str, to_str


# Summary composers.


def summary1_composer(*, desc: str, head_dict_str: str) -> dict[str, Any]:
    """Compose the loading-stage summary."""
    return {
        "summary_1": {
            "title": "数据概览与数据含义分析",
            "desc": to_str(desc),
            "df": to_str(head_dict_str),
        }
    }


def summary2_composer(
    *, code: str, desc: str, processed_df: str
) -> dict[str, Any]:
    """Compose the preprocessing-stage summary."""
    return {
        "summary_2": {
            "title": "数据预处理",
            "desc": to_str(desc),
            "processed_df": to_str(processed_df),
            "code": to_str(code),
        }
    }


def sec3_composer(*, fig_analysis: list) -> dict[str, Any]:
    """Compose the visualization-stage summary."""
    fa_out = []
    for item in fig_analysis or []:
        if isinstance(item, dict):
            fa_out.append(
                {
                    "fig": to_str(item.get("fig", "")),
                    "fig_artifact": to_str(item.get("fig_artifact", "")),
                    "desc": to_str(item.get("desc", "")),
                    "title": to_str(item.get("title", "")),
                    "analysis": to_str(item.get("analysis", "")),
                }
            )
    return {
        "summary_3": {
            "title": "数据可视化",
            "fig_analysis": fa_out,
        }
    }


def sec4_composer(
    *,
    code: str,
    desc: str,
    result: str,
    table_title: str = "",
    table_markdown: str = "",
    table_html: str = "",
) -> dict[str, Any]:
    """Compose the modeling-stage summary."""
    return {
        "summary_4": {
            "title": "建模分析",
            "desc": to_str(desc),
            "result": to_str(result),
            "code": to_str(code),
            "table_title": to_str(table_title),
            "table_markdown": to_str(table_markdown),
            "table_html": to_str(table_html),
        }
    }


def history_content_composer(
    *, content: str, history_content: str = ""
) -> dict[str, Any]:
    """Append one report section to the history context."""
    hc = to_str(history_content)
    c = to_str(content)
    if hc and c:
        return {"history_content": hc + "\n\n" + c}
    return {"history_content": hc + c}


# List helpers.


def final_list(
    *,
    processed_df_head_list: list,
    processed_df_list: list,
) -> dict[str, Any]:
    """Return the latest preprocessing output from a retry list."""
    head = ""
    if isinstance(processed_df_head_list, list) and processed_df_head_list:
        head = to_str(processed_df_head_list[-1])

    df = ""
    if isinstance(processed_df_list, list) and processed_df_list:
        df = to_str(processed_df_list[-1])

    return {
        "processed_df_head": head,
        "processed_df": df,
    }


def sec3_check_full(*, analysis_list: list) -> dict[str, Any]:
    """Join per-figure analysis items into one report context string."""
    if not isinstance(analysis_list, list):
        return {"full": ""}
    parts = []
    fig_pattern = re.compile(
        r"(?<![A-Za-z0-9_])[\[\uFF3B\u3010]?\s*FIG\s*[:\uFF1A]?\s*\d+\s*[\]\uFF3D\u3011]?(?![A-Za-z0-9_])",
        flags=re.IGNORECASE,
    )
    for index, item in enumerate(analysis_list):
        if isinstance(item, dict):
            text = to_str(item.get("analysis", ""))
        elif isinstance(item, str):
            text = item
        else:
            text = ""
        text = text.strip()
        if text and not fig_pattern.search(text):
            text = f"[FIG:{index}] {text}"
        parts.append(text)
    full = "\n\n".join(p for p in parts if p)
    return {"full": full}


# Data loading and metadata.


def loading_data(*, file_url: str) -> dict[str, Any]:
    """
    planning/128297
    The open-source workflow receives DataFrames directly, so remote URL loading
    is intentionally not used. The signature remains as a compatibility guard.
    """
    return {
        "is_success": False,
        "error": "loading_data is not used by the local workflow; construct metadata from a DataFrame instead.",
        "shape_0": 0,
        "shape_1": 0,
        "dtype_info_str": "",
        "head_dict_str": "",
        "df": "",
    }


def df_to_meta(df: pd.DataFrame) -> dict[str, Any]:
    """
    Convert a pandas DataFrame into metadata consumed by downstream workflows.
    """
    if df is None or len(df) == 0:
        return {
            "is_success": False,
            "error": "空 DataFrame",
            "shape_0": 0,
            "shape_1": 0,
            "dtype_info_str": "{}",
            "head_dict_str": "[]",
            "df": "",
        }
    return {
        "is_success": True,
        "error": "",
        "shape_0": int(df.shape[0]),
        "shape_1": int(df.shape[1]),
        "dtype_info_str": df.dtypes.astype(str).to_json(),
        "head_dict_str": df.head(5).to_json(orient="records", force_ascii=False),
        "df": df.to_json(orient="records", force_ascii=False),
    }


# Preprocessing metadata.


def get_preprocessing_suggestions(*, df: str) -> dict[str, Any]:
    """Build deterministic dataset facts for preprocessing prompts."""
    try:
        records = json.loads(df) if isinstance(df, str) else df
        if not isinstance(records, list):
            raise ValueError("df 必须是 records list")
        dataframe = pd.DataFrame(records)
    except Exception as exc:
        return {
            "columns": [],
            "dtype_counts": "",
            "n_rows": "0",
            "num_cols": "",
            "is_success": False,
            "missing_by_col": "",
            "missing_total": "0",
            "n_cols": "0",
        }

    cols = list(dataframe.columns)
    dtype_counts = dataframe.dtypes.astype(str).value_counts().to_dict()
    num_cols = list(dataframe.select_dtypes(include=["number"]).columns)
    missing_by_col = dataframe.isnull().sum().to_dict()

    return {
        "columns": cols,
        "dtype_counts": json.dumps(dtype_counts, ensure_ascii=False),
        "n_rows": str(dataframe.shape[0]),
        "num_cols": json.dumps(num_cols, ensure_ascii=False),
        "is_success": True,
        "missing_by_col": json.dumps(missing_by_col, ensure_ascii=False),
        "missing_total": str(int(dataframe.isnull().sum().sum())),
        "n_cols": str(dataframe.shape[1]),
    }


# Preprocessing code execution.


_CODE_RUNNER_TEMPLATE = '''import json, sys, traceback
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.preprocessing import OneHotEncoder as _SklearnOneHotEncoder

if not hasattr(pd.DataFrame, "concat"):
    pd.DataFrame.concat = staticmethod(pd.concat)

def pd_isna_like(value):
    return pd.isna(value)

def OneHotEncoder(*args, **kwargs):
    if "sparse" in kwargs and "sparse_output" not in kwargs:
        kwargs["sparse_output"] = kwargs.pop("sparse")
    try:
        return _SklearnOneHotEncoder(*args, **kwargs)
    except TypeError:
        if "sparse_output" in kwargs:
            kwargs["sparse"] = kwargs.pop("sparse_output")
        return _SklearnOneHotEncoder(*args, **kwargs)

_RECORDS = json.loads(sys.stdin.read())
df = pd.DataFrame(_RECORDS)

try:
__USER_CODE__
except Exception as e:
    print("__AUTOSTAT_ERROR__", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(2)

# Generated preprocessing code must assign a pandas DataFrame to process_df.
_out = locals().get("process_df")
if not isinstance(_out, pd.DataFrame):
    print("__AUTOSTAT_ERROR__", file=sys.stderr)
    print("代码必须定义 pandas.DataFrame 类型的 process_df", file=sys.stderr)
    sys.exit(3)

print(json.dumps({
    "processed_df": _out.to_json(orient="records", force_ascii=False),
    "processed_df_head": _out.head(5).to_json(orient="records", force_ascii=False),
}, ensure_ascii=False))
'''


def code_runner(
    *,
    code: str,
    df: str,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Run generated preprocessing code in a subprocess with a timeout."""
    user_code = to_str(code).strip()
    if not user_code:
        return {
            "processed_df": "",
            "processed_df_head": "",
            "error": "空代码",
            "is_success": False,
        }

    # Indent generated code into the runner's try block.
    indented = textwrap.indent(user_code, " " * 4)
    script = _CODE_RUNNER_TEMPLATE.replace("__USER_CODE__", indented)

    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            input=to_str(df) or "[]",
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "processed_df": "",
            "processed_df_head": "",
            "error": f"代码执行超时（>{timeout_seconds}s）",
            "is_success": False,
        }
    except Exception as exc:
        return {
            "processed_df": "",
            "processed_df_head": "",
            "error": f"子进程启动失败：{exc}",
            "is_success": False,
        }

    if completed.returncode != 0:
        err = completed.stderr or "未知错误"
        return {
            "processed_df": "",
            "processed_df_head": "",
            "error": err.strip()[:1500],
            "is_success": False,
        }

    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "processed_df": "",
            "processed_df_head": "",
            "error": f"输出解析失败：{completed.stdout[:500]}",
            "is_success": False,
        }

    return {
        "processed_df": result.get("processed_df", ""),
        "processed_df_head": result.get("processed_df_head", ""),
        "error": "",
        "is_success": True,
    }


# Visualization code execution.


_VIZ_RUNNER_TEMPLATE = '''import json, sys, traceback
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

_RECORDS = json.loads(sys.stdin.read())
df = pd.DataFrame(_RECORDS)

fig_dict = {}
try:
__USER_CODE__
except Exception as e:
    print("__AUTOSTAT_ERROR__", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(2)

_figs = {}
if isinstance(locals().get("fig_dict"), dict):
    for k, v in fig_dict.items():
        try:
            if isinstance(v, go.Figure):
                _figs[str(k)] = v.to_json()
            elif hasattr(v, "to_plotly_json") and v.__class__.__module__.startswith("plotly"):
                _figs[str(k)] = go.Figure(v).to_json()
        except Exception:
            pass

print(json.dumps(_figs, ensure_ascii=False))
'''


def execute_and_extract(
    *,
    code: str,
    df_data: str,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Run generated Plotly code and collect figures from `fig_dict`."""
    user_code = to_str(code).strip()
    if not user_code:
        return {"fig_task_list": []}

    indented = textwrap.indent(user_code, " " * 4)
    script = _VIZ_RUNNER_TEMPLATE.replace("__USER_CODE__", indented)

    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            input=to_str(df_data) or "[]",
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {"fig_task_list": [], "error": f"超时（>{timeout_seconds}s）"}

    if completed.returncode != 0:
        return {"fig_task_list": [], "error": (completed.stderr or "")[:1500]}

    try:
        figs = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        return {"fig_task_list": [], "error": "输出解析失败"}

    fig_task_list = [{"title": k, "fig": v} for k, v in figs.items()]
    if not fig_task_list:
        return {
            "fig_task_list": [],
            "error": (
                "No Plotly figures were collected from fig_dict. "
                "The visualization code must store every generated Plotly Figure "
                "as fig_dict['descriptive_key'] = fig and must not rely on fig.show(), "
                "temporary variables, figures lists, charts dictionaries, or other output variables."
            ),
        }
    return {"fig_task_list": fig_task_list}


def validate_viz_code(*, code: str, df_data: str) -> dict[str, Any]:
    """Smoke-test visualization code on a small DataFrame sample."""
    result = execute_and_extract(code=code, df_data=df_data, timeout_seconds=30)
    err = result.get("error", "")
    if err:
        return {
            "error_msg": err,
            "final_code": "",
            "is_success": False,
        }
    return {
        "error_msg": "",
        "final_code": to_str(code),
        "is_success": True,
    }


# Visualization prompt builders.


def summary_fig_list_prompt(
    *,
    cols_wo_id: list,
    item: dict,
    selected_model: str = "GPT-4o",
) -> dict[str, Any]:
    """Build a figure-analysis prompt for one visualization item."""
    cols_text = ", ".join([str(c) for c in (cols_wo_id or [])])
    fig_artifact = ""
    if isinstance(item, dict):
        artifact_value = item.get("fig_artifact_text", "") or item.get("fig_artifact", "")
        fig_artifact = (
            artifact_value
            if isinstance(artifact_value, str)
            else to_json_str(artifact_value)
        )
    if not fig_artifact and isinstance(item, dict):
        fig_artifact = to_str(item.get("fig", ""))[:2000]
    desc = to_str(item.get("desc", "")) if isinstance(item, dict) else ""

    # Only enable multimodal prompts for models known to support images.
    is_multimodal = selected_model.lower() in {"gpt-4o", "gpt-4-vision", "claude-3-opus"}

    prompt = (
        f"你是一名数据可视化分析师。请基于以下信息对图表做一段中文分析（80~200字）：\n"
        f"- 图表标题/说明: {desc}\n"
        f"- 数据列: {cols_text}\n"
        f"- 图表压缩证据: {fig_artifact[:5000]}\n\n"
        f"请从「图表想表达什么」「数据呈现出什么模式」「业务含义」三个角度简洁说明。"
    )

    return {
        "is_multimodal": is_multimodal,
        "prompt": prompt,
    }


def desc_fig_prompt(
    *,
    dtype_info: str,
    fig: str,
    fig_artifact: Any = "",
    selected_model: str = "GPT-4o",
) -> dict[str, Any]:
    """Build a short figure-description prompt."""
    artifact_text = fig_artifact if isinstance(fig_artifact, str) else to_json_str(fig_artifact)
    artifact_text = artifact_text or to_str(fig)[:2000]
    prompt = (
        f"请用一句话简明描述下面这张图所展示的数据分布/模式（50字以内）：\n"
        f"- 数据字段类型：{to_str(dtype_info)[:1000]}\n"
        f"- 图表压缩证据：{artifact_text[:5000]}"
    )
    return {"prompt_content": prompt}


# RAG formatting.


def format_recall(*, output_list: list) -> dict[str, Any]:
    """Format retrieved knowledge for preprocessing and modeling prompts."""
    if not isinstance(output_list, list) or not output_list:
        return {"knowledge_results": "（未召回相关算法）"}

    # Local algorithm-catalog results have their own formatter.
    first = output_list[0]
    if isinstance(first, dict) and ("name" in first or "category_l2" in first):
        from core.rag_retriever import format_recall as _format

        return {"knowledge_results": _format(output_list)}

    # Generic retrieval format: concatenate chunk text.
    parts = []
    for i, item in enumerate(output_list, 1):
        if isinstance(item, dict):
            text = to_str(item.get("output") or item.get("content") or item.get("text"))
            parts.append(f"### 召回 {i}\n{text}")
        elif isinstance(item, str):
            parts.append(f"### 召回 {i}\n{item}")
    return {"knowledge_results": "\n\n".join(parts) or "（未召回相关算法）"}
