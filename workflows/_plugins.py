"""
16 个 Coze 自定义 Plugin 的本地 Python 实现。

Plugin 的 pluginName 分为几组:
- "for AutoSTAT"                     →  数据处理/组装类
- "for AutoSTAT Visualization"       →  可视化相关
- "for AutoSTAT Composer"            →  summary 组装

每个函数的签名严格对应 Coze JSON 里节点的 inputParameters / outputs。
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from core.workflow_runner import to_str, to_json_str, safe_object


# ===================================================================
# 组 1：Composer 类（简单字段组装）
# ===================================================================


def summary1_composer(*, desc: str, head_dict_str: str) -> dict[str, Any]:
    """
    loading/125743
    组装 summary_1 = {title, desc, df}
    """
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
    """
    preprocessing/122472
    组装 summary_2 = {title, desc, processed_df, code}
    """
    return {
        "summary_2": {
            "title": "数据预处理",
            "desc": to_str(desc),
            "processed_df": to_str(processed_df),
            "code": to_str(code),
        }
    }


def sec3_composer(*, fig_analysis: list) -> dict[str, Any]:
    """
    visualizing/145846
    组装 summary_3 = {title, fig_analysis: [{fig, analysis}...]}
    """
    fa_out = []
    for item in fig_analysis or []:
        if isinstance(item, dict):
            fa_out.append(
                {
                    "fig": to_str(item.get("fig", "")),
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
    """
    modeling/148910
    组装 summary_4 = {title, desc, result, code}
    """
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
    """
    reporting_partly/168914
    把每一部分的内容拼到 history_content 末尾。
    """
    hc = to_str(history_content)
    c = to_str(content)
    if hc and c:
        return {"history_content": hc + "\n\n" + c}
    return {"history_content": hc + c}


# ===================================================================
# 组 2：列表处理类
# ===================================================================


def final_list(
    *,
    processed_df_head_list: list,
    processed_df_list: list,
) -> dict[str, Any]:
    """
    preprocessing/165429
    从 Loop 产出的 list 里取最后一个元素（最终版本）。
    """
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
    """
    visualizing/186381
    把 Loop 产出的 analysis 数组拼成一整段 full 文本。
    """
    if not isinstance(analysis_list, list):
        return {"full": ""}
    parts = []
    for item in analysis_list:
        if isinstance(item, dict):
            parts.append(to_str(item.get("analysis", "")))
        elif isinstance(item, str):
            parts.append(item)
    full = "\n\n".join(p for p in parts if p)
    return {"full": full}


# ===================================================================
# 组 3：数据加载
# ===================================================================


def loading_data(*, file_url: str) -> dict[str, Any]:
    """
    planning/128297
    Coze 原版从 URL 下载 CSV，本地化里**不会用到**
    （本地是直接把 DataFrame 传进来的）。
    保留签名作为兼容占位。
    """
    return {
        "is_success": False,
        "error": "loading_data plugin 在本地化版本中不使用 — 请直接从本地 DataFrame 构造 df_meta",
        "shape_0": 0,
        "shape_1": 0,
        "dtype_info_str": "",
        "head_dict_str": "",
        "df": "",
    }


def df_to_meta(df: pd.DataFrame) -> dict[str, Any]:
    """
    本地替代：把 pandas.DataFrame 转成下游需要的元信息字典。
    返回字段和 Coze 的 Loading_Data plugin 对齐。
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


# ===================================================================
# 组 4：预处理统计信息
# ===================================================================


def get_preprocessing_suggestions(*, df: str) -> dict[str, Any]:
    """
    preprocessing/162738
    根据 df 字符串（来自 Loading_Data 的 df 字段，是 records 格式 JSON）
    统计出给 LLM 参考的基本信息。
    """
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


# ===================================================================
# 组 5：代码执行类 —— 最关键的 plugin
# ===================================================================


_CODE_RUNNER_TEMPLATE = '''import json, sys, traceback
import pandas as pd
import numpy as np

_RECORDS = json.loads(sys.stdin.read())
df = pd.DataFrame(_RECORDS)

try:
__USER_CODE__
except Exception as e:
    print("__AUTOSTAT_ERROR__", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(2)

# 约定：用户代码处理后的结果在 process_df / processed_df / df 里任一
_out = None
for _cand in ("process_df", "processed_df", "df"):
    _v = locals().get(_cand)
    if isinstance(_v, pd.DataFrame):
        _out = _v
        break
if _out is None:
    print("__AUTOSTAT_ERROR__", file=sys.stderr)
    print("代码中未定义 process_df / processed_df / df", file=sys.stderr)
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
    """
    preprocessing/138975
    在子进程里执行 LLM 生成的预处理代码。隔离 + 超时。

    用户代码约定（Coze 那边也是同样约定）：
        - 输入变量名 df（pandas.DataFrame）
        - 处理后结果放到 process_df 或 processed_df（或直接复用 df）
    """
    user_code = to_str(code).strip()
    if not user_code:
        return {
            "processed_df": "",
            "processed_df_head": "",
            "error": "空代码",
            "is_success": False,
        }

    # 给用户代码整体加一层缩进（4 空格，对应 try: 块内）
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


# ===================================================================
# 组 6：可视化代码执行
# ===================================================================


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
            _figs[str(k)] = v.to_json()
        except Exception:
            _figs[str(k)] = str(v)
_solo = locals().get("fig")
if _solo is not None and "default" not in _figs:
    try:
        _figs["default"] = _solo.to_json()
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
    """
    visualizing/165403
    在子进程里执行 LLM 生成的 plotly 代码，收集 fig_dict。
    返回 fig_task_list: [{title, fig_json}, ...]
    """
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
    return {"fig_task_list": fig_task_list}


def validate_viz_code(*, code: str, df_data: str) -> dict[str, Any]:
    """
    visualizing/154982
    用前 5 行 df 来试跑代码，验证通过则返回 final_code。
    """
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


# ===================================================================
# 组 7：可视化的 prompt 拼装（给 Batch 节点用）
# ===================================================================


def summary_fig_list_prompt(
    *,
    cols_wo_id: list,
    item: dict,
    selected_model: str = "GPT-4o",
) -> dict[str, Any]:
    """
    visualizing/151708
    给每张图生成描述 prompt。item = {fig, desc, ...}。
    返回 {is_multimodal, prompt}。
    """
    cols_text = ", ".join([str(c) for c in (cols_wo_id or [])])
    fig = to_str(item.get("fig", "")) if isinstance(item, dict) else ""
    desc = to_str(item.get("desc", "")) if isinstance(item, dict) else ""

    # 判断是否多模态：视当前模型是否支持图片。本地默认不使用多模态（节省成本）。
    is_multimodal = selected_model.lower() in {"gpt-4o", "gpt-4-vision", "claude-3-opus"}

    prompt = (
        f"你是一名数据可视化分析师。请基于以下信息对图表做一段中文分析（80~200字）：\n"
        f"- 图表标题/说明: {desc}\n"
        f"- 数据列: {cols_text}\n"
        f"- 图表 JSON: {fig[:2000]}\n\n"
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
    selected_model: str = "GPT-4o",
) -> dict[str, Any]:
    """
    visualizing/1043512
    给每张图生成「描述文字」prompt。
    """
    prompt = (
        f"请用一句话简明描述下面这张图所展示的数据分布/模式（50字以内）：\n"
        f"- 数据字段类型：{to_str(dtype_info)[:1000]}\n"
        f"- 图表 JSON：{to_str(fig)[:2000]}"
    )
    return {"prompt_content": prompt}


# ===================================================================
# 组 8：RAG 格式化
# ===================================================================


def format_recall(*, output_list: list) -> dict[str, Any]:
    """
    preprocessing/155311 + modeling/155311
    把 Knowledge retrieval 节点返回的 outputList 格式化成下游 LLM 能读的文本。

    outputList 结构根据不同的来源可能不同，这里兼容：
    - Coze 原生：[{"output": "...", "score": 0.9}, ...]
    - 本地 retriever 产出：[{"name": "...", "description": "...", "code": "...", ...}]
    """
    if not isinstance(output_list, list) or not output_list:
        return {"knowledge_results": "（未召回相关算法）"}

    # 检查第一条是否是本地格式
    first = output_list[0]
    if isinstance(first, dict) and ("name" in first or "category_l2" in first):
        from core.rag_retriever import format_recall as _format

        return {"knowledge_results": _format(output_list)}

    # Coze 原生格式：直接拼接
    parts = []
    for i, item in enumerate(output_list, 1):
        if isinstance(item, dict):
            text = to_str(item.get("output") or item.get("content") or item.get("text"))
            parts.append(f"### 召回 {i}\n{text}")
        elif isinstance(item, str):
            parts.append(f"### 召回 {i}\n{item}")
    return {"knowledge_results": "\n\n".join(parts) or "（未召回相关算法）"}
