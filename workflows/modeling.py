"""
Modeling workflow 本地实现。

原 Coze 流程:
    Start → Condition(modeling_auto==True)
      → Sec4_get_model_suggestion(LLM)   [推荐模型]
      → Sec4_refine_suggestion(LLM)      [精炼]
      → get_query(LLM) → Knowledge(RAG) → format_recall(plugin)
      → sec4_code_generation(LLM)        [生成训练代码]
      → Variable assign_1: code_modeling = code
      → Loop(max 5): [修复循环]
          ├─ code_runner(HTTP→本地)
          ├─ if success: break
          └─ sec4_code_fixed(LLM) → 更新 code_modeling
      → Code_2(取 Loop 的 final_code + result_list)
      → sec4_result_format_prompt(LLM)   [解析结果]
      → Sec4_summary_html(LLM)           [章节正文]
      → Sec4_check_abstract(LLM)         [摘要]
      → sec4_composer(plugin)
      → Code(兜底) → End

输出:
    {
      "summary_4": {title, desc, result, code},
      "abstract_4": "...",
      "model_suggestion": "..."
    }
"""
from __future__ import annotations

from typing import Any

from core.llm_client import chat
from core.modeling_table_utils import (
    build_model_comparison_table_bundle,
)
from core.prompt_template import render_file
from core.rag_retriever import retrieve
from core.workflow_runner import to_str
from workflows._plugins import (
    format_recall,
    code_runner,
    sec4_composer,
)

MAX_FIX_ATTEMPTS = 5


def run_modeling_workflow(
    *,
    data: str,
    df_head: str,
    columns: list,
    modeling_auto: bool = True,
    target: str = "",
    train_code: str = "",
    user_input: str = "",
    user_prompt: str = "",
    add_preference: str = "",
    preference_selected: str = "",
    ref_context: str = "",
) -> dict[str, Any]:
    # ---------- Condition: modeling_auto ----------
    if not modeling_auto:
        return {
            "summary_4": {
                "title": "",
                "desc": "",
                "result": "",
                "code": "",
                "table_title": "",
                "table_markdown": "",
                "table_html": "",
            },
            "abstract_4": "",
            "model_suggestion": "",
        }

    ctx: dict[str, Any] = {
        "data": data,
        "df_head": df_head,
        "columns": columns,
        "target": target or "",
        "train_code": train_code or "",
        "user_input": user_input or "",
        "user_prompt": user_prompt or user_input or "",
        "add_preference": add_preference or "",
        "additional_preference": add_preference or "",
        "preference_selected": preference_selected or "",
        "preference_select": preference_selected or "",
        "ref_context": ref_context or "（无参考资料）",
    }

    # ---------- 节点 1: Sec4_get_model_suggestion ----------
    sug_sys = render_file("modeling/sec4_get_model_suggestion_llm_sys.txt", ctx)
    sug_user = render_file("modeling/sec4_get_model_suggestion_llm_user.txt", ctx)
    model_suggestion = chat(sug_sys, sug_user, name="model.get_suggestion").strip()
    ctx["model_suggestion"] = model_suggestion

    # ---------- 节点 2: Sec4_refine_suggestion ----------
    ref_sys = render_file("modeling/sec4_refine_suggestion_llm_sys.txt", ctx)
    ref_user = render_file("modeling/sec4_refine_suggestion_llm_user.txt", ctx)
    refined_suggestions = chat(ref_sys, ref_user, name="model.refine").strip()
    ctx["refined_suggestions"] = refined_suggestions
    ctx["refine_suggestion"] = refined_suggestions  # Coze 里的字段名

    # ---------- 节点 3: RAG ----------
    q_sys = render_file("modeling/get_query_llm_sys.txt", ctx)
    q_user = render_file("modeling/get_query_llm_user.txt", ctx)
    rag_query = chat(q_sys, q_user, name="model.get_query", temperature=0).strip()

    recall_results = retrieve(rag_query, top_k=3)
    ctx["knowledge_results"] = format_recall(output_list=recall_results)["knowledge_results"]

    # ---------- 节点 4: sec4_code_generation ----------
    cg_sys = render_file("modeling/sec4_code_generation_llm_sys.txt", ctx)
    cg_user = render_file("modeling/sec4_code_generation_llm_user.txt", ctx)
    generated_code = chat(cg_sys, cg_user, name="model.code_generation").strip()
    generated_code = _unwrap_code_block(generated_code)

    # ---------- 节点 5: Loop (训练代码自愈，最多 5 次) ----------
    current_code = generated_code
    success = False
    last_error = ""
    final_result_json: dict = {}
    final_result_str = ""

    for attempt in range(MAX_FIX_ATTEMPTS):
        run_result = _run_modeling_code(code=current_code, data=data)
        if run_result["is_success"]:
            success = True
            final_result_str = run_result.get("stdout", "")
            final_result_json = run_result.get("result_json", {})
            break

        last_error = run_result.get("error", "")
        if attempt >= MAX_FIX_ATTEMPTS - 1:
            break

        # ---------- 节点 5.x: sec4_code_fixed LLM ----------
        fix_ctx = {
            **ctx,
            "code": current_code,
            "code_modeling": current_code,
            "error_msg": last_error,
            "error": last_error,
        }
        fix_sys = render_file("modeling/sec4_code_fixed_llm_sys.txt", fix_ctx)
        fix_user = render_file("modeling/sec4_code_fixed_llm_user.txt", fix_ctx)
        fixed = chat(
            fix_sys, fix_user, name=f"model.code_fixed.{attempt+1}", temperature=0.3
        ).strip()
        fixed = _unwrap_code_block(fixed)
        if fixed:
            current_code = fixed

    final_code = current_code

    # 失败兜底
    if not success:
        return {
            "summary_4": {
                "title": "建模分析",
                "desc": f"建模代码执行失败：{last_error[:500]}",
                "result": "",
                "code": final_code,
                "table_title": "",
                "table_markdown": "",
                "table_html": "",
            },
            "abstract_4": f"建模代码执行失败：{last_error[:200]}",
            "model_suggestion": model_suggestion,
        }

    # ---------- 节点 6: sec4_result_format_prompt ----------
    ctx["final_code"] = final_code
    ctx["modeling_code"] = final_code
    ctx["result_json"] = final_result_json
    ctx["result"] = final_result_str
    table_bundle = build_model_comparison_table_bundle(
        final_result_json,
        target=ctx.get("target", ""),
        user_input=ctx.get("user_input", ""),
        additional_preference=ctx.get("additional_preference", ""),
    )
    ctx["comparison_table_title"] = table_bundle.get("title", "")
    ctx["comparison_table_markdown"] = table_bundle.get("markdown_table", "")
    ctx["comparison_table_html"] = table_bundle.get("html_table", "")

    rfp_sys = render_file("modeling/sec4_result_format_prompt_llm_sys.txt", ctx)
    rfp_user = render_file("modeling/sec4_result_format_prompt_llm_user.txt", ctx)
    result_format = chat(rfp_sys, rfp_user, name="model.result_format").strip()
    ctx["result_format"] = result_format

    # ---------- 节点 7: Sec4_summary_html ----------
    sh_sys = render_file("modeling/sec4_summary_html_llm_sys.txt", ctx)
    sh_user = render_file("modeling/sec4_summary_html_llm_user.txt", ctx)
    desc = chat(sh_sys, sh_user, name="model.summary_html").strip()

    # ---------- 节点 8: Sec4_check_abstract ----------
    ab_sys = render_file("modeling/sec4_check_abstract_llm_sys.txt", ctx)
    ab_user = render_file("modeling/sec4_check_abstract_llm_user.txt", ctx)
    abstract_4 = chat(ab_sys, ab_user, name="model.check_abstract").strip()

    # ---------- 节点 9: sec4_composer ----------
    composed = sec4_composer(
        code=final_code,
        desc=desc,
        result=result_format,
        table_title=table_bundle.get("title", ""),
        table_markdown=table_bundle.get("markdown_table", ""),
        table_html=table_bundle.get("html_table", ""),
    )

    return {
        "summary_4": composed["summary_4"],
        "abstract_4": abstract_4,
        "model_suggestion": model_suggestion,
        # 额外信息
        "_refined_suggestions": refined_suggestions,
        "_final_code": final_code,
        "_fix_attempts": attempt + 1 if success else MAX_FIX_ATTEMPTS,
    }


# ===================================================================
# 建模代码专用 runner —— 比 preprocessing 多一个 result_json 输出
# ===================================================================


def _run_modeling_code(*, code: str, data: str, timeout_seconds: int = 300) -> dict[str, Any]:
    """
    执行建模训练代码。
    约定用户代码会往 stdout 打印 JSON（如 {"accuracy": 0.95}），并设置 result 变量。
    """
    import json
    import subprocess
    import sys
    import textwrap

    user_code = to_str(code).strip()
    if not user_code:
        return {"is_success": False, "error": "空代码", "stdout": "", "result_json": {}}

    script = '''import json, sys, traceback
import pandas as pd
import numpy as np

_RECORDS = json.loads(sys.stdin.read())
df = pd.DataFrame(_RECORDS)

result = {}
try:
__USER_CODE__
except Exception as e:
    print("__AUTOSTAT_ERROR__", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(2)

# 收集 result 变量
# 兼容工作流脚本返回 result_dict，以及旧版 result / result_json。
_result_candidates = (
    locals().get("result_dict"),
    locals().get("result_json"),
    locals().get("result"),
)
_out = next((item for item in _result_candidates if item is not None), {})
if not isinstance(_out, dict):
    try:
        _out = {"value": str(_out)}
    except Exception:
        _out = {}

# 把 numpy / pandas 类型变成原生 JSON 友好类型
def _clean(o):
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            return str(o)
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o

print("__AUTOSTAT_RESULT__:" + json.dumps(_clean(_out), ensure_ascii=False))
'''

    indented = textwrap.indent(user_code, " " * 4)
    full_script = script.replace("__USER_CODE__", indented)

    try:
        completed = subprocess.run(
            [sys.executable, "-c", full_script],
            input=to_str(data) or "[]",
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "is_success": False,
            "error": f"代码执行超时（>{timeout_seconds}s）",
            "stdout": "",
            "result_json": {},
        }

    if completed.returncode != 0:
        return {
            "is_success": False,
            "error": (completed.stderr or "")[:1500],
            "stdout": completed.stdout or "",
            "result_json": {},
        }

    # 分离打印输出和 result
    result_json: dict = {}
    lines = completed.stdout.splitlines()
    stdout_clean_lines = []
    for line in lines:
        if line.startswith("__AUTOSTAT_RESULT__:"):
            try:
                result_json = json.loads(line[len("__AUTOSTAT_RESULT__:"):])
            except Exception:
                result_json = {}
        else:
            stdout_clean_lines.append(line)

    return {
        "is_success": True,
        "error": "",
        "stdout": "\n".join(stdout_clean_lines),
        "result_json": result_json,
    }


def _unwrap_code_block(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return t


# ---------- CLI 测试入口 ----------

if __name__ == "__main__":
    import sys

    import pandas as pd

    from workflows._plugins import df_to_meta

    if len(sys.argv) < 3:
        print("用法: python -m workflows.modeling <csv_path> <target_column> [user_input]")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    target = sys.argv[2]
    user_input = sys.argv[3] if len(sys.argv) > 3 else ""
    print(f"✓ 读取 {sys.argv[1]}: {df.shape}  target={target}")

    meta = df_to_meta(df)
    result = run_modeling_workflow(
        data=meta["df"],
        df_head=meta["head_dict_str"],
        columns=list(df.columns),
        modeling_auto=True,
        target=target,
        user_input=user_input,
    )

    print("\n===== model_suggestion =====")
    print(result["model_suggestion"][:500])
    print("\n===== summary_4.desc =====")
    print(result["summary_4"]["desc"][:500])
    print("\n===== summary_4.result =====")
    print(result["summary_4"]["result"][:500])
    print("\n===== abstract_4 =====")
    print(result["abstract_4"])
