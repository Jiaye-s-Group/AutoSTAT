"""Generate the loading-stage summary and abstract."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.llm_client import chat
from core.prompt_template import render_file
from workflows._plugins import summary1_composer


def run_loading_workflow(
    *,
    shape_0: int,
    shape_1: int,
    dtype_info_str: str,
    head_dict_str: str,
    loading_auto: bool = True,
    user_input: str = "",
    add_preference: str = "",
    preference_selected: str = "",
    ref_context: str = "",
) -> dict[str, Any]:
    # Return an empty stage result when planning skips loading analysis.
    if not loading_auto:
        return {
            "summary_1": {"title": "", "desc": "", "df": ""},
            "abstract_1": "",
        }

    ctx = {
        "shape_0": shape_0,
        "shape_1": shape_1,
        "dtype_info_str": dtype_info_str,
        "head_dict_str": head_dict_str,
        "user_input": user_input or "",
        "add_preference": add_preference or "",
        "preference_selected": preference_selected or "",
        "ref_context": ref_context or "（无参考资料）",
    }

    # These prompts depend only on raw metadata, so they can run in parallel.
    desc_sys = render_file("loading/do_data_description__llm_sys.txt", ctx)
    desc_user = render_file("loading/do_data_description__llm_user.txt", ctx)
    chap_sys = render_file("loading/chap1_summary_html_llm_sys.txt", ctx)
    chap_user = render_file("loading/chap1_summary_html_llm_user.txt", ctx)
    abs_sys = render_file("loading/abs1_check_abstract_llm_sys.txt", ctx)
    abs_user = render_file("loading/abs1_check_abstract_llm_user.txt", ctx)

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_desc = pool.submit(chat, desc_sys, desc_user, name="loading.do_data_description")
        f_chap = pool.submit(chat, chap_sys, chap_user, name="loading.chap1_summary_html")
        f_abs = pool.submit(chat, abs_sys, abs_user, name="loading.abs1_check_abstract")
        description = f_desc.result()
        chap_desc = f_chap.result()
        abstract_1 = f_abs.result().strip()

    composed = summary1_composer(desc=chap_desc, head_dict_str=head_dict_str)
    summary_1 = composed["summary_1"]

    return {
        "summary_1": {
            "title": summary_1.get("title") or "数据概览与数据含义分析",
            "desc": summary_1.get("desc") or "",
            "df": summary_1.get("df") or "",
        },
        "abstract_1": abstract_1 or "",
        "_description": description,
    }


# CLI smoke-test entry point.

if __name__ == "__main__":
    import sys

    import pandas as pd

    from workflows._plugins import df_to_meta

    if len(sys.argv) < 2:
        print("用法: python -m workflows.loading <csv_path> [user_input]")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    user_input = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"✓ 读取 {sys.argv[1]}: {df.shape}")

    meta = df_to_meta(df)
    result = run_loading_workflow(
        shape_0=meta["shape_0"],
        shape_1=meta["shape_1"],
        dtype_info_str=meta["dtype_info_str"],
        head_dict_str=meta["head_dict_str"],
        loading_auto=True,
        user_input=user_input,
    )

    print("\n===== summary_1.title =====")
    print(result["summary_1"]["title"])
    print("\n===== summary_1.desc =====")
    print(result["summary_1"]["desc"])
    print("\n===== abstract_1 =====")
    print(result["abstract_1"])
