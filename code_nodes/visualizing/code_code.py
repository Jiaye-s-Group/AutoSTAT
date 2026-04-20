async def main(args: Args) -> Output:
    params = args.params

    full_in = params.get("full", None)
    summary_in = params.get("summary_3", None)
    abstract_in = params.get("abstract_3", None)

    has_full = isinstance(full_in, str) and full_in != ""
    has_abstract = isinstance(abstract_in, str) and abstract_in != ""
    has_summary = isinstance(summary_in, dict) and len(summary_in) > 0

    has_any_input = has_full or has_abstract or has_summary

    # 没有任何输入：全部置空（summary 给空结构，避免下游取 key 报错）
    if not has_any_input:
        return {
            "full": "",
            "abstract_3": "",
            "summary_3": {
                "title": "",
                "fig_analysis": []
            }
        }

    # 有输入：透传 + 补齐结构
    # full / abstract
    full_out = full_in if isinstance(full_in, str) and full_in is not None else ""
    abstract_out = abstract_in if isinstance(abstract_in, str) and abstract_in is not None else ""

    # summary_3
    if has_summary:
        summary_out = dict(summary_in)  # 浅拷贝
    else:
        summary_out = {}

    # title 补齐
    if "title" not in summary_out or summary_out["title"] is None or not isinstance(summary_out["title"], str):
        summary_out["title"] = ""

    # fig_analysis 补齐为 list
    fa = summary_out.get("fig_analysis", [])
    if not isinstance(fa, list):
        fa = []

    # 每项补齐 fig / analysis
    fa_out = []
    for item in fa:
        if isinstance(item, dict):
            obj = dict(item)
        else:
            obj = {}
        if "fig" not in obj or obj["fig"] is None or not isinstance(obj["fig"], str):
            obj["fig"] = ""
        if "analysis" not in obj or obj["analysis"] is None or not isinstance(obj["analysis"], str):
            obj["analysis"] = ""
        fa_out.append(obj)

    summary_out["fig_analysis"] = fa_out

    return {
        "full": full_out,
        "abstract_3": abstract_out,
        "summary_3": summary_out
    }
