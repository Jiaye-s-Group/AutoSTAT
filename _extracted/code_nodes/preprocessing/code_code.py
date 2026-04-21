async def main(args: Args) -> Output:
    params = args.params

    summary = params.get("summary_2", None)     # Object (optional)
    abstract = params.get("abstract_2", None)   # String (optional)
    df = params.get("df", None)                # String (optional) - 建议接 Planner.df

    # 规范 df
    if not isinstance(df, str) or df is None:
        df_out = ""
    else:
        df_out = df

    # 判断 summary_2 是否“真的有输入”
    has_summary = isinstance(summary, dict) and len(summary) > 0

    if not has_summary:
        # false：按要求 processed_df=df，其它空
        summary_out = {
            "title": "",
            "desc": "",
            "processed_df": df_out,
            "code": ""
        }
        abstract_out = ""
    else:
        # true：透传 + 补齐字段，避免下游引用缺 key
        summary_out = dict(summary)

        if "title" not in summary_out or summary_out["title"] is None:
            summary_out["title"] = ""
        if "desc" not in summary_out or summary_out["desc"] is None:
            summary_out["desc"] = ""
        if "processed_df" not in summary_out or summary_out["processed_df"] is None:
            # 如果真分支居然没给 processed_df，也兜底用 df
            summary_out["processed_df"] = df_out
        if "code" not in summary_out or summary_out["code"] is None:
            summary_out["code"] = ""

        # abstract：有字符串就用，否则置空
        if isinstance(abstract, str) and abstract is not None:
            abstract_out = abstract
        else:
            abstract_out = ""

    ret: Output = {
        "summary_2": summary_out,
        "abstract_2": abstract_out
    }
    return ret
