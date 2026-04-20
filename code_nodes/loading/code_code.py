async def main(args: Args) -> Output:
    params = args.params

    # 取输入（不存在时为 None）
    summary = params.get("summary_1", None)
    abstract = params.get("abstract_1", None)

    # summary_1：如果没输入/不是 dict/是空 dict，则输出空模板
    if not isinstance(summary, dict) or len(summary) == 0:
        summary_out = {
            "title": "",
            "desc": "",
            "df": ""
        }
    else:
        # 透传 + 补齐字段，避免下游取 key 报错
        summary_out = dict(summary)  # 浅拷贝，避免改到原对象
        if "title" not in summary_out or summary_out["title"] is None:
            summary_out["title"] = ""
        if "desc" not in summary_out or summary_out["desc"] is None:
            summary_out["desc"] = ""
        if "df" not in summary_out or summary_out["df"] is None:
            summary_out["df"] = ""

    # abstract_1：没输入或不是字符串则置空
    if not isinstance(abstract, str) or abstract is None:
        abstract_out = ""
    else:
        abstract_out = abstract

    ret: Output = {
        "summary_1": summary_out,
        "abstract_1": abstract_out
    }
    return ret
