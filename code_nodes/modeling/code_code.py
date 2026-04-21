async def main(args: Args) -> Output:
    params = args.params

    summary_in = params.get("summary_4", None)     # Object (optional)
    abstract_in = params.get("abstract_4", None)   # String (optional)

    def to_str(x):
        return x if isinstance(x, str) else ""

    has_summary = isinstance(summary_in, dict) and len(summary_in) > 0
    has_abstract = isinstance(abstract_in, str) and abstract_in != ""

    # 没有任何输入：全部置空（summary 给空结构，字段齐全）
    if not (has_summary or has_abstract):
        return {
            "abstract_4": "",
            "summary_4": {
                "title": "",
                "desc": "",
                "result": "",
                "code": "",
                "table_title": "",
                "table_markdown": "",
                "table_html": ""
            }
        }

    # 有输入：透传 + 补齐字段
    s = dict(summary_in) if has_summary else {}

    title_out = to_str(s.get("title", ""))
    desc_out = to_str(s.get("desc", ""))
    result_out = to_str(s.get("result", ""))
    code_out = to_str(s.get("code", ""))
    table_title_out = to_str(s.get("table_title", ""))
    table_markdown_out = to_str(s.get("table_markdown", ""))
    table_html_out = to_str(s.get("table_html", ""))

    abstract_out = abstract_in if isinstance(abstract_in, str) else ""

    return {
        "abstract_4": abstract_out,
        "summary_4": {
            "title": title_out,
            "desc": desc_out,
            "result": result_out,
            "code": code_out,
            "table_title": table_title_out,
            "table_markdown": table_markdown_out,
            "table_html": table_html_out,
        }
    }
