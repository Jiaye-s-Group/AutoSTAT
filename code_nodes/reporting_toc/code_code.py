import re

async def main(args: Args) -> Output:
    params = args.params
    toc_in = params.get("toc_text", "")

    # 1. 统一转字符串
    if isinstance(toc_in, list):
        raw_text = "\n".join([str(i) for i in toc_in if i is not None])
    elif toc_in is None:
        raw_text = ""
    else:
        raw_text = str(toc_in)

    # 2. 把字面量 \n 转成真实换行
    raw_text = raw_text.replace("\\n", "\n")

    # 3. 拆分、清洗
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

    # 4. 只保留像目录项的行：如 1.xxx / 2.1xxx / 3.2.1xxx
    toc_lines = []
    for line in lines:
        if re.match(r"^\d+(\.\d+)*[\.．]?", line):
            toc_lines.append(line)

    # 5. 检查是否已有“5.结论与应用展望”
    has_conclusion = any("结论" in line or "展望" in line for line in toc_lines)

    # 6. 若缺失，则自动补齐
    if not has_conclusion:
        toc_lines.append("5.结论与应用展望（总结分析发现及模型表现，提出后续优化方向）")

    # 7. 输出规范化结果
    toc_out = "\n".join(toc_lines)

    return {
        "toc_text": toc_out
    }