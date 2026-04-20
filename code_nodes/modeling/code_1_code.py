import json

async def main(args: Args) -> Output:
    params = args.params
    raw = params['body']
    
    # 尝试多次 json.loads 剥掉多层字符串包裹
    data = raw
    for _ in range(5):
        if isinstance(data, dict):
            break
        if not isinstance(data, str):
            break
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            break
    
    # 如果还是字符串，说明有非标准转义
    # 尝试手动修复：把实际的 \\\" 替换为 \"（即去掉一层反斜杠）
    if isinstance(data, str):
        s = data
        # 去掉一层转义
        s = s.replace('\\"', '"')
        s = s.replace('\\\\', '\\')
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            pass
    
    # 最后兜底：用 raw_string_onetry 去处理
    if isinstance(data, str):
        try:
            # 尝试 encode 再 decode 去掉一层转义
            s = raw.encode('utf-8').decode('unicode_escape')
            # 不要替换换行，直接用 raw_decode
            decoder = json.JSONDecoder(strict=False)
            data, _ = decoder.raw_decode(s)
        except Exception:
            pass
    
    if not isinstance(data, dict):
        ret: Output = {
            "success": False,
            "error": f"解析失败，前300字符: {str(raw)[:300]}",
            "final_code": "",
            "result": {},
        }
        return ret
    
    success = data.get('success', False)
    error = data.get('error', None) or ""
    final_code = data.get('final_code', "")
    result = data.get('result', {})
    
    ret: Output = {
        "success": success,
        "error": error,
        "final_code": final_code,
        "result": result,
    }
    return ret