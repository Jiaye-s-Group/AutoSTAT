async def main(args):
    # 从输入参数中获取 data
    raw_data = args.params.get("data", "")
    
    # 逻辑处理：
    # 如果 data 本身就是以 http 开头的字符串，直接返回它
    if isinstance(raw_data, str) and raw_data.startswith("http"):
        file_url = raw_data
    # 如果是一个字典对象（预防格式变化）
    elif isinstance(raw_data, dict):
        file_url = raw_data.get("url", "")
    else:
        file_url = None

    return {
        "file_url": file_url
    }