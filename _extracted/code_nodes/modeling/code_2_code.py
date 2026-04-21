import json

async def main(args: Args) -> Output:
    params = args.params
    
    # 获取列表
    final_code_list = params.get('final_code_list', [])
    result_list = params.get('result_list', [])
    
    # 取最后一个元素
    final_code = ""
    if isinstance(final_code_list, list) and len(final_code_list) > 0:
        final_code = final_code_list[-1]
    elif isinstance(final_code_list, str):
        try:
            parsed = json.loads(final_code_list)
            if isinstance(parsed, list) and len(parsed) > 0:
                final_code = parsed[-1]
        except:
            final_code = final_code_list
    
    result_json = {}
    if isinstance(result_list, list) and len(result_list) > 0:
        result_json = result_list[-1]
    elif isinstance(result_list, str):
        try:
            parsed = json.loads(result_list)
            if isinstance(parsed, list) and len(parsed) > 0:
                result_json = parsed[-1]
        except:
            pass
    
    # 确保 result_json 是 dict
    if isinstance(result_json, str):
        try:
            result_json = json.loads(result_json)
        except:
            result_json = {"raw": result_json}
    
    ret: Output = {
        "final_code": final_code,
        "result_json": result_json,
    }
    return ret