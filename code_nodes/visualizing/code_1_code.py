import json

async def main(args: Args) -> Output:
    params = args.params
    
    final_code_list = params.get('final_code_list', [])
    
    final_code = ""
    if isinstance(final_code_list, list) and len(final_code_list) > 0:
        final_code = final_code_list[-1]
    elif isinstance(final_code_list, str):
        try:
            parsed = json.loads(final_code_list)
            if isinstance(parsed, list) and len(parsed) > 0:
                final_code = parsed[-1]
            else:
                final_code = final_code_list
        except:
            final_code = final_code_list
    
    ret: Output = {
        "final_code": final_code,
    }
    return ret