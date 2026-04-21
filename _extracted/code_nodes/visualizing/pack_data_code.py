async def main(args: Args) -> Output:
    params = args.params
    
    fig_content = params['fig']
    desc_content = params['desc']
    
    data_object = {
        "fig": fig_content,
        "desc": desc_content,
    }

    return {
        "Pack_Data": data_object
    }