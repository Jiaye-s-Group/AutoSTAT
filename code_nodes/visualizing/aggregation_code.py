
async def main(args: Args) -> Output:
    params = args.params
    
    fig_content = params['fig']
    desc_content = params['desc']
    analysis_content = params['analysis']
    
    data_object = {
        "fig": fig_content,
        "desc": desc_content,
        "analysis": analysis_content,
    }

    return {
        "aggregation": data_object
    }    