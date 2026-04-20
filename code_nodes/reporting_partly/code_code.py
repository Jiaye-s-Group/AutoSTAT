async def main(args: Args) -> Output:
    params = args.params

    final_in = params.get("final_html", [])

    if isinstance(final_in, list):
        final_out = final_in
    else:
        final_out = []

    return {
        "final_html": final_out
    }
