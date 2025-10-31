import random

def get_spinner_msg(stage="writing"):
    msg_pool = {
        "summarizing": [
            "Summarizing analysis results from each module...",
            "Please wait, summarizing the content from previous agents...",
            "AI is organizing previous analyses, please wait...",
            "Integrating conclusions from each analysis step..."
        ],
        "writing": [
            "Generating content for each chapter...",
            "Please wait, the system is writing the report in detail...",
            "AI is gradually generating report chapters...",
            "Organizing and writing each chapter..."
        ],
        "default": [
            "Processing data, please wait...",
            "AI is working hard to generate results...",
            "Please wait patiently, calculations are in progress..."
        ]
    }

    pool = msg_pool.get(stage, msg_pool["default"])
    return random.choice(pool)
