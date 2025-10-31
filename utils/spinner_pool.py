import random

def get_spinner_msg(stage="writing"):
    msg_pool = {
        "summarizing": [
            "正在汇总各模块的分析结果...",
            "稍等一下，正在总结前面几个 Agent 的内容...",
            "AI 正在整理前面的分析，请稍候...",
            "正在综合各分析步骤的结论..."
        ],
        "writing": [
            "正在生成各章节内容...",
            "请稍候，系统正在详细撰写报告...",
            "AI 正在逐步生成报告章节...",
            "正在整理并撰写每一章节..."
        ],
        "default": [
            "正在处理数据，请稍候...",
            "AI 正在努力生成结果...",
            "请耐心等待，正在计算中..."
        ]
    }

    pool = msg_pool.get(stage, msg_pool["default"])
    return random.choice(pool)
