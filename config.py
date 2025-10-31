# 大模型配置
MODEL_CONFIGS = {
    
    "GPT-4o": {
        "api_base": "https://api.openai.com/v1",
        "model_name": "gpt-4o",
    },
    "GPT-5": {
        "api_base": "https://api.openai.com/v1",
        "model_name": "gpt-5",
    },
    "Claude": {
        "api_base": "https://api.anthropic.com",
        "model_name": "claude-3-5-sonnet-latest",
    },
    "通义千问": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-max",
    },
    "DeepSeek": {
        "api_base": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat",
    },
    "智谱AI": {
        "api_base": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "model_name": "glm-4v-plus-0111",
    },
    "豆包": {
        "api_base": "https://ark.cn-beijing.volces.com/api/v3/",
        "model_name": "doubao-seed-1-6-251015",
    }
}