# LLM Configs
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
    "Qwen": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-max",
    },
    "DeepSeek": {
        "api_base": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat",
    },
    "Zhipu": {
        "api_base": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "model_name": "glm-4v-plus-0111",
    },
    "Doubao": {
        "api_base": "https://ark.cn-beijing.volces.com/api/v3/",
        "model_name": "doubao-seed-1-6251015",
    }
}