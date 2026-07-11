"""LLM provider catalog for OpenAI-compatible AutoSTAT deployments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMProvider:
    """Display metadata and default endpoint for an LLM provider."""

    name: str
    base_url: str
    model: str
    api_type: str = "openai-compatible"
    notes: str = ""


CUSTOM_PROVIDER_NAME = "Custom OpenAI-compatible"


PROVIDERS: tuple[LLMProvider, ...] = (
    LLMProvider(
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        notes="Cost-effective OpenAI-compatible chat endpoint.",
    ),
    LLMProvider(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        notes="Official OpenAI API endpoint.",
    ),
    LLMProvider(
        name="Claude via OpenAI-compatible gateway",
        base_url="",
        model="claude-3-5-sonnet-latest",
        notes="Use an OpenAI-compatible gateway URL for Claude-compatible deployments.",
    ),
    LLMProvider(
        name="Qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        notes="Alibaba Cloud DashScope OpenAI-compatible mode.",
    ),
    LLMProvider(
        name="ZhipuAI",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4-flash",
        notes="Zhipu OpenAI-compatible endpoint.",
    ),
    LLMProvider(
        name="Doubao",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        model="doubao-1-5-pro-256k-250115",
        notes="Volcengine Ark OpenAI-compatible endpoint.",
    ),
    LLMProvider(
        name="AIHubMix",
        base_url="https://aihubmix.com/v1",
        model="gpt-4o-mini",
        notes="Third-party OpenAI-compatible model gateway.",
    ),
    LLMProvider(
        name=CUSTOM_PROVIDER_NAME,
        base_url="",
        model="",
        notes="Any service that exposes the OpenAI Chat Completions API.",
    ),
)


def provider_names() -> list[str]:
    return [provider.name for provider in PROVIDERS]


def provider_by_name(name: str | None) -> LLMProvider:
    normalized = str(name or "").strip()
    for provider in PROVIDERS:
        if provider.name == normalized:
            return provider
    return PROVIDERS[-1]
