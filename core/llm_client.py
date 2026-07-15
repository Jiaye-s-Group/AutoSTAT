"""
OpenAI-compatible LLM client.

Features:
- Unified `chat(sys, user)` interface
- `chat_json()` with best-effort JSON parsing fallback
- Optional `.env` loading
- Automatic retries on transient failures
"""
from __future__ import annotations

import json
import os
import re
import time
from contextvars import ContextVar, copy_context
from concurrent.futures import Executor, Future
from pathlib import Path
from typing import Any

try:  # Optional dependency
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

_OpenAI = None
_context_client: ContextVar["LLMClient | None"] = ContextVar(
    "autostat_llm_client",
    default=None,
)
_context_blocked_reason: ContextVar[str | None] = ContextVar(
    "autostat_llm_blocked_reason",
    default=None,
)


def _get_openai_cls():
    global _OpenAI
    if _OpenAI is None:
        try:
            from openai import OpenAI  # type: ignore

            _OpenAI = OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "请先安装 openai>=1.0: pip install openai python-dotenv"
            ) from e
    return _OpenAI


DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
DEFAULT_CODE_MAX_TOKENS = int(os.getenv("LLM_CODE_MAX_TOKENS", "8192"))
DEFAULT_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))


class LLMClient:
    """OpenAI-compatible client with process and session-level configuration."""

    _instance: "LLMClient | None" = None

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if not api_key:
            raise RuntimeError(
                "未配置 API Key。请在侧边栏“大模型配置”中填写 API Key、Base URL 和 Model 后保存。"
            )
        if not base_url:
            raise RuntimeError(
                "未配置 Base URL。请在侧边栏“大模型配置”中填写 Base URL，例如 https://api.openai.com/v1。"
            )

        OpenAI = _get_openai_cls()
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.base_url = base_url

    @classmethod
    def get(cls) -> "LLMClient":
        blocked_reason = _context_blocked_reason.get()
        if blocked_reason:
            raise RuntimeError(blocked_reason)
        context_client = _context_client.get()
        if context_client is not None:
            return context_client
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def configure_context(
        cls,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> "LLMClient":
        client = cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
        )
        _context_client.set(client)
        _context_blocked_reason.set(None)
        return client

    @classmethod
    def clear_context(cls) -> None:
        _context_client.set(None)
        _context_blocked_reason.set(None)

    @classmethod
    def block_context(cls, reason: str) -> None:
        _context_client.set(None)
        _context_blocked_reason.set(reason)

    @classmethod
    def reconfigure(
        cls,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> "LLMClient":
        cls._instance = cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
        )
        _context_blocked_reason.set(None)
        return cls._instance

    def chat(
        self,
        sys: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        response_json: bool = False,
        name: str = "unnamed",
        retries: int = 2,
    ) -> str:
        messages = []
        if sys:
            messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": user})

        return self._complete_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            response_json=response_json,
            name=name,
            retries=retries,
        )

    def chat_multimodal(
        self,
        sys: str,
        text: str,
        *,
        image_data_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        response_json: bool = False,
        name: str = "unnamed",
        retries: int = 2,
        fallback_to_text: bool = True,
    ) -> str:
        if not image_data_url:
            return self.chat(
                sys,
                text,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                response_json=response_json,
                name=name,
                retries=retries,
            )

        messages: list[dict[str, Any]] = []
        if sys:
            messages.append({"role": "system", "content": sys})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        )
        try:
            return self._complete_messages(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                response_json=response_json,
                name=name,
                retries=retries,
            )
        except Exception:
            if not fallback_to_text:
                raise
            return self.chat(
                sys,
                text,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                response_json=response_json,
                name=f"{name}.text_fallback",
                retries=retries,
            )

    def _complete_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None,
        max_tokens: int | None,
        model: str | None,
        response_json: bool,
        name: str,
        retries: int,
    ) -> str:
        _ = name
        model_name = model or self.model

        use_max_completion_tokens = "api.openai.com" in (self.base_url or "")
        kwargs = self._build_chat_kwargs(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            response_json=response_json,
            use_max_completion_tokens=use_max_completion_tokens,
        )

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                return text
            except Exception as exc:  # pragma: no cover
                last_err = exc
                err_text = str(exc).lower()

                if use_max_completion_tokens and "max_completion_tokens" in err_text:
                    use_max_completion_tokens = False
                    kwargs = self._build_chat_kwargs(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_json="response_format" in kwargs,
                        use_max_completion_tokens=False,
                    )
                    continue

                if response_json and "response_format" in err_text:
                    kwargs.pop("response_format", None)
                    continue
                if attempt < retries:
                    time.sleep(1 + attempt)
                    continue
                raise
        raise last_err  # type: ignore

    def _build_chat_kwargs(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        response_json: bool,
        use_max_completion_tokens: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else DEFAULT_TEMPERATURE,
        }
        token_key = "max_completion_tokens" if use_max_completion_tokens else "max_tokens"
        kwargs[token_key] = max_tokens or DEFAULT_MAX_TOKENS
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs


def chat(sys: str, user: str, **kwargs: Any) -> str:
    return LLMClient.get().chat(sys, user, **kwargs)


def chat_code(sys: str, user: str, **kwargs: Any) -> str:
    kwargs.setdefault("max_tokens", DEFAULT_CODE_MAX_TOKENS)
    return chat(sys, user, **kwargs)


def chat_multimodal(sys: str, text: str, **kwargs: Any) -> str:
    return LLMClient.get().chat_multimodal(sys, text, **kwargs)


def chat_json(sys: str, user: str, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("response_json", True)
    raw = chat(sys, user, **kwargs)
    return parse_json_best_effort(raw)


def submit_with_context(
    executor: Executor,
    fn,
    /,
    *args: Any,
    **kwargs: Any,
) -> Future:
    """Submit work while preserving the current session's LLM client."""
    context = copy_context()
    return executor.submit(context.run, fn, *args, **kwargs)


def parse_json_best_effort(raw: str) -> dict[str, Any]:
    s = (raw or "").strip()
    if not s:
        return {}

    try:
        return json.loads(s)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", s, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(s[first : last + 1])
        except Exception:
            pass

    return {"_raw": s, "_parse_failed": True}
