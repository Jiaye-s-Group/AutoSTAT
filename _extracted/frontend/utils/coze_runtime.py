"""
本地化后的 coze_runtime stub。
原版用于 Coze OAuth；本地化后不调用 Coze，但保留 API 签名。
"""
from __future__ import annotations

import os
from typing import Any

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

COZE_REGION_LOCAL = "本地模式（不再依赖 Coze）"
COZE_REGION_OPTIONS = [COZE_REGION_LOCAL]
COZE_REGION_TO_URL = {COZE_REGION_LOCAL: "(本地 Python 实现)"}


def ensure_coze_session_defaults() -> None:
    defaults = {
        "coze_region": COZE_REGION_LOCAL,
        "coze_auth_saved": True,
        "coze_auth_error": "",
        "coze_access_token": "local",
        "coze_refresh_token": "",
        "coze_token_expires_at": 0,
        "coze_redirect_uri": "",
        "coze_oauth_state": "",
        "coze_oauth_authorize_url": "",
        "coze_oauth_authorize_redirect_uri": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_coze_auth() -> None:
    pass


def load_coze_oauth_config():
    return None


def get_coze_oauth_authorize_url(force_refresh: bool = False) -> str:
    return ""


def handle_coze_oauth_callback() -> None:
    return


def refresh_coze_access_token_if_needed(buffer_seconds: int = 120) -> str:
    return ""


def resolve_coze_runtime(default_api_key: str = "", default_url: str = "") -> dict[str, Any]:
    return {
        "region": COZE_REGION_LOCAL,
        "api_key": "local",
        "coze_url": COZE_REGION_TO_URL[COZE_REGION_LOCAL],
        "auth_source": "local",
        "token_expires_at": 0,
    }


def get_coze_auth_ui_state() -> dict[str, Any]:
    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    if has_key:
        return {
            "variant": "ok",
            "message": f"状态：本地模式已就绪，LLM={model}",
            "configured": True,
            "redirect_uri": "",
            "config_source": ".env",
        }
    return {
        "variant": "warn",
        "message": "状态：未找到 OPENAI_API_KEY，请在 .env 中配置",
        "configured": False,
        "redirect_uri": "",
        "config_source": "",
    }
