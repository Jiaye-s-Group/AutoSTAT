"""
Runtime status helpers for the local open-source app.

The current app runs workflows directly in Python. These helpers keep the UI
state that older pages expect, while presenting the runtime as a local engine.
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


LOCAL_RUNTIME_LABEL = "Local Python workflow engine"
LOCAL_RUNTIME_URL = "(local process)"


def ensure_runtime_session_defaults() -> None:
    defaults = {
        "runtime_region": LOCAL_RUNTIME_LABEL,
        "runtime_auth_saved": True,
        "runtime_auth_error": "",
        "runtime_access_token": "local",
        "runtime_token_expires_at": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def handle_runtime_callback() -> None:
    return


def resolve_runtime(default_api_key: str = "", default_url: str = "") -> dict[str, Any]:
    return {
        "region": LOCAL_RUNTIME_LABEL,
        "api_key": default_api_key or "local",
        "runtime_url": default_url or LOCAL_RUNTIME_URL,
        "auth_source": "local",
        "token_expires_at": 0,
    }


def get_runtime_ui_state() -> dict[str, Any]:
    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    if has_key:
        return {
            "variant": "ok",
            "message": f"Status: local workflow engine ready, LLM={model}",
            "configured": True,
            "config_source": ".env",
        }
    return {
        "variant": "warn",
        "message": "Status: OPENAI_API_KEY is not configured.",
        "configured": False,
        "config_source": "",
    }
