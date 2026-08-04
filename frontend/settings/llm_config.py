"""Sidebar controls for LLM provider and credential configuration."""

from __future__ import annotations

import re

import streamlit as st

from core.config_store import (
    LLMConfig,
    apply_llm_config_to_env,
    config_path,
    load_llm_config,
    llm_config_from_env,
    save_llm_config,
)
from core.llm_providers import CUSTOM_PROVIDER_NAME, provider_by_name, provider_names
from utils.i18n import t


def render_llm_config_panel() -> None:
    _inject_llm_config_style()
    st.caption(t("sidebar.llm_caption"))
    _initialize_llm_state()

    names = provider_names()
    provider_choice = st.selectbox(
        t("sidebar.llm_provider"),
        options=names,
        index=names.index(st.session_state.llm_provider)
        if st.session_state.llm_provider in names
        else names.index(CUSTOM_PROVIDER_NAME),
        key="llm_provider_select",
    )
    _apply_provider_selection(provider_choice)

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-xxxxxxxxxxxxxxxx",
        key="llm_key_input",
    )
    base_url = st.text_input(
        "Base URL",
        placeholder="https://api.openai.com/v1",
        key="llm_url_input",
    )
    model_name = st.text_input(
        "Model",
        placeholder="gpt-4o / deepseek-chat / qwen-plus ...",
        key="llm_model_input",
    )

    remember_config = st.checkbox(
        t("sidebar.remember_config"),
        value=False,
        help=t("sidebar.remember_config_help", path=config_path()),
        key="llm_remember_config",
    )

    if st.button(t("sidebar.save_config"), use_container_width=True, key="llm_save_btn"):
        config = LLMConfig(
            provider=provider_choice,
            api_key=_clean(api_key),
            base_url=_clean(base_url),
            model=_clean(model_name),
        )
        _save_current_config(config, remember_config=remember_config)

    _render_connection_status(
        LLMConfig(
            provider=provider_choice,
            api_key=_clean(api_key),
            base_url=_clean(base_url),
            model=_clean(model_name),
        )
    )


def _initialize_llm_state() -> None:
    stored_config = load_llm_config()
    env_config = llm_config_from_env()
    initial_config = stored_config if stored_config.is_complete() else env_config

    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = initial_config.provider or CUSTOM_PROVIDER_NAME
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = initial_config.api_key
    if "llm_base_url" not in st.session_state:
        st.session_state.llm_base_url = initial_config.base_url
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = initial_config.model
    if "llm_configured" not in st.session_state:
        st.session_state.llm_configured = initial_config.is_complete()
    if "llm_connection_signature" not in st.session_state:
        st.session_state.llm_connection_signature = (
            _signature(initial_config) if initial_config.is_complete() else None
        )
    if "llm_key_input" not in st.session_state:
        st.session_state.llm_key_input = st.session_state.llm_api_key
    if "llm_url_input" not in st.session_state:
        st.session_state.llm_url_input = st.session_state.llm_base_url
    if "llm_model_input" not in st.session_state:
        st.session_state.llm_model_input = st.session_state.llm_model

    if initial_config.is_complete():
        apply_llm_config_to_env(initial_config)


def _apply_provider_selection(provider_choice: str) -> None:
    if provider_choice == st.session_state.llm_provider:
        return

    provider = provider_by_name(provider_choice)
    st.session_state.llm_provider = provider_choice
    st.session_state.llm_api_key = ""
    st.session_state.llm_key_input = ""
    st.session_state.llm_base_url = provider.base_url
    st.session_state.llm_url_input = provider.base_url
    st.session_state.llm_model = provider.model
    st.session_state.llm_model_input = provider.model
    st.session_state.llm_configured = False
    st.session_state.llm_connection_signature = None


def _save_current_config(config: LLMConfig, *, remember_config: bool) -> None:
    st.session_state.llm_provider = config.provider
    st.session_state.llm_api_key = config.api_key
    st.session_state.llm_base_url = config.base_url
    st.session_state.llm_model = config.model

    apply_llm_config_to_env(config)
    if config.is_complete():
        try:
            from core.llm_client import LLMClient

            LLMClient.reconfigure(
                base_url=config.base_url,
                api_key=config.api_key,
                model=config.model,
            )
            if remember_config:
                save_llm_config(config)
            st.session_state.llm_configured = True
            st.session_state.llm_connection_signature = _signature(config)
            st.success(t("sidebar.config_saved"))
        except Exception as exc:
            st.session_state.llm_configured = False
            st.session_state.llm_connection_signature = None
            st.error(t("sidebar.config_invalid", error=exc))
    else:
        _reset_llm_runtime()
        st.session_state.llm_configured = False
        st.session_state.llm_connection_signature = None
        st.warning(t("sidebar.fill_llm_fields"))


def _render_connection_status(config: LLMConfig) -> None:
    connected = (
        st.session_state.get("llm_configured")
        and config.is_complete()
        and st.session_state.get("llm_connection_signature") == _signature(config)
    )
    if connected:
        domain = _short_domain(config.base_url)
        st.markdown(
            f'<div class="llm-status llm-ok">{t("sidebar.status_ready")} · <code>{config.model}</code><br/>'
            f'<span style="font-size:0.8rem;opacity:0.7;">{domain}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="llm-status llm-warn">{t("sidebar.status_not_connected")}</div>',
            unsafe_allow_html=True,
        )


def _reset_llm_runtime() -> None:
    try:
        from core.llm_client import LLMClient

        LLMClient._instance = None
    except Exception:
        pass


def _signature(config: LLMConfig) -> tuple[str, str, str, str]:
    return (
        _clean(config.provider),
        _clean(config.api_key),
        _clean(config.base_url),
        _clean(config.model),
    )


def _clean(value: object) -> str:
    return str(value or "").strip()


def _short_domain(base_url: str) -> str:
    match = re.search(r"://([^/]+)", base_url)
    return match.group(1) if match else base_url


def _inject_llm_config_style() -> None:
    st.markdown(
        """
        <style>
        .llm-status {
            font-size: 0.88rem; line-height: 1.5;
            border-radius: 8px; padding: 0.45rem 0.65rem;
            margin-top: 0.2rem; margin-bottom: 0.5rem;
        }
        .llm-ok { color:#065f46; background:#d1fae5; border:1px solid #a7f3d0; }
        .llm-warn { color:#92400e; background:#fef3c7; border:1px solid #fde68a; }
        .st-key-llm_save_btn button {
            background: #2563eb !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
