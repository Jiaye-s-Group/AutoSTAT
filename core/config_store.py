"""User-level configuration persistence for AutoSTAT."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


CONFIG_HOME_ENV = "AUTOSTAT_CONFIG_HOME"
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "autostat"
CONFIG_FILE_NAME = "config.toml"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = ""
    api_key: str = ""
    base_url: str = ""
    model: str = ""

    def is_complete(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def redacted_api_key(self) -> str:
        if not self.api_key:
            return ""
        if len(self.api_key) <= 8:
            return "****"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"


def config_dir() -> Path:
    configured_home = os.getenv(CONFIG_HOME_ENV)
    if configured_home:
        return Path(configured_home).expanduser()
    return DEFAULT_CONFIG_DIR


def config_path() -> Path:
    return config_dir() / CONFIG_FILE_NAME


def load_llm_config(path: Path | None = None) -> LLMConfig:
    target = path or config_path()
    if not target.exists() or tomllib is None:
        return LLMConfig()

    try:
        with target.open("rb") as file:
            data = tomllib.load(file)
    except Exception:
        return LLMConfig()

    llm_data = data.get("llm") if isinstance(data, dict) else {}
    if not isinstance(llm_data, dict):
        return LLMConfig()

    return LLMConfig(
        provider=str(llm_data.get("provider") or "").strip(),
        api_key=str(llm_data.get("api_key") or "").strip(),
        base_url=str(llm_data.get("base_url") or "").strip(),
        model=str(llm_data.get("model") or "").strip(),
    )


def save_llm_config(config: LLMConfig, path: Path | None = None) -> Path:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "[llm]",
            f'provider = "{_toml_escape(config.provider)}"',
            f'api_key = "{_toml_escape(config.api_key)}"',
            f'base_url = "{_toml_escape(config.base_url)}"',
            f'model = "{_toml_escape(config.model)}"',
            "",
        ]
    )
    target.write_text(content, encoding="utf-8")
    return target


def apply_llm_config_to_env(config: LLMConfig) -> None:
    env_values = {
        "OPENAI_API_KEY": config.api_key,
        "OPENAI_BASE_URL": config.base_url,
        "OPENAI_MODEL": config.model,
    }
    for env_key, env_value in env_values.items():
        if env_value:
            os.environ[env_key] = env_value
        else:
            os.environ.pop(env_key, None)


def llm_config_from_env(provider: str = "") -> LLMConfig:
    return LLMConfig(
        provider=provider,
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
        model=os.getenv("OPENAI_MODEL", "").strip(),
    )


def _toml_escape(value: str) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"')
