import toml
from pathlib import Path

# We place secrets in the .streamlit folder at the project root
BASE = Path(__file__).parent
SECRETS_DIR  = BASE / ".streamlit"
SECRETS_FILE = SECRETS_DIR / "secrets.toml"

def load_local_api_keys() -> dict[str, str]:
    """
    Read the [api_keys] section from .streamlit/secrets.toml in the project directory.
    If the file or section doesn't exist, return an empty dictionary.
    """
    if not SECRETS_FILE.exists():
        return {}
    data = toml.load(SECRETS_FILE)
    return data.get("api_keys", {})


def update_local_api_key(model_name: str, api_key: str) -> None:
    """
    Write a model_name: api_key pair to the [api_keys] section in .streamlit/secrets.toml.
    If the file or section doesn't exist, it will be automatically created; other existing settings are preserved.
    """
    SECRETS_DIR.mkdir(exist_ok=True)
    if SECRETS_FILE.exists():
        data = toml.load(SECRETS_FILE)
    else:
        data = {}
    data.setdefault("api_keys", {})[model_name] = api_key
    with SECRETS_FILE.open("w", encoding="utf-8") as f:
        toml.dump(data, f)
