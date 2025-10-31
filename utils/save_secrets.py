import toml
from pathlib import Path

# 我们把 secrets 放在项目根目录下的 .streamlit 文件夹
BASE = Path(__file__).parent
SECRETS_DIR  = BASE / ".streamlit"
SECRETS_FILE = SECRETS_DIR / "secrets.toml"


def load_local_api_keys() -> dict[str, str]:
    """
    从项目目录的 .streamlit/secrets.toml 中读取 [api_keys] 部分。
    如果文件或该节不存在，返回空字典。
    """
    if not SECRETS_FILE.exists():
        return {}
    data = toml.load(SECRETS_FILE)
    return data.get("api_keys", {})


def update_local_api_key(model_name: str, api_key: str) -> None:
    """
    将一对 model_name: api_key 写入 .streamlit/secrets.toml 的 [api_keys]。
    如果文件或该节不存在，会自动创建；保留其它已有设置。
    """
    SECRETS_DIR.mkdir(exist_ok=True)
    if SECRETS_FILE.exists():
        data = toml.load(SECRETS_FILE)
    else:
        data = {}
    data.setdefault("api_keys", {})[model_name] = api_key
    with SECRETS_FILE.open("w", encoding="utf-8") as f:
        toml.dump(data, f)
