from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parent.parent


def page_file(*parts: str) -> str:
    return str(FRONTEND_DIR.joinpath("workflow", *parts))


def asset_file(*parts: str) -> str:
    return str(FRONTEND_DIR.joinpath(*parts))
