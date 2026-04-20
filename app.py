from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_frontend_module():
    project_root = Path(__file__).resolve().parent
    frontend_app_path = project_root / "frontend" / "app.py"
    frontend_dir = frontend_app_path.parent

    for path in (str(frontend_dir), str(project_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    spec = importlib.util.spec_from_file_location("autostat_frontend_app", frontend_app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Streamlit frontend entrypoint: {frontend_app_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frontend_app = _load_frontend_module()
run_app = frontend_app.run_app


if __name__ == "__main__":
    run_app()
