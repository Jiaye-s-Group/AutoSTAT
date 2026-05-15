"""Background process entry point for the local Reporting_partly workflow."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = REPO_ROOT / "frontend"
for import_root in (REPO_ROOT, FRONTEND_ROOT):
    import_root_text = str(import_root)
    if import_root_text not in sys.path:
        sys.path.insert(0, import_root_text)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
    tmp_path.replace(path)


def _configure_llm(llm_config: dict[str, Any]) -> None:
    api_key = str(llm_config.get("api_key") or os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = str(llm_config.get("base_url") or os.getenv("OPENAI_BASE_URL") or "").strip()
    model = str(llm_config.get("model") or os.getenv("OPENAI_MODEL") or "").strip()

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if model:
        os.environ["OPENAI_MODEL"] = model

    if api_key and base_url and model:
        from core.llm_client import LLMClient

        LLMClient.reconfigure(base_url=base_url, api_key=api_key, model=model)


def _run_reporting_partly(inputs: dict[str, Any]) -> dict[str, Any]:
    from workflows.reporting_partly import run_reporting_partly_workflow

    return run_reporting_partly_workflow(
        toc_text=str(inputs.get("toc_text", "")),
        selected_full_conten=str(inputs.get("selected_full_conten", "")),
        load_abstract=str(inputs.get("load_abstract", "")),
        preproc_abstract=str(inputs.get("preproc_abstract", "")),
        visual_abstract=str(inputs.get("visual_abstract", "")),
        coding_abstract=str(inputs.get("coding_abstract", "")),
        user_input=str(inputs.get("user_input", "")),
        add_preference=str(inputs.get("add_preference", "")),
        preference_select=str(inputs.get("preference_select") or inputs.get("preference_selected") or ""),
        ref_context=str(inputs.get("ref_context", "")),
    )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: python -m workflows.reporting_partly_worker <input.json> <output.json>", file=sys.stderr)
        return 2

    input_path = Path(argv[1])
    output_path = Path(argv[2])

    try:
        payload = _read_json(input_path)
        llm_config = payload.get("llm_config") if isinstance(payload.get("llm_config"), dict) else {}
        inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
        _configure_llm(llm_config)
        result = _run_reporting_partly(inputs)
        _write_json_atomic(output_path, {"ok": True, "result": result})
        return 0
    except BaseException as exc:
        _write_json_atomic(
            output_path,
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
