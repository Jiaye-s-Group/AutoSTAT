from __future__ import annotations

from typing import Any


VISUALIZATION_FIGURE_ARTIFACTS_KEY = "visualization_figure_artifacts"
VISUALIZATION_FAILED_FIGURES_KEY = "visualization_failed_figures"


def figure_chart_id(index: int, key: Any = "") -> str:
    key_text = str(key or "").strip()
    if key_text:
        safe_key = "".join(ch.lower() if ch.isalnum() else "_" for ch in key_text)
        safe_key = "_".join(part for part in safe_key.split("_") if part)
        if safe_key:
            return f"viz_{index + 1:04d}_{safe_key[:48]}"
    return f"viz_{index + 1:04d}"


def normalize_figure_artifact(
    item: Any,
    index: int,
    *,
    title: str = "",
    description: str = "",
    language: str = "",
) -> dict[str, Any]:
    source = dict(item) if isinstance(item, dict) else {"fig": item}
    fig_dict_key = str(source.get("fig_dict_key") or source.get("figure") or "").strip()
    chart_id = str(source.get("chart_id") or figure_chart_id(index, fig_dict_key)).strip()
    resolved_title = str(title or source.get("title") or fig_dict_key or chart_id).strip()
    resolved_desc = str(description or source.get("desc") or source.get("analysis") or "").strip()
    try:
        generation_order = int(source.get("generation_order") if source.get("generation_order") is not None else index)
    except Exception:
        generation_order = index

    artifact = dict(source)
    artifact.update(
        {
            "chart_id": chart_id,
            "candidate_id": str(source.get("candidate_id") or fig_dict_key or chart_id),
            "fig_dict_key": fig_dict_key or chart_id,
            "title": resolved_title,
            "desc": resolved_desc,
            "description": resolved_desc,
            "stage": str(source.get("stage") or "visualization"),
            "section_scope": str(source.get("section_scope") or "visualization"),
            "render_status": str(source.get("render_status") or "success"),
            "generation_order": generation_order,
        }
    )
    if language:
        artifact["language"] = language
    elif source.get("language"):
        artifact["language"] = source.get("language")
    return artifact


def successful_figure_artifacts(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    artifacts: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if str(item.get("render_status") or "success").lower() != "success":
            continue
        artifacts.append(normalize_figure_artifact(item, index))
    return artifacts


def report_figure_artifact_metadata(items: Any) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for item in successful_figure_artifacts(items):
        metadata.append(
            {
                "chart_id": item.get("chart_id"),
                "candidate_id": item.get("candidate_id"),
                "fig_dict_key": item.get("fig_dict_key"),
                "title": item.get("title"),
                "description": item.get("description") or item.get("desc"),
                "desc": item.get("description") or item.get("desc"),
                "stage": item.get("stage") or "visualization",
                "section_scope": item.get("section_scope") or "visualization",
                "render_status": item.get("render_status") or "success",
                "generation_order": item.get("generation_order"),
                "language": item.get("language"),
            }
        )
    return metadata


def figure_artifact_contexts(artifacts: Any) -> dict[int, str]:
    contexts: dict[int, str] = {}
    for index, item in enumerate(successful_figure_artifacts(artifacts)):
        title = str(item.get("title") or "").strip()
        desc = str(item.get("description") or item.get("desc") or "").strip()
        parts = [f"[FIG:{index}]"]
        if title:
            parts.append(f"Title: {title}")
        if desc:
            parts.append(desc)
        contexts[index] = "\n".join(parts)
    return contexts
