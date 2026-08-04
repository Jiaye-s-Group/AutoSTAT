"""Chart-level execution contract for generated visualizations."""

from __future__ import annotations

import json
import re
from typing import Any


_CHART_BULLET_PREFIX = re.compile(
    r"^(?:[-*•]+|\d+[.)、]|[（(]\d+[）)]|图\s*\d+\s*[:：、.]?)\s*"
)


def contract_as_prompt(contract: dict[str, Any]) -> str:
    return json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True)


def _chart_specs(refined_suggestions: str) -> list[str]:
    specs: list[str] = []
    for raw_line in str(refined_suggestions or "").splitlines():
        line = _CHART_BULLET_PREFIX.sub("", raw_line.strip()).strip(" -•\t")
        if not line:
            continue
        if ":" in line or "：" in line:
            _, _, right = re.sub("：", ":", line).rpartition(":")
            values = [value.strip() for value in re.split(r"[，,；;、]", right) if value.strip()]
            specs.extend(values or [line])
        else:
            specs.append(line)
    return specs


def build_visualization_contract(
    *,
    visual_recommendation: str = "",
    refined_suggestions: str = "",
    user_input: str = "",
    add_preference: str = "",
) -> dict[str, Any]:
    """Assign stable ids to every finalised chart requirement.

    The ids are intentionally independent of source columns or a particular
    dataset.  A code generator may append a human-readable suffix, while the
    validator uses the stable prefix to verify that every required chart was
    attempted.
    """
    specs = _chart_specs(refined_suggestions)
    charts = [
        {"id": f"chart_{index:02d}", "spec": spec, "required": True}
        for index, spec in enumerate(specs, start=1)
    ]
    return {
        "version": 1,
        "charts": charts,
        "minimum_chart_count": max(1, len(charts)),
        "allow_partial_results": True,
        "source_text_present": bool(
            "\n".join([visual_recommendation, refined_suggestions, user_input, add_preference]).strip()
        ),
    }


def validate_visualization_result(
    *,
    figure_keys: list[Any],
    contract: dict[str, Any] | None,
) -> dict[str, Any]:
    contract = dict(contract or {})
    keys = [str(key) for key in figure_keys]
    charts = [item for item in contract.get("charts") or [] if isinstance(item, dict)]
    missing: list[dict[str, str]] = []
    for chart in charts:
        chart_id = str(chart.get("id") or "")
        if chart_id and not any(key == chart_id or key.startswith(chart_id + "__") for key in keys):
            missing.append({"id": chart_id, "spec": str(chart.get("spec") or "")})
    minimum = int(contract.get("minimum_chart_count") or 1)
    if len(keys) < minimum and not charts:
        missing.append({"id": "minimum_chart_count", "spec": f"at least {minimum} charts"})
    status = "complete" if keys and not missing else ("partial" if keys else "failed")
    return {
        "status": status,
        "missing_charts": missing,
        "generated_chart_ids": keys,
        "is_complete": status == "complete",
    }
