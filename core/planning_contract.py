from __future__ import annotations


PLANNING_STAGE_ORDER = (
    "loading_auto",
    "prep_auto",
    "vis_auto",
    "modeling_auto",
    "report_auto",
)

DEFAULT_STAGE_PLAN = {stage: True for stage in PLANNING_STAGE_ORDER}
