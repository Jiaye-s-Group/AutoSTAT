from __future__ import annotations

import re

from core.llm_client import chat
from core.prompt_template import render_file


_INTERACTIVE_TAIL_PATTERN = re.compile(
    r"(?:"
    r"请问|请告知我|请告诉我|告诉我您的|"
    r"(?:您|你)希望(?:从|先|选择)|"
    r"(?:请|可否)(?:您|你)?(?:从中)?选择|"
    r"等待(?:您|你)(?:的)?(?:回复|确认|选择)|"
    r"(?:若|如)(?:您|你).{0,120}(?:可随时|欢迎|请补充|请提出)|"
    r"(?:would you like|which .{0,80} would you|let me know|"
    r"please tell me|tell me (?:which|what)|please choose)"
    r")",
    re.IGNORECASE,
)


def normalize_suggestion_output(text: str) -> str:
    """Remove only trailing invitations that incorrectly pause an executable suggestion."""
    lines = (text or "").strip().splitlines()
    while lines:
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            break
        match = _INTERACTIVE_TAIL_PATTERN.search(lines[-1])
        if not match:
            break
        prefix = lines[-1][:match.start()].rstrip()
        sentence_end = max((prefix.rfind(mark) for mark in "。！？.!?"), default=-1)
        if sentence_end >= 0:
            lines[-1] = prefix[:sentence_end + 1]
            break
        lines.pop()
    return "\n".join(lines).rstrip()


def revise_suggestion(
    *,
    stage_label: str,
    original_requirements: str,
    current_suggestion: str,
    revision_instruction: str,
    hard_constraints: str = "",
    language_instruction: str = "",
) -> str:
    ctx = {
        "stage_label": stage_label,
        "original_requirements": original_requirements or "",
        "current_suggestion": current_suggestion or "",
        "revision_instruction": revision_instruction or "",
        "hard_constraints": hard_constraints or "",
        "language_instruction": language_instruction or "",
    }
    system_prompt = render_file("shared/revise_suggestion_llm_sys.txt", ctx, strict=True)
    user_prompt = render_file("shared/revise_suggestion_llm_user.txt", ctx, strict=True)
    revised = chat(system_prompt, user_prompt, name=f"suggestion.revise.{stage_label}")
    return normalize_suggestion_output(revised)
