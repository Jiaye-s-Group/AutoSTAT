from __future__ import annotations

from copy import deepcopy
from typing import Any, MutableMapping
from uuid import uuid4

from utils.workflow_state import stable_fingerprint


STORE_KEY = "_suggestion_interactions"


def _new_state() -> dict[str, Any]:
    return {
        "status": "idle",
        "base_requirements": [],
        "active_suggestion": "",
        "versions": [],
        "confirmed_version": None,
        "confirmed_suggestion": "",
        "messages": [],
        "pending_initial_request": "",
        "pending_revision": "",
        "pending_code_revision": "",
        "active_suggestion_fingerprint": "",
        "confirmed_suggestion_fingerprint": "",
        "code_source_suggestion_fingerprint": "",
        "current_code_fingerprint": "",
        "executed_code_fingerprint": "",
        "execution_run_id": "",
        "code_status": "idle",
        "last_execution_error": "",
        "auto_repair_attempts": 0,
        "repair_notice": "",
        "repair_in_progress": False,
        "repair_exhausted_fingerprint": "",
        "validation_attempts": 0,
        "validated_code_fingerprint": "",
    }


def get_suggestion_state(session: MutableMapping[str, Any], stage: str) -> dict[str, Any]:
    store = session.setdefault(STORE_KEY, {})
    state = store.setdefault(stage, _new_state())
    for key, value in _new_state().items():
        state.setdefault(key, deepcopy(value))
    return state


def clear_suggestion_state(session: MutableMapping[str, Any], stage: str | None = None) -> None:
    if stage is None:
        session.pop(STORE_KEY, None)
        return
    store = session.get(STORE_KEY)
    if isinstance(store, dict):
        store.pop(stage, None)


def add_requirement(state: dict[str, Any], text: str) -> None:
    value = str(text or "").strip()
    if not value:
        return
    state["base_requirements"].append(value)
    state["messages"].append({"role": "user", "content": value, "kind": "requirement"})
    state["status"] = "collecting"


def queue_initial_request(state: dict[str, Any], text: str) -> str:
    value = str(text or "").strip()
    if value:
        add_requirement(state, value)
        state["pending_initial_request"] = value
    return value


def take_pending_initial_request(state: dict[str, Any]) -> str:
    return str(state.pop("pending_initial_request", "") or "").strip()


def base_requirements_text(state: dict[str, Any], default: str = "") -> str:
    requirements = [str(item).strip() for item in state.get("base_requirements") or [] if str(item).strip()]
    return "\n".join(requirements) or str(default or "").strip()


def revision_fallback_text(
    state: dict[str, Any],
    revision_instruction: str,
    *,
    default: str = "",
) -> str:
    """Build a safe regeneration request when the hidden revision context is gone.

    Suggestions can survive longer than their phase-1 context in Streamlit session
    state.  If the user sends a revision in that stale state, regenerating from
    the visible suggestion and the new message is safer than silently returning.
    """
    parts: list[str] = []
    base_text = base_requirements_text(state, default)
    if base_text:
        parts.append(base_text)

    active_suggestion = str(state.get("active_suggestion") or "").strip()
    if active_suggestion:
        parts.append(
            "当前页面已显示的建议 / Currently displayed suggestion:\n"
            f"{active_suggestion}"
        )

    revision_text = str(revision_instruction or "").strip()
    if revision_text:
        parts.append(
            "用户刚刚发送的修改意见 / Latest user revision:\n"
            f"{revision_text}"
        )

    return "\n\n".join(parts).strip() or revision_text or base_text


def add_revision_request(state: dict[str, Any], text: str) -> str:
    value = str(text or "").strip()
    if value:
        state["messages"].append({"role": "user", "content": value, "kind": "revision"})
        state["status"] = "revising"
    return value


def queue_revision_request(state: dict[str, Any], text: str) -> str:
    value = add_revision_request(state, text)
    if value:
        state["pending_revision"] = value
    return value


def queue_code_revision_request(state: dict[str, Any], text: str) -> str:
    value = str(text or "").strip()
    if value:
        state["messages"].append({"role": "user", "content": value, "kind": "code_revision"})
        state["pending_code_revision"] = value
    return value


def take_pending_code_revision(state: dict[str, Any]) -> str:
    return str(state.pop("pending_code_revision", "") or "").strip()


def take_pending_revision(state: dict[str, Any]) -> str:
    return str(state.pop("pending_revision", "") or "").strip()


def replace_active_suggestion(
    state: dict[str, Any],
    suggestion: str,
    *,
    revision_instruction: str = "",
) -> int:
    value = str(suggestion or "").strip()
    version = len(state.get("versions") or []) + 1
    state["versions"].append({
        "version": version,
        "suggestion": value,
        "revision_instruction": str(revision_instruction or "").strip(),
    })
    state["active_suggestion"] = value
    state["active_suggestion_fingerprint"] = stable_fingerprint(value)
    state["confirmed_version"] = None
    state["confirmed_suggestion"] = ""
    state["confirmed_suggestion_fingerprint"] = ""
    state["status"] = "awaiting_approval"
    state["messages"].append({
        "role": "assistant",
        "content": value,
        "kind": "suggestion",
        "version": version,
    })
    return version


def confirm_active_suggestion(state: dict[str, Any]) -> bool:
    value = str(state.get("active_suggestion") or "").strip()
    if not value:
        return False
    version = len(state.get("versions") or [])
    state["confirmed_version"] = version
    state["confirmed_suggestion"] = value
    state["confirmed_suggestion_fingerprint"] = stable_fingerprint(value)
    state["status"] = "confirmed"
    return True


def mark_current_code_source_suggestion(state: dict[str, Any]) -> None:
    state["code_source_suggestion_fingerprint"] = str(
        state.get("confirmed_suggestion_fingerprint")
        or state.get("active_suggestion_fingerprint")
        or ""
    )


def code_matches_current_suggestion(state: dict[str, Any]) -> bool:
    current_code = str(state.get("current_code_fingerprint") or "")
    if not current_code:
        return True
    current_suggestion = str(
        state.get("confirmed_suggestion_fingerprint")
        or state.get("active_suggestion_fingerprint")
        or ""
    )
    source_suggestion = str(state.get("code_source_suggestion_fingerprint") or "")
    if not current_suggestion or not source_suggestion:
        return True
    return current_suggestion == source_suggestion


def visible_messages(state: dict[str, Any]) -> list[dict[str, Any]]:
    return list(state.get("messages") or [])


def mark_code_draft(state: dict[str, Any], code: str) -> tuple[str, bool]:
    fingerprint = stable_fingerprint(str(code or ""))
    previous_fingerprint = state.get("current_code_fingerprint")
    if previous_fingerprint and previous_fingerprint != fingerprint:
        state["auto_repair_attempts"] = 0
        state["repair_exhausted_fingerprint"] = ""
        state["repair_notice"] = ""
    state["current_code_fingerprint"] = fingerprint
    is_current = bool(
        state.get("executed_code_fingerprint")
        and state.get("executed_code_fingerprint") == fingerprint
    )
    if state.get("executed_code_fingerprint") and not is_current:
        state["code_status"] = "stale"
    return fingerprint, is_current


def begin_code_execution(state: dict[str, Any], code: str) -> str:
    fingerprint, _ = mark_code_draft(state, code)
    run_id = uuid4().hex
    state["execution_run_id"] = run_id
    state["running_code_fingerprint"] = fingerprint
    state["code_status"] = "running"
    state["repair_notice"] = ""
    return run_id


def finish_code_execution(state: dict[str, Any], run_id: str, *, success: bool) -> bool:
    if not run_id or state.get("execution_run_id") != run_id:
        return False
    if success:
        state["executed_code_fingerprint"] = state.get("running_code_fingerprint") or ""
        state["code_status"] = "succeeded"
        state["last_execution_error"] = ""
        state["auto_repair_attempts"] = 0
        state["repair_notice"] = ""
    else:
        state["code_status"] = "failed"
    return True


def record_execution_failure(state: dict[str, Any], error: str) -> None:
    state["last_execution_error"] = str(error or "").strip()
    state["code_status"] = "failed"


def can_auto_repair(state: dict[str, Any], *, max_attempts: int = 5) -> bool:
    return (
        not bool(state.get("repair_in_progress"))
        and state.get("repair_exhausted_fingerprint") != state.get("current_code_fingerprint")
        and int(state.get("auto_repair_attempts") or 0) < max_attempts
    )


def record_auto_repair(state: dict[str, Any], code: str) -> None:
    attempts = int(state.get("auto_repair_attempts") or 0) + 1
    state["repair_notice"] = "ready"
    state["last_execution_error"] = ""
    mark_code_draft(state, code)
    mark_current_code_source_suggestion(state)
    state["auto_repair_attempts"] = attempts
    state["code_status"] = "ready"


def record_validated_code(state: dict[str, Any], code: str, *, attempts: int) -> None:
    fingerprint, _ = mark_code_draft(state, code)
    mark_current_code_source_suggestion(state)
    state["validated_code_fingerprint"] = fingerprint
    state["validation_attempts"] = int(attempts or 0)
    state["repair_in_progress"] = False
    state["repair_exhausted_fingerprint"] = ""
    state["auto_repair_attempts"] = 0
    state["last_execution_error"] = ""
    state["repair_notice"] = "validated"
    state["code_status"] = "ready"


def record_validation_failure(state: dict[str, Any], code: str, error: str, *, attempts: int) -> None:
    fingerprint, _ = mark_code_draft(state, code)
    state["validation_attempts"] = int(attempts or 0)
    state["repair_in_progress"] = False
    state["repair_exhausted_fingerprint"] = fingerprint
    state["auto_repair_attempts"] = max(5, int(attempts or 0))
    state["last_execution_error"] = str(error or "").strip()
    state["repair_notice"] = "exhausted"
    state["code_status"] = "failed"


def record_successful_code(state: dict[str, Any], code: str) -> None:
    fingerprint = stable_fingerprint(str(code or ""))
    state["current_code_fingerprint"] = fingerprint
    state["executed_code_fingerprint"] = fingerprint
    mark_current_code_source_suggestion(state)
    state["running_code_fingerprint"] = ""
    state["execution_run_id"] = ""
    state["code_status"] = "succeeded"
