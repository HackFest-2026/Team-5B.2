"""Strategy lifecycle: DRAFT → VALIDATED → APPROVED → ACTIVE (published) → RETIRED.

Drafts are mutable (DRAFT/VALIDATED). Approval freezes a validated draft into an immutable
version row (APPROVED). Publishing activates one version per domain; rollback re-activates a
retired one. Any transition not listed here is rejected with 409 INVALID_TRANSITION.
"""

from __future__ import annotations

from app.errors import ApiError

DRAFT_TRANSITIONS = {
    ("DRAFT", "VALIDATED"): "validate",
    ("VALIDATED", "DRAFT"): "edit",
    ("DRAFT", "DRAFT"): "edit",
    ("VALIDATED", "VALIDATED"): "validate",
}
VERSION_TRANSITIONS = {
    (None, "APPROVED"): "approve",
    ("APPROVED", "ACTIVE"): "publish",
    ("RETIRED", "ACTIVE"): "rollback",
    ("ACTIVE", "RETIRED"): "replace",
}


def check_version_transition(current: str | None, target: str, operation: str) -> None:
    allowed = VERSION_TRANSITIONS.get((current, target))
    if allowed is None or (operation in ("publish", "rollback") and allowed != operation):
        raise ApiError(409, "INVALID_TRANSITION", f"Cannot {operation} a version that is {current or 'new'}.",
                       {"from": current, "to": target, "operation": operation})


def check_draft_can_be_approved(status: str) -> None:
    if status != "VALIDATED":
        raise ApiError(409, "STRATEGY_NOT_VALIDATED", "Validate the current draft before approving it.",
                       {"from": status, "to": "APPROVED"})
