"""Strategy validation report (schema errors and semantic issues in one shape)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from .common import StrictModel

Severity = Literal["error", "warning", "info"]
IssueCode = Literal[
    "schema",
    "unknown_attribute",
    "unsupported_operator",
    "invalid_value",
    "floor_above_ceiling",
    "bounds_conflict",
    "discount_over_100",
    "invalid_constraint",
    "unreachable",
    "priority_tie",
    "formula",
    "formula_not_allowed",
    "ood_risk",
    "policy",
]


class ValidationIssue(StrictModel):
    id: str
    severity: Severity
    code: IssueCode
    message: str
    path: Optional[str] = None
    rule_ids: list[str] = Field(default_factory=list)
    details: Optional[dict[str, Any]] = None


class ValidationReport(StrictModel):
    ok: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    counts: dict[Severity, int] = Field(default_factory=lambda: {"error": 0, "warning": 0, "info": 0})

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]
