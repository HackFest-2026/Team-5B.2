"""Differences between two strategy versions (rules added/removed/changed, constraints, policies)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.schemas.constraints import Constraint
from app.schemas.rules import Rule
from app.schemas.strategy import StrategyConfig

from .hashing import canonical_json

RULE_FIELDS = ("name", "description", "condition", "action", "priority", "stage", "effective_from", "effective_to",
               "enabled", "customer_message")
POLICY_FIELDS = ("conflict_policy", "candidate_policy", "optimizer_policy", "validation_policy", "objective_text", "name")


class RuleChange(BaseModel):
    before: Rule
    after: Rule
    fields: list[str]


class SettingChange(BaseModel):
    field: str
    before: Any
    after: Any


class StrategyDiff(BaseModel):
    added: list[Rule] = Field(default_factory=list)
    removed: list[Rule] = Field(default_factory=list)
    changed: list[RuleChange] = Field(default_factory=list)
    constraints_before: list[Constraint] = Field(default_factory=list)
    constraints_after: list[Constraint] = Field(default_factory=list)
    constraints_changed: bool = False
    settings: list[SettingChange] = Field(default_factory=list)
    has_changes: bool = False


def _same(a: Any, b: Any) -> bool:
    return canonical_json(a) == canonical_json(b)


def diff_strategies(before: Optional[StrategyConfig], after: StrategyConfig) -> StrategyDiff:
    b_rules = {r.id: r for r in (before.rules if before else [])}
    a_rules = {r.id: r for r in after.rules}
    d = StrategyDiff(
        added=[r for r in after.rules if r.id not in b_rules],
        removed=[r for r in (before.rules if before else []) if r.id not in a_rules],
    )
    for r in after.rules:
        old = b_rules.get(r.id)
        if old is None:
            continue
        fields = [f for f in RULE_FIELDS if not _same(getattr(old, f), getattr(r, f))]
        if fields:
            d.changed.append(RuleChange(before=old, after=r, fields=fields))
    d.constraints_before = list(before.constraints) if before else []
    d.constraints_after = list(after.constraints)
    d.constraints_changed = not _same(d.constraints_before, d.constraints_after)
    if before is not None:
        for f in POLICY_FIELDS:
            if not _same(getattr(before, f), getattr(after, f)):
                d.settings.append(SettingChange(field=f, before=getattr(before, f), after=getattr(after, f)))
    d.has_changes = bool(d.added or d.removed or d.changed or d.settings or d.constraints_changed)
    return d
