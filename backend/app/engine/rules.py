"""Rule ordering and selection.

Ordering is fully deterministic: pipeline stage index, then priority (ascending), then rule id.
It never depends on creation time, list order or dictionary iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.schemas.pricing_parts import SkippedRule
from app.schemas.rules import Rule

from .conditions import evaluate_condition


def rule_sort_key(pipeline: Sequence[str]):
    index = {s: i for i, s in enumerate(pipeline)}

    def key(r: Rule) -> tuple[int, int, str]:
        return (index.get(r.stage, len(index)), r.priority, r.id)

    return key


def order_rules(rules: Sequence[Rule], pipeline: Sequence[str]) -> list[Rule]:
    return sorted(rules, key=rule_sort_key(pipeline))


@dataclass
class RuleSelection:
    fired: list[Rule] = field(default_factory=list)
    skipped: list[SkippedRule] = field(default_factory=list)


def select_rules(rules: Sequence[Rule], ctx: Mapping[str, Any], as_of: str, pipeline: Sequence[str]) -> RuleSelection:
    """Decide which rules match this context, with a reason for every rule that does not."""
    out = RuleSelection()
    for r in order_rules(rules, pipeline):
        if not r.enabled:
            out.skipped.append(SkippedRule(rule_id=r.id, reason="rule is disabled"))
            continue
        if r.stage not in pipeline:
            out.skipped.append(SkippedRule(rule_id=r.id, reason=f'stage "{r.stage}" is not in the pipeline'))
            continue
        if r.effective_from and as_of < r.effective_from:
            out.skipped.append(SkippedRule(rule_id=r.id, reason=f"not effective until {r.effective_from}"))
            continue
        if r.effective_to and as_of > r.effective_to:
            out.skipped.append(SkippedRule(rule_id=r.id, reason=f"expired on {r.effective_to}"))
            continue
        res = evaluate_condition(r.condition, ctx)
        if res.ok:
            out.fired.append(r)
        else:
            out.skipped.append(SkippedRule(rule_id=r.id, reason=res.reason))
    return out
