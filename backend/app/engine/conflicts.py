"""Conflict resolution between matching rules, driven only by the strategy's ConflictPolicy."""

from __future__ import annotations

from typing import Sequence

from app.schemas.pricing_parts import OverriddenRule
from app.schemas.rules import Rule
from app.schemas.strategy import ConflictMode


def resolve_conflicts(
    fired: Sequence[Rule], mode: ConflictMode, pipeline: Sequence[str]
) -> tuple[list[Rule], list[OverriddenRule]]:
    """Return (applied, overridden). ``fired`` must already be in deterministic order."""
    if mode == "stack" or len(fired) <= 1:
        return list(fired), []

    applied: list[Rule] = []
    overridden: list[OverriddenRule] = []

    if mode == "first_match_per_stage":
        winner_by_stage: dict[str, Rule] = {}
        for r in fired:
            win = winner_by_stage.get(r.stage)
            if win is None:
                winner_by_stage[r.stage] = r
                applied.append(r)
            else:
                overridden.append(OverriddenRule(
                    rule_id=r.id,
                    overridden_by=win.id,
                    reason=f'stage "{r.stage}" applies only its first matching rule: "{win.name}" '
                           f"(priority {win.priority}) takes precedence",
                ))
        return applied, overridden

    # highest_priority_wins: the single matching rule with the lowest priority number;
    # ties broken by stage order, then rule id.
    index = {s: i for i, s in enumerate(pipeline)}
    win = min(fired, key=lambda r: (r.priority, index.get(r.stage, len(index)), r.id))
    for r in fired:
        if r is win:
            applied.append(r)
        else:
            overridden.append(OverriddenRule(
                rule_id=r.id,
                overridden_by=win.id,
                reason=f'only the highest-priority matching rule applies: "{win.name}" (priority {win.priority})',
            ))
    return applied, overridden
