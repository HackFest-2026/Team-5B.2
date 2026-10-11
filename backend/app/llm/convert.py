"""Convert the flat LLM output into a StrategyConfig (assign ids, build condition trees)."""

from __future__ import annotations

from typing import Any

from app.schemas.llm import LlmCondition, LlmStrategyOut


def _condition(c: LlmCondition) -> dict[str, Any]:
    leaves: list[dict[str, Any]] = []
    for leaf in c.conditions:
        node = {"attr": leaf.attr, "op": leaf.op, "value": leaf.value}
        leaves.append({"not": node} if leaf.negate else node)
    if not leaves:
        return {"all": []}
    if len(leaves) == 1:
        return leaves[0]
    return {c.match: leaves}


def to_strategy_dict(out: LlmStrategyOut, domain_id: str, provider: str, now: str) -> dict[str, Any]:
    prefix = "MOCK" if provider == "mock" else "LLM"
    rules = []
    for i, r in enumerate(out.rules, 1):
        action: dict[str, Any] = {"type": r.action.type}
        if r.action.type != "none":
            action["value"] = r.action.value
        rules.append({
            "id": f"{prefix}-{i:02d}",
            "name": r.name,
            "description": r.description,
            "condition": _condition(r.condition),
            "action": action,
            "priority": r.priority,
            "stage": r.stage,
            "customer_message": r.customer_message,
            "created_by": f"{provider}-generator",
            "created_at": now,
        })
    return {
        "domain_id": domain_id,
        "name": out.name,
        "objective_text": out.objective_text,
        "rules": rules,
        "constraints": [c.model_dump() for c in out.constraints],
        "conflict_policy": out.conflict_policy.model_dump(),
        "optimizer_policy": out.optimizer_policy.model_dump(),
        "source": "mock" if provider == "mock" else "llm",
    }
