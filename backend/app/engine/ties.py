"""Detect same-priority rules whose order (or which one wins) changes the price."""

from __future__ import annotations

from typing import Any, Optional

from app.schemas.domain import Attribute, DomainConfig, Product
from app.schemas.rules import AllCondition, AnyCondition, Condition, LeafCondition
from app.schemas.strategy import StrategyConfig

from .conditions import evaluate_condition
from .money import round_to_step
from .pipeline import apply_rules
from .reachability import numeric



def sample_context_for(conds: list[Condition], attrs: list[Attribute], product: Product, as_of: str = "2026-01-15") -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "as_of": as_of, "base_price": product.base_price, "cost": product.cost,
        "previous_price": product.previous_price, "product_id": product.id, "customer_id": None,
    }
    for a in attrs:
        ctx[a.key] = a.default if a.default is not None else product.inventory.get(a.key, _default_for(a))
    by_key = {a.key: a for a in attrs}

    def satisfy(c: Condition) -> None:
        if isinstance(c, LeafCondition):
            ctx[c.attr] = _satisfying_value(c, by_key.get(c.attr))
        elif isinstance(c, AllCondition):
            for s in c.all:
                satisfy(s)
        elif isinstance(c, AnyCondition) and c.any:
            satisfy(c.any[0])

    for c in conds:
        satisfy(c)
    return ctx


def _default_for(a: Attribute) -> Any:
    return {"number": a.min if a.min is not None else 0, "boolean": False,
            "enum": (a.allowed_values or [""])[0], "date": "2026-01-15"}.get(a.type, "")


def _satisfying_value(c: LeafCondition, attr: Optional[Attribute]) -> Any:
    v = c.value
    is_num = numeric(v) and not isinstance(v, list) and ((attr is not None and attr.type == "number") or not isinstance(v, str))
    if c.op == ">":
        return float(v) + 1 if is_num else f"{v}~"
    if c.op == "<":
        return float(v) - 1 if is_num else ""
    if c.op in (">=", "<=", "=="):
        return v
    if c.op in ("in", "between"):
        return v[0] if isinstance(v, list) else v
    if c.op in ("!=", "not_in"):
        excluded = [str(x) for x in (v if isinstance(v, list) else [v])]
        if attr is not None and attr.type == "boolean":
            return not (v is True or v == "true")
        other = next((x for x in (attr.allowed_values or []) if x not in excluded), None) if attr else None
        if other is not None:
            return other
        return float(excluded[0]) + 1 if is_num and numeric(excluded[0]) else f"{excluded[0]}_other"
    return v


def order_sensitive_ties(domain: DomainConfig, strategy: StrategyConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    product = domain.products[0] if domain.products else Product(id="sample", name="Sample", base_price=100, cost=60)
    active = sorted((r for r in strategy.rules if r.enabled and r.stage in domain.pipeline), key=lambda r: r.id)
    mode = strategy.conflict_policy.mode
    combine = strategy.conflict_policy.combine
    step = strategy.rounding_step()
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            if a.priority != b.priority or (mode != "highest_priority_wins" and a.stage != b.stage):
                continue
            ctx = sample_context_for([a.condition, b.condition], domain.attributes, product)
            if not (evaluate_condition(a.condition, ctx).ok and evaluate_condition(b.condition, ctx).ok):
                continue
            first, second = ([a, b], [b, a]) if mode == "stack" else ([a], [b])
            p1 = round_to_step(apply_rules(product.base_price, first, combine, ctx).anchor, step)
            p2 = round_to_step(apply_rules(product.base_price, second, combine, ctx).anchor, step)
            if p1 != p2:
                winner = a.id  # ids sort ascending, so `a` is applied first / wins
                out.append({
                    "message": (f'"{a.name}" and "{b.name}" share priority {a.priority}'
                                f'{"" if mode == "highest_priority_wins" else f" in stage {a.stage!r}"}, and which one '
                                f"{'applies first' if mode == 'stack' else 'wins'} changes the price ({p1:g} vs {p2:g}). "
                                f'It is resolved deterministically by rule id ("{winner}" first); give them distinct priorities to make the intent explicit.'),
                    "rule_ids": [a.id, b.id],
                    "details": {
                        "sample_context": {k: v for k, v in ctx.items()},
                        "product_id": product.id,
                        "order_a": {"rule_ids": [r.id for r in first], "price": p1},
                        "order_b": {"rule_ids": [r.id for r in second], "price": p2},
                    },
                })
    return out
