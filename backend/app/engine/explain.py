"""Plain-English read-back of rules and decisions.

Two audiences: the business explanation is complete (rules, constraints, optimizer comparison);
the customer explanation uses only the rules' customer messages and never mentions cost, margin,
purchase probability or rule ids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from app.schemas.constraints import Constraint
from app.schemas.domain import Attribute, OutputUnit
from app.schemas.rules import AllCondition, AnyCondition, Condition, LeafCondition, NotCondition, Rule, RuleAction

from .expression import js_str

if TYPE_CHECKING:  # pragma: no cover
    from app.schemas.pricing import PricingDecision

    from .prepare import Prepared

OP_WORDS = {
    ">": "is more than", ">=": "is at least", "<": "is less than", "<=": "is at most",
    "==": "is", "!=": "is not", "in": "is one of", "not_in": "is not one of", "between": "is between",
}


def format_unit(value: float, unit: OutputUnit) -> str:
    return f"{unit.prefix}{value:,.{unit.decimals}f}{unit.suffix}"


def format_money(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def _label(key: str, attrs: Sequence[Attribute]) -> str:
    return next((a.label for a in attrs if a.key == key), key)


def _show(v) -> str:
    if isinstance(v, list):
        return ", ".join(js_str(x) for x in v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    return js_str(v)


def condition_to_english(c: Condition, attrs: Sequence[Attribute]) -> str:
    if isinstance(c, LeafCondition):
        label = _label(c.attr, attrs)
        attr = next((a for a in attrs if a.key == c.attr), None)
        if attr is not None and attr.type == "boolean" and c.op == "==":
            return f"{label} is {'yes' if c.value in (True, 'true') else 'no'}"
        if c.op == "between" and isinstance(c.value, list):
            return f"{label} is between {_show(c.value[0])} and {_show(c.value[1])}"
        return f"{label} {OP_WORDS.get(c.op, c.op)} {_show(c.value)}"
    if isinstance(c, AllCondition):
        return " and ".join(_wrap(s, attrs) for s in c.all) if c.all else "always"
    if isinstance(c, AnyCondition):
        return " or ".join(_wrap(s, attrs) for s in c.any) if c.any else "never"
    if isinstance(c, NotCondition):
        return f"not ({condition_to_english(c.not_, attrs)})"
    return "(invalid condition)"


def _wrap(c: Condition, attrs: Sequence[Attribute]) -> str:
    s = condition_to_english(c, attrs)
    return s if isinstance(c, (LeafCondition, NotCondition)) else f"({s})"


def action_to_english(a: RuleAction, unit: OutputUnit) -> str:
    v = a.value or 0
    if a.type == "percent":
        return f"decrease the price by {abs(v):g}%" if v < 0 else f"increase the price by {v:g}%"
    if a.type == "fixed":
        return f"subtract {format_unit(abs(v), unit)}" if v < 0 else f"add {format_unit(v, unit)}"
    if a.type == "multiply":
        return f"multiply the price by {v:g}"
    if a.type == "set":
        return f"set the price to {format_unit(v, unit)}"
    if a.type == "none":
        return "leave the price unchanged"
    return f"set the price to the formula `{a.expression or ''}`"


def rule_to_english(rule: Rule, attrs: Sequence[Attribute], unit: OutputUnit) -> str:
    cond = condition_to_english(rule.condition, attrs)
    s = f"{'Always' if cond == 'always' else f'If {cond}'}, {action_to_english(rule.action, unit)}"
    if rule.effective_from and rule.effective_to:
        s += f" (valid {rule.effective_from} to {rule.effective_to})"
    elif rule.effective_from:
        s += f" (from {rule.effective_from})"
    elif rule.effective_to:
        s += f" (until {rule.effective_to})"
    return s + "."


def constraint_to_english(c: Constraint, unit: OutputUnit) -> str:
    return {
        "floor": f"Price never goes below {format_unit(c.value, unit)}.",
        "ceiling": f"Price never goes above {format_unit(c.value, unit)}.",
        "margin_floor": f"Gross margin, (price - cost) / price, must be at least {c.value:g}%.",
        "markup_floor": f"Markup, (price - cost) / cost, must be at least {c.value:g}%.",
        "max_discount": f"Price can be at most {c.value:g}% below the base price.",
        "max_increase": f"Price can be at most {c.value:g}% above the base price.",
        "max_daily_change": f"Price may move at most {c.value:g}% from the previous price.",
        "rounding": f"Prices are rounded to the nearest {c.value:g}.",
    }[c.type]


def _p(prepared: "Prepared", value: float) -> str:
    """Price in the domain's own unit, e.g. ₹6,500 or 11.60%. Suffixes starting with a space (" /night") are omitted inline."""
    suffix = prepared.unit_suffix if prepared.unit_suffix and not prepared.unit_suffix.startswith(" ") else ""
    return f"{prepared.unit_prefix}{format_money(value, prepared.decimals)}{suffix}"


def business_explanation(prepared: "Prepared", decision: "PricingDecision") -> list[str]:
    d = prepared.decimals
    out = [f"Reference price {_p(prepared, prepared.reference_price)} ({prepared.unit_label})."]
    for a in prepared.applied_rules:
        out.append(f'"{a.name}": {a.reason} ({_p(prepared, a.before)} → {_p(prepared, a.after)}).')
    for o in prepared.overridden_rules:
        out.append(f'Rule "{o.rule_id}" matched but was overridden: {o.reason}.')
    out.append(f"Rule-adjusted anchor {_p(prepared, prepared.anchor_price)}.")
    rejected = [c for c in decision.candidates if not c.eligible]
    for c in rejected:
        out.append(f"Candidate {_p(prepared, c.price)} rejected: " + "; ".join(v.message for v in c.violations) + ".")
    floor = [c for c in decision.candidates if c.selection_status == "BELOW_CONVERSION_FLOOR"]
    for c in floor:
        out.append(f"Candidate {_p(prepared, c.price)} excluded: P(buy) {c.p_buy:.0%} is below the conversion floor.")
    if decision.status == "NO_VALID_PRICE":
        out.append(f"No valid price: {decision.status_reason}.")
        return out
    metric = "expected_profit" if decision.objective == "max_expected_profit" else "expected_revenue"
    label = "expected profit" if metric == "expected_profit" else "expected revenue"
    sel = next(c for c in decision.candidates if c.selection_status == "SELECTED")
    others = sorted(
        (c for c in decision.candidates if c.selection_status == "NOT_SELECTED" and getattr(c, metric) is not None),
        key=lambda c: getattr(c, metric), reverse=True,
    )
    model = decision.provenance.optimizer
    kind = "model-predicted" if model is not None and model.kind == "ml" else "simulated"
    text = (f"Selected {_p(prepared, sel.price)}: highest {kind} {label} "
            f"{_p(prepared, getattr(sel, metric))} (P(buy) {sel.p_buy:.0%})")
    if others:
        r = others[0]
        text += f", vs {_p(prepared, r.price)} at {_p(prepared, getattr(r, metric))} (P(buy) {r.p_buy:.0%})"
    out.append(text + ".")
    out.append("All hard constraints pass for the selected price.")
    return out


def customer_explanation(prepared: "Prepared", decision: "PricingDecision") -> list[str]:
    if decision.status == "NO_VALID_PRICE":
        return ["This option isn't available for the details provided."]
    msgs: list[str] = []
    for m in prepared.customer_messages:
        if m.message not in msgs:
            msgs.append(m.message)
    return msgs or ["Price based on current availability and demand."]
