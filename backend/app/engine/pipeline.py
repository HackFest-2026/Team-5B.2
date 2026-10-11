"""Apply rule actions in order, with decimal-safe arithmetic and a logged step for each one."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from typing import Any, Mapping, Optional, Sequence

from app.schemas.pricing_parts import AppliedRule
from app.schemas.rules import Rule
from app.schemas.strategy import CombineMode

from .expression import ExpressionError, evaluate_expression
from .money import _CTX, D, norm

CAP_RULE_ID = "policy:max_total_adjustment"


@dataclass
class RuleError:
    rule_id: str
    message: str


@dataclass
class ApplyResult:
    anchor: float
    breakdown: list[AppliedRule] = field(default_factory=list)
    errors: list[RuleError] = field(default_factory=list)


def _signed(v: Decimal) -> str:
    s = format(v.normalize(), "f")
    return s if v < 0 else f"+{s}"


def apply_rules(base: float, rules: Sequence[Rule], combine: CombineMode, ctx: Mapping[str, Any]) -> ApplyResult:
    """Apply actions in the given order.

    combine:
      - multiplicative: percent adjustments compound on the running price.
      - additive: percent adjustments are taken against the base price, so they sum in any order.
    fixed / multiply / set / formula always act on the running price; ``none`` changes nothing.
    """
    with localcontext(_CTX):
        base_d = D(base)
        price = base_d
        out = ApplyResult(anchor=norm(base))
        for r in rules:
            before = price
            a = r.action
            v = D(a.value) if a.value is not None else Decimal(0)
            if a.type == "percent":
                ref = base_d if combine == "additive" else price
                price = price + ref * v / Decimal(100)
                reason = f"{_signed(v)}% of {'base' if combine == 'additive' else 'running'} price"
            elif a.type == "fixed":
                price = price + v
                reason = f"{_signed(v)} fixed adjustment"
            elif a.type == "multiply":
                price = price * v
                reason = f"× {format(v.normalize(), 'f')}"
            elif a.type == "set":
                price = v
                reason = f"set to target price {format(v.normalize(), 'f')}"
            elif a.type == "none":
                reason = "no adjustment"
            elif a.type == "formula":
                try:
                    res = evaluate_expression(a.expression or "", {**ctx, "price": float(price), "base": base})
                    if isinstance(res, bool) or not isinstance(res, (int, float)) or not math.isfinite(res):
                        raise ExpressionError(f"formula returned {res!r}")
                    price = D(float(res))
                    reason = f"formula {a.expression}"
                except ExpressionError as e:
                    out.errors.append(RuleError(r.id, str(e)))
                    continue
            else:  # pragma: no cover - schema rejects unknown actions
                continue
            out.breakdown.append(AppliedRule(
                rule_id=r.id, name=r.name, stage=r.stage,
                before=norm(before), after=norm(price), delta=norm(price - before), reason=reason,
            ))
        out.anchor = norm(price)
        return out


def cap_total_adjustment(base: float, result: ApplyResult, cap_pct: Optional[float]) -> ApplyResult:
    """Limit the net effect of all rules to ±cap_pct of base (an explicit, logged policy step)."""
    if not cap_pct:
        return result
    with localcontext(_CTX):
        hi = norm(D(base) * (1 + D(cap_pct) / 100))
        lo = norm(D(base) * (1 - D(cap_pct) / 100))
        capped = min(max(result.anchor, lo), hi)
        if capped != result.anchor:
            result.breakdown.append(AppliedRule(
                rule_id=CAP_RULE_ID, name="Maximum total adjustment",
                before=result.anchor, after=capped, delta=norm(D(capped) - D(result.anchor)),
                reason=f"net rule adjustment limited to ±{cap_pct:g}% of base price",
            ))
            result.anchor = capped
    return result
