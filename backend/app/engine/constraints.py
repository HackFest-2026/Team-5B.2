"""Hard constraints: bounds, eligibility checks and the constraint report for a final price.

Constraints are authoritative. Nothing here clamps a price into range: a price that breaks a
constraint is reported as a violation, and if no price satisfies every constraint the decision
is NO_VALID_PRICE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Optional, Sequence

from app.schemas.constraints import Constraint
from app.schemas.pricing_parts import Bounds, ConstraintEval, Violation

from .money import _CTX, D, is_multiple_of, norm

EPS = 1e-9


@dataclass(frozen=True)
class PriceFacts:
    base: float
    cost: float
    previous_price: Optional[float] = None
    allow_below_cost: bool = False


@dataclass(frozen=True)
class BoundSpec:
    name: str  # constraint type, or the implicit "positive_price" / "cost_floor"
    value: Optional[float]
    kind: str  # lower | upper | step
    bound: Optional[float]
    strict: bool = False  # strict lower bound (price must be > bound)
    label: str = ""


def _label(c: Constraint) -> str:
    return {
        "floor": f"price floor {c.value:g}",
        "ceiling": f"price ceiling {c.value:g}",
        "margin_floor": f"gross margin ≥ {c.value:g}%",
        "markup_floor": f"markup ≥ {c.value:g}%",
        "max_discount": f"max discount {c.value:g}% below base",
        "max_increase": f"max increase {c.value:g}% above base",
        "max_daily_change": f"max change {c.value:g}% vs previous price",
        "rounding": f"rounding step {c.value:g}",
    }[c.type]


def bound_specs(constraints: Sequence[Constraint], f: PriceFacts) -> list[BoundSpec]:
    """Every bound the constraints impose for these facts, including the implicit ones."""
    with localcontext(_CTX):
        out = [BoundSpec("positive_price", None, "lower", 0.0, strict=True, label="price must be positive")]
        if not f.allow_below_cost:
            out.append(BoundSpec("cost_floor", None, "lower", norm(f.cost), label=f"price ≥ unit cost {f.cost:g}"))
        for c in constraints:
            v, lbl = c.value, _label(c)
            pct = D(v) / Decimal(100)
            if c.type == "floor":
                out.append(BoundSpec(c.type, v, "lower", norm(v), label=lbl))
            elif c.type == "ceiling":
                out.append(BoundSpec(c.type, v, "upper", norm(v), label=lbl))
            elif c.type == "margin_floor":
                b = math.inf if v >= 100 else norm(D(f.cost) / (1 - pct))
                out.append(BoundSpec(c.type, v, "lower", b, label=lbl))
            elif c.type == "markup_floor":
                out.append(BoundSpec(c.type, v, "lower", norm(D(f.cost) * (1 + pct)), label=lbl))
            elif c.type == "max_discount":
                out.append(BoundSpec(c.type, v, "lower", norm(D(f.base) * (1 - pct)), label=lbl))
            elif c.type == "max_increase":
                out.append(BoundSpec(c.type, v, "upper", norm(D(f.base) * (1 + pct)), label=lbl))
            elif c.type == "max_daily_change":
                if f.previous_price:
                    out.append(BoundSpec(c.type, v, "lower", norm(D(f.previous_price) * (1 - pct)), label=lbl))
                    out.append(BoundSpec(c.type, v, "upper", norm(D(f.previous_price) * (1 + pct)), label=lbl))
            elif c.type == "rounding":
                out.append(BoundSpec(c.type, v, "step", v, label=lbl))
        return out


def compute_bounds(constraints: Sequence[Constraint], f: PriceFacts) -> Bounds:
    lower: Optional[float] = None
    upper: Optional[float] = None
    lower_by: list[str] = []
    upper_by: list[str] = []
    for s in bound_specs(constraints, f):
        if s.kind == "lower" and s.bound is not None:
            if lower is None or s.bound > lower + EPS:
                lower, lower_by = s.bound, [s.name]
            elif abs(s.bound - lower) <= EPS and s.name not in lower_by:
                lower_by.append(s.name)
        elif s.kind == "upper" and s.bound is not None:
            if upper is None or s.bound < upper - EPS:
                upper, upper_by = s.bound, [s.name]
            elif abs(s.bound - upper) <= EPS and s.name not in upper_by:
                upper_by.append(s.name)
    conflict = lower is not None and upper is not None and (lower > upper + EPS or math.isinf(lower))
    if lower is not None and math.isinf(lower):
        conflict = True
    return Bounds(
        lower=None if lower is None or math.isinf(lower) else lower,
        upper=upper, lower_by=lower_by, upper_by=upper_by, conflict=conflict,
    )


def _passes(price: float, s: BoundSpec) -> bool:
    if s.kind == "lower":
        if s.bound is None:
            return True
        return price > s.bound + EPS if s.strict else price >= s.bound - EPS
    if s.kind == "upper":
        return s.bound is None or price <= s.bound + EPS
    return s.bound is None or is_multiple_of(price, s.bound)


def _message(price: float, s: BoundSpec, passed: bool) -> str:
    if s.kind == "step":
        return f"{price:g} is a multiple of {s.bound:g}" if passed else f"{price:g} is not a multiple of {s.bound:g}"
    if s.bound is not None and math.isinf(s.bound):
        return f"{s.label}: impossible to satisfy"
    rel = (">" if s.strict else "≥") if s.kind == "lower" else "≤"
    status = "ok" if passed else "violated"
    return f"{s.label}: requires price {rel} {s.bound:g} ({status})"


def check_price(price: float, constraints: Sequence[Constraint], f: PriceFacts) -> list[Violation]:
    return [
        Violation(constraint=s.name, message=_message(price, s, False))
        for s in bound_specs(constraints, f)
        if not _passes(price, s)
    ]


def evaluate_constraints(price: float, constraints: Sequence[Constraint], f: PriceFacts) -> list[ConstraintEval]:
    out: list[ConstraintEval] = []
    for s in bound_specs(constraints, f):
        ok = _passes(price, s)
        out.append(ConstraintEval(
            constraint=s.name, value=s.value, kind=s.kind,  # type: ignore[arg-type]
            bound=None if s.bound is not None and math.isinf(s.bound) else s.bound,
            passed=ok, message=_message(price, s, ok),
        ))
    return out


def gross_margin_pct(price: float, cost: float) -> Optional[float]:
    return None if price <= 0 else (price - cost) / price * 100


def markup_pct(price: float, cost: float) -> Optional[float]:
    return None if cost <= 0 else (price - cost) / cost * 100
