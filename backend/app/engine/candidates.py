"""Candidate-price generation (separate from evaluation) and eligibility checks.

Candidates come from the strategy: offsets around the rule-adjusted anchor, optionally the base
price, and (only when the anchor breaks a bound) the nearest bound. They are rounded, de-duplicated
with their reasons merged, and checked against every hard constraint. Nothing is clamped:
candidates that break a constraint stay visible, marked ineligible with the reason.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import Optional, Sequence

from app.schemas.constraints import Constraint
from app.schemas.pricing_parts import Bounds, CheckedCandidate, GeneratedCandidate
from app.schemas.strategy import CandidatePolicy

from .constraints import PriceFacts, check_price
from .money import _CTX, D, norm, round_to, round_to_step

# Lower number = kept first when trimming to the configured count.
_P_ANCHOR, _P_PIN, _P_BASE, _P_OFFSET = 0, 1, 2, 3


def _round(price: float, step: Optional[float], decimals: int) -> float:
    p = round_to_step(price, step) if step else price
    return round_to(p, decimals)


def _pin(bound: float, direction: str, step: Optional[float], decimals: int) -> float:
    """Round a bound pin without leaving the bound (lower pins round up, upper pins round down)."""
    p = _round(bound, step, decimals)
    unit = step if step else 10 ** -decimals
    with localcontext(_CTX):
        if direction == "lower" and p < bound - 1e-9:
            p = round_to(D(p) + D(unit), decimals)
        elif direction == "upper" and p > bound + 1e-9:
            p = round_to(D(p) - D(unit), decimals)
    return p


def generate_candidates(
    anchor: float,
    base: float,
    bounds: Bounds,
    policy: CandidatePolicy,
    step: Optional[float],
    decimals: int,
) -> list[GeneratedCandidate]:
    raw: list[tuple[float, str, float]] = []  # (price, reason, priority)
    with localcontext(_CTX):
        raw.append((_round(anchor, step, decimals), "rule-adjusted anchor", _P_ANCHOR))
        for o in policy.offsets_pct:
            if o == 0:
                continue
            p = norm(D(anchor) * (1 + D(o) / Decimal(100)))
            raw.append((_round(p, step, decimals), f"anchor {o:+g}%", _P_OFFSET + abs(o) / 1000))
    if policy.include_base_price:
        raw.append((_round(base, step, decimals), "reference (base) price", _P_BASE))
    if policy.include_bound_pins and not bounds.conflict:
        if bounds.lower is not None and anchor < bounds.lower:
            raw.append((_pin(bounds.lower, "lower", step, decimals),
                        f"pinned to lower bound ({', '.join(bounds.lower_by)})", _P_PIN))
        if bounds.upper is not None and anchor > bounds.upper:
            raw.append((_pin(bounds.upper, "upper", step, decimals),
                        f"pinned to upper bound ({', '.join(bounds.upper_by)})", _P_PIN))

    merged: dict[float, tuple[list[str], float]] = {}
    for price, reason, prio in raw:
        reasons, best = merged.get(price, ([], prio))
        if reason not in reasons:
            reasons.append(reason)
        merged[price] = (reasons, min(best, prio))

    kept = sorted(merged.items(), key=lambda kv: (kv[1][1], kv[0]))[: policy.count]
    return [GeneratedCandidate(price=p, reasons=r) for p, (r, _) in sorted(kept, key=lambda kv: kv[0])]


def check_candidates(
    candidates: Sequence[GeneratedCandidate], constraints: Sequence[Constraint], facts: PriceFacts
) -> list[CheckedCandidate]:
    out: list[CheckedCandidate] = []
    for c in candidates:
        violations = check_price(c.price, constraints, facts)
        out.append(CheckedCandidate(price=c.price, reasons=c.reasons, eligible=not violations, violations=violations))
    return out
