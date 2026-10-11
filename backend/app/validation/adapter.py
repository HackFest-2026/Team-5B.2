"""Build the simulator's domain-neutral Offer from an engine decision (same mapping for every domain)."""

from __future__ import annotations

from typing import Optional

from app.schemas.pricing import PricingDecision

from .sim import cas


def _unit(v: Optional[float]) -> tuple[float, bool]:
    if v is None:
        return 0.0, False
    return min(max(float(v), 0.0), 1.0), not (0.0 <= v <= 1.0)


def to_offer(decision: PricingDecision, price: Optional[float] = None) -> tuple[object, list[str]]:
    """Return (Offer, mapping warnings). Competitor price stays None when unmapped (never invented)."""
    sig = decision.signals
    warnings: list[str] = []
    perceived = sig.get("perceivedValue")
    if perceived is None or perceived <= 0:
        perceived = decision.reference_price
        warnings.append("perceived value not mapped; using the reference price")
    urgency, clipped_u = _unit(sig.get("urgency"))
    scarcity, clipped_s = _unit(sig.get("scarcity"))
    if clipped_u or clipped_s:
        warnings.append("urgency/scarcity clamped to [0, 1]")
    competitor = sig.get("competitorPrice")
    offer = cas.Offer(
        price=float(price if price is not None else decision.final_price),
        base_price=decision.reference_price,
        perceived_value=float(perceived),
        competitor_price=float(competitor) if competitor and competitor > 0 else None,
        urgency=urgency,
        scarcity=scarcity,
        unit_cost=float(sig.get("unitCost", 0.0)),
        context={"domain_id": decision.domain_id, "version": decision.strategy_version_id},
    )
    return offer, warnings


def ml_ladder_check(decision: PricingDecision, tolerance: float = 0.02) -> tuple[bool, list[dict]]:
    """Check F: the optimizer's P(buy) should not rise as price rises across the candidate set."""
    pts = sorted((c.price, c.p_buy) for c in decision.candidates if c.p_buy is not None)
    inversions = [
        {"from_price": a[0], "to_price": b[0], "from_p": round(a[1], 4), "to_p": round(b[1], 4)}
        for a, b in zip(pts, pts[1:]) if b[1] > a[1] + tolerance
    ]
    return not inversions, inversions
