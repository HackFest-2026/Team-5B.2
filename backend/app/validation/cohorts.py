"""Named persona cohorts: buy/defer/reject shares per candidate price, plus an explainable concern.

Shares are the cohort's mean analytical choice probabilities (no sampling noise). The main concern
comes from decomposing the agents' buy utility into the simulator's documented terms; the most
negative term picks the concern. All output is simulated, advisory feedback.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Optional, Sequence

from app.schemas.domain import Persona
from app.schemas.persona import PersonaReaction

from .populations import cohort
from .sim import cas

DEFAULT_PERSONAS_FILE = Path(__file__).with_name("default_personas.json")

CONCERNS = {
    "price": ("The price is above what I expect to pay.", "Consider a smaller increase or a targeted offer for this segment."),
    "budget": ("This is over my budget.", "Offer a lower-priced alternative or a smaller package."),
    "competitor": ("A competitor is cheaper.", "Review the competitor gap; loyalty perks may offset it."),
    "value": ("It doesn't feel worth this price.", "Explain the value better or bundle extras."),
    "wait": ("I'd rather wait for a better deal.", "A limited-time or early-booking offer can convert this group."),
    "risk": ("Paying a premium feels risky.", "Flexible cancellation or a guarantee lowers the perceived risk."),
    "none": ("No major concern at this price.", None),
}


def default_personas() -> list[Persona]:
    return [Persona.model_validate(p) for p in json.loads(DEFAULT_PERSONAS_FILE.read_text(encoding="utf-8"))]


def utility_terms(policy, agent, offer) -> dict[str, float]:
    """Mirror of StandardDecisionPolicy.calculate_probabilities, split into its named terms.

    A test checks that the terms sum to the module's own buy utility.
    """
    t, s = agent.traits, agent.state
    r_eff = max(0.01, offer.base_price * (1.0 + s.reference_price_bias))
    r = offer.price / r_eff
    surplus = t.value_orientation * (offer.perceived_value - offer.price) / r_eff * policy.surplus_weight
    price = -t.price_sensitivity * (r - 1.0) * policy.price_sensitivity_weight * (policy.loss_aversion_factor if r > 1 else 1.0)
    budget_cap = offer.base_price * s.budget_ratio
    budget = (-(1.0 + t.risk_aversion) * (offer.price - budget_cap) / max(0.01, budget_cap) * policy.budget_penalty_weight
              if offer.price > budget_cap else 0.0)
    comp = 0.0
    if offer.competitor_price is not None:
        diff = (offer.competitor_price - offer.price) / r_eff
        comp = ((1.0 + 0.5 * t.loyalty) * diff if diff >= 0 else -max(0.2, 2.0 - t.loyalty) * abs(diff)) * policy.competitor_weight
    urgency = (offer.urgency * s.need_intensity * 1.5 + offer.scarcity * 1.0) * policy.urgency_weight
    risk = -0.5 * t.risk_aversion * (r - 1.0) if r > 1 else 0.0
    return {"intercept": policy.base_intercept, "value": surplus, "price": price, "budget": budget,
            "competitor": comp, "urgency": urgency, "risk": risk}


def _reaction(buy: float) -> str:
    return "DELIGHTED" if buy >= 0.75 else "ACCEPTS" if buy >= 0.5 else "HESITANT" if buy >= 0.25 else "REJECTS"


def evaluate_cohort(persona: Persona, offer_template, prices: Sequence[float], size: int, seed: int) -> list[PersonaReaction]:
    policy = cas.StandardDecisionPolicy()
    agents = cohort(persona, size, seed)
    out: list[PersonaReaction] = []
    for price in prices:
        offer = cas.Offer(price=float(price), base_price=offer_template.base_price, perceived_value=offer_template.perceived_value,
                          competitor_price=offer_template.competitor_price, urgency=offer_template.urgency,
                          scarcity=offer_template.scarcity, unit_cost=offer_template.unit_cost)
        probs = [policy.calculate_probabilities(a, offer) for a in agents]
        buy, defer, reject = (fmean(p[i] for p in probs) for i in range(3))
        terms = [utility_terms(policy, a, offer) for a in agents]
        means = {k: fmean(t[k] for t in terms) for k in ("value", "price", "budget", "competitor", "risk")}
        negatives = {k: v for k, v in means.items() if v < -0.05}
        driver: str
        if defer > max(reject, 0.3) and buy < 0.5:
            driver = "wait"
        elif negatives and buy < 0.75:
            driver = min(negatives, key=negatives.get)
        else:
            driver = "none"
        concern, suggestion = CONCERNS[driver]
        out.append(PersonaReaction(
            persona_id=persona.id, name=persona.name, description=persona.description, preferences=persona.preferences,
            price=float(price), buy_share=round(buy, 4), defer_share=round(defer, 4), reject_share=round(reject, 4),
            reaction=_reaction(buy), main_concern=concern, concern_driver=driver,  # type: ignore[arg-type]
            explanation=(f"Simulated cohort of {len(agents)} agents at {price:g}: {buy:.0%} buy, {defer:.0%} wait, "
                         f"{reject:.0%} walk away."),
            suggestion=suggestion,
        ))
    return out


def persona_grid(personas: Sequence[Persona], offer_template, prices: Sequence[float], size: int, seed: int,
                 only: Optional[Sequence[str]] = None) -> list[list[PersonaReaction]]:
    chosen = [p for p in personas if not only or p.id in only]
    return [evaluate_cohort(p, offer_template, prices, size, seed + 101 * i) for i, p in enumerate(chosen)]
