"""Customer-safe view of a decision: no cost, margin, P(buy), rule ids or validation internals."""

from __future__ import annotations

from app.schemas.pricing import Alternative, CustomerQuote, PricingDecision

from .explain import format_money


def to_customer_quote(decision: PricingDecision, show_alternatives: bool = True, max_alternatives: int = 2) -> CustomerQuote:
    decimals = decision.decimals
    alternatives = None
    if show_alternatives and decision.final_price is not None:
        others = [
            c.price for c in decision.candidates
            if c.selection_status == "NOT_SELECTED" and c.eligible and c.price != decision.final_price
        ]
        others.sort(key=lambda p: abs(p - decision.final_price))
        alternatives = [
            Alternative(price=p, label=f"{decision.unit_prefix}{format_money(p, decimals)}{decision.unit_suffix}")
            for p in sorted(others[:max_alternatives])
        ] or None
    return CustomerQuote(
        decision_id=decision.decision_id,
        status=decision.status,
        final_price=decision.final_price,
        currency=decision.currency,
        unit_label=decision.unit_label,
        unit_prefix=decision.unit_prefix,
        unit_suffix=decision.unit_suffix,
        decimals=decision.decimals,
        factors=list(decision.explanation.customer),
        alternatives=alternatives,
    )
