"""Stage 2 of a quote (pure, sync): verify the optimizer's output and build the decision.

The engine is authoritative. It re-checks every evaluated price against the hard constraints and
the conversion floor, recomputes expected revenue/profit, and selects the argmax itself. If the
optimizer picked something else, the engine's choice wins and a warning is recorded.
"""

from __future__ import annotations

from typing import Optional

from app.schemas.optimizer import OptimizerResponse
from app.schemas.pricing import DecisionCandidate, PricingDecision, Provenance
from app.schemas.pricing_parts import CheckedCandidate

from .constraints import check_price, evaluate_constraints
from .explain import business_explanation, customer_explanation
from .hashing import sha256_hex
from .money import round_to
from .prepare import Prepared

PRICE_TOL = 1e-6


def decision_id_for(prepared: Prepared, model_version: Optional[str]) -> str:
    payload = {
        "request": prepared.request,
        "version_id": prepared.strategy_version_id,
        "model_version": model_version or "none",
    }
    return "D-" + sha256_hex(payload)[:16]


def _find(cands: list[CheckedCandidate], price: float) -> Optional[CheckedCandidate]:
    return next((c for c in cands if abs(c.price - price) <= PRICE_TOL), None)


def finalize(prepared: Prepared, response: Optional[OptimizerResponse], created_at: Optional[str] = None) -> PricingDecision:
    warnings = list(prepared.warnings)
    facts = prepared.facts
    policy = prepared.optimizer_policy
    if response is None and prepared.status == "OK":
        raise ValueError("an optimizer response is required when eligible candidates exist")
    if response is not None:
        warnings.extend(response.warnings)

    evals = {}
    if response is not None:
        for e in response.evaluations:
            cand = _find(prepared.candidates, e.price)
            if cand is None:
                warnings.append(f"OPTIMIZER_UNKNOWN_CANDIDATE:{e.price:g}")
                continue
            evals[cand.price] = e

    rows: list[DecisionCandidate] = []
    for c in prepared.candidates:
        # Re-verify eligibility independently of what the engine computed earlier or the optimizer claims.
        violations = check_price(c.price, prepared.constraints, facts)
        eligible = not violations
        e = evals.get(c.price)
        if e is not None and e.eligible and not eligible:
            warnings.append(f"OPTIMIZER_ELIGIBILITY_MISMATCH:{c.price:g}")
        p_buy = e.p_buy if e is not None else None
        revenue = round_to(p_buy * c.price, 6) if p_buy is not None else None
        profit = round_to(p_buy * (c.price - prepared.unit_cost), 6) if p_buy is not None else None
        if not eligible:
            status = "INELIGIBLE"
        elif p_buy is None:
            status = "NOT_SELECTED"
            warnings.append(f"OPTIMIZER_MISSING_EVALUATION:{c.price:g}")
        elif p_buy < policy.conversion_floor:
            status = "BELOW_CONVERSION_FLOOR"
        else:
            status = "NOT_SELECTED"
        rows.append(DecisionCandidate(
            price=c.price, reasons=c.reasons, eligible=eligible, violations=violations, p_buy=p_buy,
            expected_revenue=revenue, expected_profit=profit, selection_status=status,
            explanation=e.explanation if e is not None else "",
        ))

    metric = "expected_profit" if policy.objective == "max_expected_profit" else "expected_revenue"
    pool = [r for r in rows if r.selection_status == "NOT_SELECTED" and r.p_buy is not None]
    best: Optional[DecisionCandidate] = None
    for r in sorted(pool, key=lambda x: x.price):  # ascending, so ties keep the lower price
        if best is None or getattr(r, metric) > getattr(best, metric) + 1e-12:
            best = r

    status = prepared.status
    reason = prepared.status_reason
    final_price: Optional[float] = None
    if best is not None and status == "OK":
        # Final authoritative check before anything is returned.
        if check_price(best.price, prepared.constraints, facts):
            status, reason = "NO_VALID_PRICE", "selected price failed the final constraint check"
        else:
            best.selection_status = "SELECTED"
            final_price = best.price
    elif status == "OK":
        status = "NO_VALID_PRICE"
        if any(r.selection_status == "BELOW_CONVERSION_FLOOR" for r in rows):
            reason = f"no eligible candidate reaches the conversion floor ({policy.conversion_floor:.0%})"
        else:
            reason = "no eligible candidate was evaluated"

    if response is not None and response.selected_price is not None:
        if final_price is None or abs(response.selected_price - final_price) > PRICE_TOL:
            warnings.append(
                f"OPTIMIZER_SELECTION_OVERRIDDEN: optimizer chose {response.selected_price:g}, "
                f"engine selected {'none' if final_price is None else f'{final_price:g}'}"
            )

    model = response.model if response is not None else None
    constraints_evaluated = (
        evaluate_constraints(final_price, prepared.constraints, facts) if final_price is not None else []
    )
    decision = PricingDecision(
        decision_id=decision_id_for(prepared, model.version if model else None),
        domain_id=prepared.domain_id,
        product_id=prepared.product_id,
        strategy_version_id=prepared.strategy_version_id,
        version_number=prepared.version_number,
        status=status,
        status_reason=reason,
        final_price=final_price,
        currency=prepared.currency,
        unit_label=prepared.unit_label,
        unit_prefix=prepared.unit_prefix,
        unit_suffix=prepared.unit_suffix,
        decimals=prepared.decimals,
        objective=policy.objective,
        reference_price=prepared.reference_price,
        anchor_price=prepared.anchor_price,
        bounds=prepared.bounds,
        applied_rules=prepared.applied_rules,
        overridden_rules=prepared.overridden_rules,
        skipped_rules=prepared.skipped_rules,
        constraints_evaluated=constraints_evaluated,
        candidates=rows,
        provenance=Provenance(
            optimizer=model,
            llm_generated_strategy=prepared.strategy_source == "llm",
            simulated=model is None or model.kind == "simulated" or model.training_data == "synthetic",
        ),
        signals=prepared.signals,
        context_snapshot=prepared.context,
        request=prepared.request,
        warnings=warnings,
        created_at=created_at,
    )
    decision.explanation.business = business_explanation(prepared, decision)
    decision.explanation.customer = customer_explanation(prepared, decision)
    return decision
