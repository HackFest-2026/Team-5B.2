"""Stage 1 of a quote (pure, sync): context → rules → conflicts → anchor → bounds → candidates.

The output (``Prepared``) is what gets sent to a PriceOptimizer. The engine then finishes the
decision in ``finalize``. Splitting here lets an async optimizer (an ML model or a remote
service) sit between the two halves without the engine knowing which one it is.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from app.schemas.common import StrictModel
from app.schemas.constraints import Constraint
from app.schemas.domain import DomainConfig, Product
from app.schemas.optimizer import OptimizerRequest
from app.schemas.pricing_parts import (
    AppliedRule,
    Bounds,
    CheckedCandidate,
    CustomerProfile,
    DecisionStatus,
    OverriddenRule,
    PriceRequest,
    SkippedRule,
)
from app.schemas.strategy import OptimizerPolicy, StrategyConfig, StrategySource

from .candidates import check_candidates, generate_candidates
from .conflicts import resolve_conflicts
from .constraints import PriceFacts, compute_bounds
from .context import InvalidInputError, enrich_context
from .pipeline import apply_rules, cap_total_adjustment
from .rules import select_rules
from .signals import map_signals


class CustomerMessage(StrictModel):
    rule_id: str
    message: str


class Prepared(StrictModel):
    domain_id: str
    product_id: str
    strategy_version_id: str
    version_number: Optional[int] = None
    status: DecisionStatus
    status_reason: Optional[str] = None
    request: PriceRequest
    context: dict[str, Any]
    signals: dict[str, float] = Field(default_factory=dict)
    signal_sources: dict[str, str] = Field(default_factory=dict)
    reference_price: float
    unit_cost: float
    previous_price: Optional[float] = None
    allow_below_cost: bool = False
    currency: str
    unit_label: str = ""
    unit_prefix: str = ""
    unit_suffix: str = ""
    decimals: int = 2
    anchor_price: float
    bounds: Bounds
    applied_rules: list[AppliedRule] = Field(default_factory=list)
    overridden_rules: list[OverriddenRule] = Field(default_factory=list)
    skipped_rules: list[SkippedRule] = Field(default_factory=list)
    candidates: list[CheckedCandidate] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    optimizer_policy: OptimizerPolicy
    show_alternatives: bool = True
    customer_messages: list[CustomerMessage] = Field(default_factory=list)
    strategy_source: StrategySource = "manual"
    warnings: list[str] = Field(default_factory=list)

    @property
    def facts(self) -> PriceFacts:
        return PriceFacts(self.reference_price, self.unit_cost, self.previous_price, self.allow_below_cost)

    def eligible_candidates(self) -> list[CheckedCandidate]:
        return [c for c in self.candidates if c.eligible]

    def optimizer_request(self) -> OptimizerRequest:
        return OptimizerRequest(
            domain_id=self.domain_id,
            strategy_version_id=self.strategy_version_id,
            base_price=self.reference_price,
            unit_cost=self.unit_cost,
            competitor_price=self.signals.get("competitorPrice"),
            perceived_value=self.signals.get("perceivedValue"),
            signals={k: v for k, v in self.signals.items() if k not in ("basePrice", "unitCost")},
            candidates=self.candidates,
            objective=self.optimizer_policy.objective,
            conversion_floor=self.optimizer_policy.conversion_floor,
        )


def prepare(
    domain: DomainConfig,
    product: Product,
    strategy: StrategyConfig,
    request: PriceRequest,
    strategy_version_id: str,
    profile: Optional[CustomerProfile] = None,
    version_number: Optional[int] = None,
) -> Prepared:
    """Run the deterministic half of the pipeline. Raises InvalidInputError on bad input."""
    errors: list[str] = []
    if request.domain_id != domain.id:
        errors.append(f'request is for domain "{request.domain_id}", not "{domain.id}"')
    if request.product_id != product.id:
        errors.append(f'request is for product "{request.product_id}", not "{product.id}"')
    if strategy.domain_id != domain.id:
        errors.append(f'strategy belongs to domain "{strategy.domain_id}", not "{domain.id}"')
    if errors:
        raise InvalidInputError(errors)

    enriched = enrich_context(domain, product, request.attributes, request.as_of, profile)
    if enriched.errors:
        raise InvalidInputError(enriched.errors)
    ctx = enriched.context
    signals = map_signals(domain, ctx)
    warnings = enriched.warnings + signals.warnings

    selection = select_rules(strategy.rules, ctx, request.as_of, domain.pipeline)
    applied, overridden = resolve_conflicts(selection.fired, strategy.conflict_policy.mode, domain.pipeline)
    result = apply_rules(product.base_price, applied, strategy.conflict_policy.combine, ctx)
    result = cap_total_adjustment(product.base_price, result, strategy.conflict_policy.max_total_adjustment_pct)
    skipped = list(selection.skipped)
    errored = {e.rule_id for e in result.errors}
    for e in result.errors:
        skipped.append(SkippedRule(rule_id=e.rule_id, reason=f"formula error: {e.message} (no change)"))
        warnings.append(f'Rule "{e.rule_id}" formula failed: {e.message}')

    facts = PriceFacts(product.base_price, product.cost, product.previous_price, domain.allow_below_cost)
    bounds = compute_bounds(strategy.constraints, facts)
    generated = generate_candidates(
        result.anchor, product.base_price, bounds, strategy.candidate_policy,
        strategy.rounding_step(), domain.output_unit.decimals,
    )
    candidates = check_candidates(generated, strategy.constraints, facts)

    status: DecisionStatus = "OK"
    reason: Optional[str] = None
    if bounds.conflict:
        status = "NO_VALID_PRICE"
        lo = f"{bounds.lower:g}" if bounds.lower is not None else "∞"
        reason = (f"constraints conflict: lower bound {lo} ({', '.join(bounds.lower_by)}) is above "
                  f"upper bound {bounds.upper:g} ({', '.join(bounds.upper_by)})"
                  if bounds.upper is not None else
                  f"constraints conflict: {', '.join(bounds.lower_by)} cannot be satisfied")
    elif not any(c.eligible for c in candidates):
        status = "NO_VALID_PRICE"
        reason = "no candidate price satisfies every hard constraint"

    applied_ids = [a.rule_id for a in result.breakdown]
    messages = [
        CustomerMessage(rule_id=r.id, message=r.customer_message)
        for r in applied
        if r.customer_message and r.id in applied_ids and r.id not in errored
    ]
    return Prepared(
        domain_id=domain.id, product_id=product.id, strategy_version_id=strategy_version_id,
        version_number=version_number, status=status, status_reason=reason, request=request,
        context=ctx, signals=signals.values, signal_sources=signals.sources,
        reference_price=product.base_price, unit_cost=product.cost, previous_price=product.previous_price,
        allow_below_cost=domain.allow_below_cost, currency=domain.currency, unit_label=domain.output_unit.label,
        unit_prefix=domain.output_unit.prefix, unit_suffix=domain.output_unit.suffix,
        decimals=domain.output_unit.decimals, anchor_price=result.anchor, bounds=bounds,
        applied_rules=result.breakdown, overridden_rules=overridden, skipped_rules=skipped,
        candidates=candidates, constraints=strategy.constraints, optimizer_policy=strategy.optimizer_policy,
        show_alternatives=strategy.candidate_policy.show_alternatives, customer_messages=messages,
        strategy_source=strategy.source, warnings=warnings,
    )
