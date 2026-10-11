"""Quotes and simulations: prepare → optimizer → finalize → persist → shadow validation."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import BackgroundTasks
from pydantic import Field
from sqlalchemy import select

from app.audit import record
from app.auth import Principal
from app.db import utcnow
from app.engine import content_hash, finalize, prepare, to_customer_quote, validate_strategy
from app.errors import ApiError, not_found
from app.models import CandidateEvaluationRow, CustomerRow, DecisionRow, ShadowValidationRow, SimulationRunRow
from app.schemas.common import Id, IsoDate, Key, Scalar, StrictModel
from app.schemas.domain import DomainConfig
from app.schemas.persona import ShadowValidation
from app.schemas.pricing import PricingDecision
from app.schemas.pricing_parts import CustomerProfile, PriceRequest
from app.schemas.strategy import StrategyConfig
from app.validation import reactions_at
from app.validation.narrative import rewrite

from .common import AppContext, get_domain, get_draft, get_version, load_profile, personas_for, require_active

log = logging.getLogger(__name__)


class QuoteRequest(StrictModel):
    """No price field on purpose: a caller-supplied price is rejected (extra fields are forbidden)."""

    domain_id: Id
    product_id: Id
    customer_id: Optional[Id] = None
    attributes: dict[Key, Scalar] = Field(default_factory=dict)
    as_of: Optional[IsoDate] = None
    include_personas: bool = False


class Scenario(StrictModel):
    label: str = Field(default="Scenario", max_length=80)
    product_id: Id
    customer_id: Optional[Id] = None
    attributes: dict[Key, Scalar] = Field(default_factory=dict)


class SimulateRequest(StrictModel):
    version_id: Optional[str] = None
    as_of: Optional[IsoDate] = None
    scenarios: list[Scenario] = Field(min_length=1, max_length=10)
    include_personas: bool = True


def _today() -> str:
    return utcnow()[:10]


def price_once(ctx: AppContext, domain: DomainConfig, strategy: StrategyConfig, version_id: str, version_number: Optional[int],
               product_id: str, attributes: dict, as_of: str, profile: Optional[CustomerProfile]) -> tuple[Any, PricingDecision]:
    product = domain.product(product_id)
    if product is None:
        raise not_found("Product", product_id)
    request = PriceRequest(domain_id=domain.id, product_id=product_id, customer_id=profile.id if profile else None,
                           attributes=attributes, as_of=as_of)
    prepared = prepare(domain, product, strategy, request, version_id, profile, version_number)
    response = ctx.optimizer.evaluate(prepared.optimizer_request()) if prepared.candidates else None
    return prepared, finalize(prepared, response, created_at=utcnow())


def _validate(ctx: AppContext, decision: PricingDecision, domain: DomainConfig, strategy: StrategyConfig,
              population: int, include_personas: bool) -> ShadowValidation:
    v = ctx.validation.run(decision, strategy.validation_policy, personas_for(ctx, domain), population, include_personas)
    return rewrite(v, ctx.persona_llm, decision.final_price)


def _attach(decision: PricingDecision, v: ShadowValidation) -> PricingDecision:
    """Return a copy with validation attached. Price, candidates and status are untouched (shadow mode)."""
    return decision.model_copy(update={"shadow_validation": v, "persona_summary": reactions_at(v, decision.final_price),
                                       "provenance": decision.provenance.model_copy(update={"personas": "agent_sim"})})


def _customer_attributes(domain: DomainConfig, attributes: dict) -> dict:
    allowed = {a.key for a in domain.attributes if a.customer_input}
    extra = sorted(set(attributes) - allowed)
    if extra:
        raise ApiError(422, "INVALID_INPUT", "Some fields cannot be set by customers.", [f'"{k}" is not a customer input' for k in extra])
    return attributes


def quote(ctx: AppContext, who: Principal, body: QuoteRequest, tasks: BackgroundTasks) -> dict[str, Any]:
    with ctx.sessions() as s:
        get_domain(s, body.domain_id)  # 404 if the domain does not exist
        version = require_active(s, body.domain_id)
        domain = DomainConfig.model_validate(version.domain_snapshot)  # the version's frozen domain, not live config
        strategy = StrategyConfig.model_validate(version.config)
        attributes = body.attributes
        customer_id = body.customer_id
        if not who.is_business:
            attributes = _customer_attributes(domain, attributes)
            customer_id = who.user if s.get(CustomerRow, who.user) is not None else None
        profile = load_profile(s, customer_id)
        _, decision = price_once(ctx, domain, strategy, version.id, version.number, body.product_id, attributes,
                                 body.as_of or _today(), profile)

        existing = s.get(DecisionRow, decision.decision_id)
        if existing is not None:
            decision = PricingDecision.model_validate(existing.result)  # identical inputs → same decision id
        else:
            s.add(DecisionRow(id=decision.decision_id, domain_id=decision.domain_id, version_id=decision.strategy_version_id,
                              product_id=decision.product_id, status=decision.status, final_price=decision.final_price,
                              request=decision.request.model_dump(mode="json"), result=decision.model_dump(mode="json"),
                              created_by=who.user, created_at=decision.created_at or utcnow()))
            s.flush()
            for c in decision.candidates:
                s.add(CandidateEvaluationRow(decision_id=decision.decision_id, price=c.price, eligible=c.eligible,
                                             violations=[v.model_dump() for v in c.violations], p_buy=c.p_buy,
                                             expected_revenue=c.expected_revenue, expected_profit=c.expected_profit,
                                             selection_status=c.selection_status))
        record(s, who, "QUOTE", "decision", decision.decision_id, version=decision.strategy_version_id,
               outcome=decision.status, details={"product_id": body.product_id, "final_price": decision.final_price})

        mode = ctx.settings.validation_mode
        has_validation = s.scalars(select(ShadowValidationRow).where(ShadowValidationRow.owner_id == decision.decision_id)).first()
        if mode != "off" and existing is None and has_validation is None:
            if mode == "sync" or (body.include_personas and who.is_business):
                v = _validate(ctx, decision, domain, strategy, ctx.settings.validation_pop_quote, body.include_personas)
                _store_validation(s, "decision", decision.decision_id, v)
            else:
                pending = ShadowValidation(status="PENDING", population_size=ctx.settings.validation_pop_quote, seed=ctx.validation.seed)
                _store_validation(s, "decision", decision.decision_id, pending)
                tasks.add_task(run_background_validation, ctx, decision.decision_id, body.include_personas)
        s.commit()

    if not who.is_business:
        return to_customer_quote(decision, strategy.candidate_policy.show_alternatives).model_dump(mode="json")
    return get_decision(ctx, decision.decision_id)


def _store_validation(s, owner_type: str, owner_id: str, v: ShadowValidation) -> None:
    row = s.scalars(select(ShadowValidationRow).where(ShadowValidationRow.owner_type == owner_type,
                                                      ShadowValidationRow.owner_id == owner_id)).first()
    data = v.model_dump(mode="json")
    if row is None:
        s.add(ShadowValidationRow(owner_type=owner_type, owner_id=owner_id, status=v.status, result=data,
                                  population_size=v.population_size, seed=v.seed, created_at=utcnow()))
    else:
        row.status, row.result, row.population_size = v.status, data, v.population_size


def run_background_validation(ctx: AppContext, decision_id: str, include_personas: bool) -> None:
    try:
        with ctx.sessions() as s:
            row = s.get(DecisionRow, decision_id)
            decision = PricingDecision.model_validate(row.result)
            version = get_version(s, row.domain_id, row.version_id)
            domain = DomainConfig.model_validate(version.domain_snapshot)
            strategy = StrategyConfig.model_validate(version.config)
        v = _validate(ctx, decision, domain, strategy, ctx.settings.validation_pop_quote, include_personas)
    except Exception as e:  # noqa: BLE001 - shadow mode
        log.exception("background validation failed")
        v = ShadowValidation(status="ERROR", error=str(e), population_size=ctx.settings.validation_pop_quote, seed=ctx.validation.seed)
    with ctx.sessions() as s:
        _store_validation(s, "decision", decision_id, v)
        s.commit()


def get_decision(ctx: AppContext, decision_id: str) -> dict[str, Any]:
    with ctx.sessions() as s:
        row = s.get(DecisionRow, decision_id)
        if row is None:
            raise not_found("Decision", decision_id)
        decision = PricingDecision.model_validate(row.result)
        v = s.scalars(select(ShadowValidationRow).where(ShadowValidationRow.owner_id == decision_id)).first()
        if v is not None:
            decision = _attach(decision, ShadowValidation.model_validate(v.result))
        return decision.model_dump(mode="json")


def list_decisions(ctx: AppContext, domain_id: Optional[str], status: Optional[str], limit: int) -> list[dict[str, Any]]:
    with ctx.sessions() as s:
        q = select(DecisionRow).order_by(DecisionRow.created_at.desc(), DecisionRow.id).limit(limit)
        if domain_id:
            q = q.where(DecisionRow.domain_id == domain_id)
        if status:
            q = q.where(DecisionRow.status == status)
        rows = s.scalars(q).all()
        ids = [r.id for r in rows]
        vstatus = {v.owner_id: v.status for v in s.scalars(select(ShadowValidationRow).where(ShadowValidationRow.owner_id.in_(ids)))}
        return [{"decision_id": r.id, "domain_id": r.domain_id, "product_id": r.product_id, "version_id": r.version_id,
                 "status": r.status, "final_price": r.final_price, "currency": r.result.get("currency"), "decimals": r.result.get("decimals", 2), "unit_prefix": r.result.get("unit_prefix"), "unit_suffix": r.result.get("unit_suffix"),
                 "created_at": r.created_at, "created_by": r.created_by, "validation_status": vstatus.get(r.id)} for r in rows]


def simulate(ctx: AppContext, who: Principal, domain_id: str, body: SimulateRequest) -> dict[str, Any]:
    with ctx.sessions() as s:
        _, live = get_domain(s, domain_id)
        if body.version_id:
            v = get_version(s, domain_id, body.version_id)
            domain, strategy = DomainConfig.model_validate(v.domain_snapshot), StrategyConfig.model_validate(v.config)
            version_id, number, chash = v.id, v.number, v.content_hash
        else:
            row, strategy = get_draft(s, domain_id)
            domain, version_id, number, chash = live, f"draft:{row.content_hash}", None, row.content_hash
        report = validate_strategy(domain, strategy)
        profiles = {sc.customer_id: load_profile(s, sc.customer_id) for sc in body.scenarios if sc.customer_id}
    as_of = body.as_of or _today()
    results = []
    for sc in body.scenarios:
        try:
            _, decision = price_once(ctx, domain, strategy, version_id, number, sc.product_id, sc.attributes, as_of,
                                     profiles.get(sc.customer_id))
            sv = _validate(ctx, decision, domain, strategy, ctx.settings.validation_pop_sim, body.include_personas)
            results.append({"label": sc.label, "decision": _attach(decision, sv).model_dump(mode="json"), "error": None})
        except ApiError:
            raise
        except Exception as e:  # noqa: BLE001 - report per-scenario input errors
            results.append({"label": sc.label, "decision": None,
                            "error": {"message": str(e), "details": getattr(e, "errors", None)}})
    sim_id = "S-" + content_hash({"domain": domain_id, "version": version_id, "req": body.model_dump(mode="json"), "at": utcnow()})
    with ctx.sessions() as s:
        s.add(SimulationRunRow(id=sim_id, domain_id=domain_id, content_hash=chash,
                               version_id=None if version_id.startswith("draft:") else version_id,
                               request=body.model_dump(mode="json"), result={"results": results}, created_by=who.user,
                               created_at=utcnow()))
        for i, r in enumerate(results):
            if r["decision"] and r["decision"].get("shadow_validation"):
                sv = ShadowValidation.model_validate(r["decision"]["shadow_validation"])
                _store_validation(s, "simulation", f"{sim_id}#{i}", sv)
        record(s, who, "SIMULATE", "strategy", domain_id, version=version_id, details={"simulation_id": sim_id, "scenarios": len(results)})
        s.commit()
    return {"simulation_id": sim_id, "version_id": version_id, "content_hash": chash, "validation": report.model_dump(),
            "results": results}

