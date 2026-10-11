"""Domains, drafts, generation, validation, approval, publication and rollback."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import ValidationError
from sqlalchemy import func, select

from app.audit import audited, record
from app.auth import Principal
from app.db import utcnow
from app.engine import content_hash, validate_strategy, validate_strategy_payload
from app.engine.validate import schema_issues
from app.errors import ApiError
from app.lifecycle import check_draft_can_be_approved, check_version_transition
from app.llm import mock as mock_llm
from app.llm.convert import to_strategy_dict
from app.llm.prompts import DOMAIN_SYSTEM, STRATEGY_SYSTEM, domain_user_prompt, strategy_user_prompt
from app.market import compact_for_prompt
from app.schemas.market import MarketContext
from app.models import DomainRow, StrategyDraftRow, StrategyVersionRow, ValidationRunRow
from app.schemas.domain import DomainConfig
from app.schemas.llm import LlmStrategyOut
from app.schemas.strategy import StrategyConfig
from app.schemas.validation import ValidationReport

from .common import AppContext, active_version, get_domain, get_draft, get_version


def _dump(model) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=True)


def _issues_from(e: ValidationError, prefix: str = "") -> list[dict[str, Any]]:
    return [{"code": code, "path": path, "message": msg} for code, path, msg in schema_issues(e, prefix)]


def parse_domain(raw: Any) -> tuple[DomainConfig, Optional[StrategyConfig]]:
    """Accept a bare DomainConfig or a {domain, strategy} bundle; raise 422 with path-based errors."""
    if isinstance(raw, dict) and "domain" in raw:
        try:
            domain = DomainConfig.model_validate(raw["domain"])
        except ValidationError as e:
            raise ApiError(422, "INVALID_DOMAIN", "The domain configuration is invalid.", _issues_from(e, "domain.")) from None
        strategy = None
        if raw.get("strategy") is not None:
            try:
                strategy = StrategyConfig.model_validate(raw["strategy"])
            except ValidationError as e:
                raise ApiError(422, "INVALID_STRATEGY", "The strategy configuration is invalid.", _issues_from(e, "strategy.")) from None
            if strategy.domain_id != domain.id:
                raise ApiError(422, "INVALID_STRATEGY", "strategy.domain_id must match domain.id.")
        return domain, strategy
    try:
        return DomainConfig.model_validate(raw), None
    except ValidationError as e:
        raise ApiError(422, "INVALID_DOMAIN", "The domain configuration is invalid.", _issues_from(e)) from None


def _set_draft(row: StrategyDraftRow, cfg: StrategyConfig, who: Principal) -> None:
    row.config = _dump(cfg)
    row.content_hash = content_hash(row.config)
    row.status = "DRAFT"
    row.source = cfg.source
    row.updated_by = who.user
    row.updated_at = utcnow()


def create_domain(ctx: AppContext, who: Principal, raw: Any) -> DomainConfig:
    domain, strategy = parse_domain(raw)
    with audited(ctx.sessions, who, "CREATE_DOMAIN", "domain", domain.id) as s:
        if s.get(DomainRow, domain.id) is not None:
            raise ApiError(409, "DOMAIN_EXISTS", f'Domain "{domain.id}" already exists.')
        now = utcnow()
        s.add(DomainRow(id=domain.id, name=domain.name, config=_dump(domain), created_by=who.user, created_at=now, updated_at=now))
        cfg = strategy or StrategyConfig(domain_id=domain.id, name=f"{domain.name} strategy")
        draft = StrategyDraftRow(domain_id=domain.id)
        _set_draft(draft, cfg, who)
        s.add(draft)
        record(s, who, "CREATE_DOMAIN", "domain", domain.id, details={"with_strategy": strategy is not None})
    return domain


def update_domain(ctx: AppContext, who: Principal, domain_id: str, raw: Any) -> DomainConfig:
    domain, _ = parse_domain(raw)
    if domain.id != domain_id:
        raise ApiError(422, "INVALID_DOMAIN", "The domain id cannot be changed.")
    with audited(ctx.sessions, who, "UPDATE_DOMAIN", "domain", domain_id) as s:
        row, _old = get_domain(s, domain_id)
        row.config, row.name, row.updated_at = _dump(domain), domain.name, utcnow()
        draft = s.get(StrategyDraftRow, domain_id)
        if draft is not None and draft.status == "VALIDATED":
            draft.status = "DRAFT"  # the domain changed, so the old validation no longer applies
        record(s, who, "UPDATE_DOMAIN", "domain", domain_id)
    return domain


def put_draft(ctx: AppContext, who: Principal, domain_id: str, raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        raw = {**raw, "domain_id": domain_id}
        raw.setdefault("source", "manual")
    with audited(ctx.sessions, who, "UPDATE_DRAFT", "strategy", domain_id) as s:
        _, domain = get_domain(s, domain_id)
        cfg, report = validate_strategy_payload(domain, raw)
        if cfg is None:
            raise ApiError(422, "INVALID_STRATEGY", "The strategy does not match the schema.", [i.model_dump() for i in report.issues])
        row = s.get(StrategyDraftRow, domain_id)
        if row is None:
            row = StrategyDraftRow(domain_id=domain_id)
            s.add(row)
        _set_draft(row, cfg, who)
        record(s, who, "UPDATE_DRAFT", "strategy", domain_id, details={"content_hash": row.content_hash})
        return draft_view(row, report)


def draft_view(row: StrategyDraftRow, report: Optional[ValidationReport] = None) -> dict[str, Any]:
    return {"domain_id": row.domain_id, "status": row.status, "source": row.source, "content_hash": row.content_hash,
            "updated_by": row.updated_by, "updated_at": row.updated_at, "strategy": row.config,
            "issues": [i.model_dump() for i in report.issues] if report else None}


def validate_draft(ctx: AppContext, who: Principal, domain_id: str) -> dict[str, Any]:
    with audited(ctx.sessions, who, "VALIDATE", "strategy", domain_id) as s:
        _, domain = get_domain(s, domain_id)
        row, cfg = get_draft(s, domain_id)
        report = validate_strategy(domain, cfg)
        s.add(ValidationRunRow(domain_id=domain_id, content_hash=row.content_hash, ok=report.ok, result=report.model_dump(),
                               created_by=who.user, created_at=utcnow()))
        row.status = "VALIDATED" if report.ok else "DRAFT"
        record(s, who, "VALIDATE", "strategy", domain_id, outcome="SUCCESS" if report.ok else "REJECTED",
               details={"content_hash": row.content_hash, "errors": report.counts["error"], "warnings": report.counts["warning"]})
        return {"report": report.model_dump(), **draft_view(row)}


def research_market(ctx: AppContext, who: Principal, domain_id: str) -> Optional[MarketContext]:
    """Run web market research for a domain (None when no search provider is configured)."""
    with ctx.sessions() as s:
        _, domain = get_domain(s, domain_id)
    if ctx.market is None:
        return None
    market = ctx.market.research(domain)
    with ctx.sessions() as s:
        record(s, who, "MARKET_RESEARCH", "domain", domain_id, details={
            "provider": market.provider, "searches_used": market.searches_used, "cache_hits": market.cache_hits,
            "products_with_prices": sum(1 for m in market.products if m.n)})
        s.commit()
    return market


def generate(ctx: AppContext, who: Principal, domain_id: str, description: str, use_market: bool = True) -> dict[str, Any]:
    if not description.strip():
        raise ApiError(422, "EMPTY_DESCRIPTION", "Describe the pricing objective first.")
    with ctx.sessions() as s:
        _, domain = get_domain(s, domain_id)
    market: Optional[MarketContext] = None
    research_error: Optional[str] = None
    if use_market and ctx.market is not None:
        try:
            market = research_market(ctx, who, domain_id)
        except Exception as e:  # noqa: BLE001 - research is optional; never block generation
            research_error = str(e)
    market_prompt = compact_for_prompt(market)
    provider = ctx.llm_name
    now = utcnow()
    observations: list[str] = []
    assumptions: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    cfg: Optional[StrategyConfig] = None
    for attempt in range(1 if ctx.llm is None else 2):
        try:
            raw = (mock_llm.generate_strategy(domain, description, market_prompt) if ctx.llm is None else
                   ctx.llm.generate_json(STRATEGY_SYSTEM, strategy_user_prompt(domain, description, errors or None, market_prompt),
                                         LlmStrategyOut))
            out = LlmStrategyOut.model_validate(raw)
        except ValidationError as e:
            errors = [m for _, _, m in schema_issues(e)]
            continue
        except Exception as e:  # noqa: BLE001 - provider failures
            raise ApiError(502, "LLM_UNAVAILABLE", f"The {provider} provider failed: {e}") from None
        assumptions, warnings, observations = out.assumptions, out.warnings, out.market_observations
        candidate, report = validate_strategy_payload(domain, to_strategy_dict(out, domain_id, provider, now), allow_formula=False)
        if candidate is not None and report.ok:
            cfg = candidate
            warnings = warnings + [i.message for i in report.issues if i.severity == "warning"]
            break
        errors = [i.message for i in report.errors()]
    if cfg is None:
        with ctx.sessions() as s:
            record(s, who, "GENERATE_STRATEGY", "strategy", domain_id, outcome="REJECTED", details={"provider": provider, "errors": errors})
            s.commit()
        raise ApiError(422, "GENERATED_STRATEGY_INVALID", f"The {provider} output failed validation; the draft was not changed.",
                       {"provider": provider, "errors": errors})
    with audited(ctx.sessions, who, "GENERATE_STRATEGY", "strategy", domain_id) as s:
        row = s.get(StrategyDraftRow, domain_id)
        _set_draft(row, cfg, who)
        record(s, who, "GENERATE_STRATEGY", "strategy", domain_id, details={
            "provider": provider, "rules": len(cfg.rules), "market_research": market is not None,
            "searches_used": market.searches_used if market else 0})
        view = draft_view(row)
    if research_error:
        warnings = [f"Market research failed ({research_error}); generated without it."] + warnings
    label = {"mock": "Mock generator — no LLM configured", "groq": f"Groq · {getattr(ctx.llm, 'model', '')}"}.get(provider, provider)
    return {**view, "provider": provider, "provider_label": label, "assumptions": assumptions, "warnings": warnings,
            "market_observations": observations, "market": market.model_dump(mode="json") if market else None}


def suggest_domain(ctx: AppContext, description: str) -> dict[str, Any]:
    if not description.strip():
        raise ApiError(422, "EMPTY_DESCRIPTION", "Describe the business first.")
    if ctx.llm is None:
        raw, provider = mock_llm.suggest_domain(description), "mock"
    else:
        try:
            raw, provider = ctx.llm.generate_json(DOMAIN_SYSTEM, domain_user_prompt(description), DomainConfig), ctx.llm.name
        except Exception as e:  # noqa: BLE001
            raise ApiError(502, "LLM_UNAVAILABLE", f"The {ctx.llm.name} provider failed: {e}") from None
    try:
        domain = DomainConfig.model_validate(raw)
    except ValidationError as e:
        raise ApiError(422, "GENERATED_DOMAIN_INVALID", "The suggested domain failed validation.", _issues_from(e)) from None
    return {"domain": _dump(domain), "provider": provider}


def _version_view(v: StrategyVersionRow, include_config: bool = True) -> dict[str, Any]:
    d = {k: getattr(v, k) for k in ("id", "domain_id", "number", "status", "content_hash", "created_by", "approved_by",
                                    "approved_at", "published_at", "retired_at", "notes")}
    if include_config:
        d["strategy"] = v.config
    return d


def approve(ctx: AppContext, who: Principal, domain_id: str, notes: str = "") -> dict[str, Any]:
    with audited(ctx.sessions, who, "APPROVE", "strategy", domain_id) as s:
        domain_row, domain = get_domain(s, domain_id)
        row, cfg = get_draft(s, domain_id)
        check_draft_can_be_approved(row.status)
        last = s.scalars(select(ValidationRunRow).where(ValidationRunRow.domain_id == domain_id)
                         .order_by(ValidationRunRow.id.desc())).first()
        if last is None or not last.ok or last.content_hash != row.content_hash:
            raise ApiError(409, "STRATEGY_NOT_VALIDATED", "The current draft has no passing validation.")
        if not validate_strategy(domain, cfg).ok:  # re-check right before freezing
            raise ApiError(409, "STRATEGY_NOT_VALIDATED", "The draft no longer validates.")
        if row.updated_by == who.user:
            raise ApiError(403, "MAKER_CHECKER", "You last edited this draft, so someone else must approve it.")
        check_version_transition(None, "APPROVED", "approve")
        number = (s.scalar(select(func.max(StrategyVersionRow.number)).where(StrategyVersionRow.domain_id == domain_id)) or 0) + 1
        now = utcnow()
        v = StrategyVersionRow(id=f"{domain_id}-v{number}", domain_id=domain_id, number=number, status="APPROVED",
                               config=row.config, domain_snapshot=domain_row.config, content_hash=row.content_hash,
                               created_by=row.updated_by, approved_by=who.user, approved_at=now, notes=notes)
        s.add(v)
        record(s, who, "APPROVE", "version", v.id, version=v.id, details={"content_hash": row.content_hash})
        return _version_view(v)


def _activate(ctx: AppContext, who: Principal, domain_id: str, version_id: str, operation: str) -> dict[str, Any]:
    with audited(ctx.sessions, who, operation.upper(), "version", version_id) as s:
        target = get_version(s, domain_id, version_id)
        check_version_transition(target.status, "ACTIVE", operation)
        now = utcnow()
        current = active_version(s, domain_id)
        if current is not None:
            current.status, current.retired_at = "RETIRED", now
            s.flush()  # free the one-ACTIVE-per-domain slot before activating the target
        target.status, target.published_at, target.retired_at = "ACTIVE", now, None
        record(s, who, operation.upper(), "version", target.id, version=target.id,
               details={"previous_active": current.id if current else None})
        return _version_view(target)


def publish(ctx: AppContext, who: Principal, domain_id: str, version_id: str) -> dict[str, Any]:
    return _activate(ctx, who, domain_id, version_id, "publish")


def rollback(ctx: AppContext, who: Principal, domain_id: str, version_id: str) -> dict[str, Any]:
    return _activate(ctx, who, domain_id, version_id, "rollback")


def list_versions(ctx: AppContext, domain_id: str) -> list[dict[str, Any]]:
    with ctx.sessions() as s:
        get_domain(s, domain_id)
        rows = s.scalars(select(StrategyVersionRow).where(StrategyVersionRow.domain_id == domain_id)
                         .order_by(StrategyVersionRow.number.desc())).all()
        return [_version_view(v) for v in rows]
