"""Shared application context and lookups used by every service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.errors import ApiError, not_found
from app.llm.base import JsonLlm
from app.market import MarketResearcher
from app.models import CustomerRow, DomainRow, StrategyDraftRow, StrategyVersionRow
from app.optimizer.base import PriceOptimizer
from app.optimizer.factory import OptimizerStatus
from app.schemas.domain import DomainConfig, Persona
from app.schemas.pricing_parts import CustomerProfile
from app.schemas.strategy import StrategyConfig
from app.validation import ValidationService, default_personas


@dataclass
class AppContext:
    settings: Settings
    sessions: sessionmaker[Session]
    optimizer: PriceOptimizer
    optimizer_status: OptimizerStatus
    llm: Optional[JsonLlm]
    persona_llm: Optional[JsonLlm]
    validation: ValidationService
    market: Optional[MarketResearcher] = None
    default_personas: list[Persona] = field(default_factory=default_personas)

    @property
    def llm_name(self) -> str:
        return self.llm.name if self.llm else "mock"


def get_domain(s: Session, domain_id: str) -> tuple[DomainRow, DomainConfig]:
    row = s.get(DomainRow, domain_id)
    if row is None:
        raise not_found("Domain", domain_id)
    return row, DomainConfig.model_validate(row.config)


def get_draft(s: Session, domain_id: str) -> tuple[StrategyDraftRow, StrategyConfig]:
    row = s.get(StrategyDraftRow, domain_id)
    if row is None:
        raise not_found("Strategy draft for domain", domain_id)
    return row, StrategyConfig.model_validate(row.config)


def get_version(s: Session, domain_id: str, version_id: str) -> StrategyVersionRow:
    row = s.get(StrategyVersionRow, version_id)
    if row is None or row.domain_id != domain_id:
        raise not_found("Version", version_id)
    return row


def active_version(s: Session, domain_id: str) -> Optional[StrategyVersionRow]:
    return s.scalars(select(StrategyVersionRow).where(
        StrategyVersionRow.domain_id == domain_id, StrategyVersionRow.status == "ACTIVE")).first()


def require_active(s: Session, domain_id: str) -> StrategyVersionRow:
    v = active_version(s, domain_id)
    if v is None:
        raise ApiError(409, "NO_ACTIVE_VERSION", f'Domain "{domain_id}" has no published strategy yet.')
    return v


def personas_for(ctx: AppContext, domain: DomainConfig) -> list[Persona]:
    """Default personas plus the domain's own; a domain persona with a default's id replaces it."""
    own = {p.id: p for p in domain.personas}
    return [own.pop(p.id, p) for p in ctx.default_personas] + list(own.values())


def load_profile(s: Session, customer_id: Optional[str]) -> Optional[CustomerProfile]:
    if not customer_id:
        return None
    row = s.get(CustomerRow, customer_id)
    if row is None:
        raise not_found("Customer", customer_id)
    return CustomerProfile(id=row.id, attributes=row.profile)
