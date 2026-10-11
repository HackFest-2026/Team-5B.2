"""Quotes, decisions, the customer catalog, audit logs, customers, personas, health and admin."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from sqlalchemy import select

from app.auth import ANY, BUSINESS, Principal, require
from app.models import AuditLogRow, CustomerRow, DomainRow, ShadowValidationRow
from app.schemas.domain import DomainConfig
from app.services import pricing
from app.services.common import active_version

router = APIRouter(prefix="/api/v1", tags=["pricing"])
business = require(*BUSINESS)
anyone = require(*ANY)


def ctx(request: Request):
    return request.app.state.ctx


@router.post("/pricing/quote")
def quote(body: pricing.QuoteRequest, request: Request, tasks: BackgroundTasks, who: Principal = Depends(anyone)) -> dict[str, Any]:
    return pricing.quote(ctx(request), who, body, tasks)


@router.get("/pricing/decisions")
def decisions(request: Request, domain_id: Optional[str] = None, status: Optional[str] = None,
              limit: int = Query(default=50, ge=1, le=500), who: Principal = Depends(business)) -> list[dict[str, Any]]:
    return pricing.list_decisions(ctx(request), domain_id, status, limit)


@router.get("/pricing/decisions/{decision_id}")
def decision(decision_id: str, request: Request, who: Principal = Depends(business)) -> dict[str, Any]:
    return pricing.get_decision(ctx(request), decision_id)


@router.get("/pricing/decisions/{decision_id}/validation")
def decision_validation(decision_id: str, request: Request, who: Principal = Depends(business)) -> dict[str, Any]:
    pricing.get_decision(ctx(request), decision_id)  # 404 if unknown
    with ctx(request).sessions() as s:
        row = s.scalars(select(ShadowValidationRow).where(ShadowValidationRow.owner_id == decision_id)).first()
        return row.result if row else {"status": "SKIPPED", "reason": "validation is turned off"}


@router.get("/catalog")
def catalog(request: Request, who: Principal = Depends(anyone)) -> list[dict[str, Any]]:
    """Published domains with customer-safe fields only (no cost, rules or internal attributes)."""
    out = []
    with ctx(request).sessions() as s:
        for row in s.scalars(select(DomainRow).order_by(DomainRow.created_at, DomainRow.id)):
            v = active_version(s, row.id)
            if v is None:
                continue
            d = DomainConfig.model_validate(v.domain_snapshot)
            out.append({
                "domain_id": d.id, "name": d.name, "description": d.description, "currency": d.currency,
                "unit_label": d.output_unit.label, "product_noun": d.product_noun,
                "products": [{"id": p.id, "name": p.name} for p in d.products],
                "inputs": [a.model_dump(include={"key", "label", "type", "allowed_values", "default", "min", "max", "required",
                                                 "unit", "description"}, exclude_none=True)
                           for a in d.attributes if a.customer_input],
            })
    return out


@router.get("/audit-logs")
def audit_logs(request: Request, resource_type: Optional[str] = None, resource_id: Optional[str] = None,
               actor: Optional[str] = None, operation: Optional[str] = None,
               limit: int = Query(default=100, ge=1, le=1000), who: Principal = Depends(business)) -> list[dict[str, Any]]:
    with ctx(request).sessions() as s:
        q = select(AuditLogRow).order_by(AuditLogRow.id.desc()).limit(limit)
        for col, val in ((AuditLogRow.resource_type, resource_type), (AuditLogRow.resource_id, resource_id),
                         (AuditLogRow.actor, actor), (AuditLogRow.operation, operation)):
            if val:
                q = q.where(col == val)
        return [{k: getattr(r, k) for k in ("id", "at", "actor", "role", "operation", "resource_type", "resource_id", "version",
                                            "outcome", "details")} for r in s.scalars(q)]


@router.get("/customers")
def customers(request: Request, domain_id: Optional[str] = None, who: Principal = Depends(business)) -> list[dict[str, Any]]:
    with ctx(request).sessions() as s:
        q = select(CustomerRow).order_by(CustomerRow.id)
        if domain_id:
            q = q.where(CustomerRow.domain_id == domain_id)
        return [{"id": c.id, "domain_id": c.domain_id, "name": c.name, "profile": c.profile} for c in s.scalars(q)]


@router.get("/personas/defaults")
def persona_defaults(request: Request, who: Principal = Depends(business)) -> list[dict[str, Any]]:
    return [p.model_dump(mode="json") for p in ctx(request).default_personas]


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    c = ctx(request)
    info = c.optimizer.info
    return {
        "status": "ok",
        "llm": {"provider": c.llm_name, "model": getattr(c.llm, "model", None), "mock": c.llm is None},
        "optimizer": {"name": info.name, "version": info.version, "kind": info.kind, "training_data": info.training_data,
                      "fallback_reason": c.optimizer_status.fallback_reason},
        "market": {"provider": c.market.search.name if c.market else None, "enabled": c.market is not None,
                   "max_products": c.market.max_products if c.market else 0},
        "validation": {"mode": c.settings.validation_mode, "module": "customer_agent_sim", "advisory_only": True,
                       "persona_text": c.persona_llm.name if c.persona_llm else "templates"},
        "auth": "development tokens (Bearer dev:<role>:<user>) — not production security",
    }


@router.post("/admin/reseed")
def reseed(request: Request, who: Principal = Depends(require("admin"))) -> dict[str, Any]:
    from app.seed.seed import seed

    return {"seeded": seed(ctx(request).sessions, reset=True)}
