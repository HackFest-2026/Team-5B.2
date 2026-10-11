"""Domains, strategy drafts, generation, validation, simulation and the version lifecycle."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, Request
from pydantic import Field

from app.auth import APPROVERS, BUSINESS, Principal, require
from app.engine import diff_strategies
from app.errors import ApiError
from app.models import DomainRow
from app.schemas.common import StrictModel
from app.schemas.strategy import StrategyConfig
from app.services import pricing, strategy as svc
from app.services.common import active_version, get_domain, get_draft, get_version

router = APIRouter(prefix="/api/v1/domains", tags=["domains"])
business = require(*BUSINESS)
approver = require(*APPROVERS)


def ctx(request: Request):
    return request.app.state.ctx


class DescribeBody(StrictModel):
    description: str = Field(min_length=1, max_length=4000)


class GenerateBody(DescribeBody):
    use_market_research: bool = True


class VersionBody(StrictModel):
    version_id: str


class ApproveBody(StrictModel):
    notes: str = Field(default="", max_length=1000)


@router.get("")
def list_domains(request: Request, who: Principal = Depends(business)) -> list[dict[str, Any]]:
    c = ctx(request)
    with c.sessions() as s:
        out = []
        for row in s.query(DomainRow).order_by(DomainRow.created_at, DomainRow.id).all():
            draft_row = s.get(svc.StrategyDraftRow, row.id)
            active = active_version(s, row.id)
            out.append({"id": row.id, "name": row.name, "description": row.config.get("description", ""),
                        "currency": row.config.get("currency"), "products": len(row.config.get("products", [])),
                        "draft_status": draft_row.status if draft_row else None,
                        "active_version": active.id if active else None, "created_at": row.created_at})
        return out


@router.post("", status_code=201)
def create_domain(request: Request, body: Any = Body(...), who: Principal = Depends(business)) -> dict[str, Any]:
    return svc.create_domain(ctx(request), who, body).model_dump(mode="json")


@router.post("/suggest")
def suggest(request: Request, body: DescribeBody, who: Principal = Depends(business)) -> dict[str, Any]:
    return svc.suggest_domain(ctx(request), body.description)


@router.get("/{domain_id}")
def get_one(domain_id: str, request: Request, who: Principal = Depends(business)) -> dict[str, Any]:
    with ctx(request).sessions() as s:
        _, domain = get_domain(s, domain_id)
        return domain.model_dump(mode="json")


@router.put("/{domain_id}")
def update(domain_id: str, request: Request, body: Any = Body(...), who: Principal = Depends(business)) -> dict[str, Any]:
    return svc.update_domain(ctx(request), who, domain_id, body).model_dump(mode="json")


@router.post("/{domain_id}/generate-strategy")
def generate(domain_id: str, request: Request, body: GenerateBody, who: Principal = Depends(business)) -> dict[str, Any]:
    return svc.generate(ctx(request), who, domain_id, body.description, body.use_market_research)


@router.post("/{domain_id}/market-research")
def market_research(domain_id: str, request: Request, who: Principal = Depends(business)) -> dict[str, Any]:
    market = svc.research_market(ctx(request), who, domain_id)
    if market is None:
        raise ApiError(409, "MARKET_RESEARCH_DISABLED", "No search provider is configured (set SERPAPI_API_KEY or SERPER_API_KEY).")
    return market.model_dump(mode="json")


@router.get("/{domain_id}/strategy")
def get_strategy(domain_id: str, request: Request, who: Principal = Depends(business)) -> dict[str, Any]:
    with ctx(request).sessions() as s:
        get_domain(s, domain_id)
        row, _ = get_draft(s, domain_id)
        return svc.draft_view(row)


@router.put("/{domain_id}/strategy")
def put_strategy(domain_id: str, request: Request, body: Any = Body(...), who: Principal = Depends(business)) -> dict[str, Any]:
    return svc.put_draft(ctx(request), who, domain_id, body)


@router.post("/{domain_id}/validate")
def validate(domain_id: str, request: Request, who: Principal = Depends(business)) -> dict[str, Any]:
    return svc.validate_draft(ctx(request), who, domain_id)


@router.post("/{domain_id}/simulate")
def simulate(domain_id: str, request: Request, body: pricing.SimulateRequest, who: Principal = Depends(business)) -> dict[str, Any]:
    return pricing.simulate(ctx(request), who, domain_id, body)


@router.post("/{domain_id}/approve")
def approve(domain_id: str, request: Request, body: Optional[ApproveBody] = None, who: Principal = Depends(approver)) -> dict[str, Any]:
    return svc.approve(ctx(request), who, domain_id, body.notes if body else "")


@router.post("/{domain_id}/publish")
def publish(domain_id: str, request: Request, body: VersionBody, who: Principal = Depends(approver)) -> dict[str, Any]:
    return svc.publish(ctx(request), who, domain_id, body.version_id)


@router.post("/{domain_id}/rollback")
def rollback(domain_id: str, request: Request, body: VersionBody, who: Principal = Depends(approver)) -> dict[str, Any]:
    return svc.rollback(ctx(request), who, domain_id, body.version_id)


@router.get("/{domain_id}/versions")
def versions(domain_id: str, request: Request, who: Principal = Depends(business)) -> list[dict[str, Any]]:
    return svc.list_versions(ctx(request), domain_id)


@router.get("/{domain_id}/versions/{a}/diff/{b}")
def diff(domain_id: str, a: str, b: str, request: Request, who: Principal = Depends(business)) -> dict[str, Any]:
    """Diff two versions; use "draft" for the current draft."""
    with ctx(request).sessions() as s:
        get_domain(s, domain_id)

        def load(ref: str) -> StrategyConfig:
            if ref == "draft":
                return get_draft(s, domain_id)[1]
            return StrategyConfig.model_validate(get_version(s, domain_id, ref).config)

        if a == b:
            raise ApiError(422, "SAME_VERSION", "Choose two different versions to compare.")
        return diff_strategies(load(a), load(b)).model_dump(mode="json")
