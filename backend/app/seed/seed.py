"""Seed a fresh demo database: three domains, each with v1 approved and published, plus demo customers.

All values are synthetic demo data (see each domain's description). The fourth example domain,
car-rental, is deliberately NOT seeded: it is imported live to show that a new domain needs only
configuration. Run with ``python -m app.seed.seed`` (``--reset`` wipes existing data first).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.auth import Principal
from app.config import REPO_DIR, get_settings
from app.db import make_engine, make_session_factory, run_migrations
from app.models import (
    AuditLogRow,
    CandidateEvaluationRow,
    CustomerRow,
    DecisionRow,
    DomainRow,
    ShadowValidationRow,
    SimulationRunRow,
    StrategyDraftRow,
    StrategyVersionRow,
    ValidationRunRow,
)

EXAMPLES = REPO_DIR / "examples" / "domains"
SEEDED = ("hospitality", "gpu-compute", "edtech", "ecommerce", "banking")
ANALYST = Principal("analyst", "seed-analyst")
MANAGER = Principal("manager", "seed-manager")

CUSTOMERS: list[dict[str, Any]] = [
    {"id": "guest-gold", "domain_id": "hospitality", "name": "Asha (Gold member)", "profile": {"loyalty_tier": "Gold", "loyalty_score": 82}},
    {"id": "guest-new", "domain_id": "hospitality", "name": "Ben (new guest)", "profile": {"loyalty_tier": "Standard", "loyalty_score": 5}},
    {"id": "lab-research", "domain_id": "gpu-compute", "name": "Research lab", "profile": {}},
    {"id": "student-returning", "domain_id": "edtech", "name": "Chen (returning student)", "profile": {"student_tier": "returning", "sessions_completed": 18}},
    {"id": "student-new", "domain_id": "edtech", "name": "Dana (new student)", "profile": {"student_tier": "new", "sessions_completed": 0}},
    {"id": "shopper-plus", "domain_id": "ecommerce", "name": "Eli (Plus member)", "profile": {"membership": "plus", "orders_last_year": 14}},
    {"id": "shopper-new", "domain_id": "ecommerce", "name": "Fatima (first order)", "profile": {"membership": "none", "orders_last_year": 0}},
    {"id": "borrower-prime", "domain_id": "banking", "name": "Gita (8-year customer)", "profile": {"debt_to_income": 22, "relationship_years": 8, "salary_account": True}},
    {"id": "borrower-new", "domain_id": "banking", "name": "Hari (new to bank)", "profile": {"debt_to_income": 48, "relationship_years": 0, "salary_account": False}},
]


def load_example(name: str) -> dict[str, Any]:
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


def wipe(s: Session) -> None:
    for model in (CandidateEvaluationRow, ShadowValidationRow, DecisionRow, SimulationRunRow, ValidationRunRow,
                  StrategyVersionRow, StrategyDraftRow, CustomerRow, AuditLogRow, DomainRow):
        s.execute(delete(model))
    s.commit()


def seed(sessions: sessionmaker[Session], reset: bool = False) -> list[str]:
    """Create and publish the demo domains. Uses the same services as the API (validation included)."""
    from app.services import strategy as svc
    from app.services.common import AppContext
    from app.validation import ValidationService
    from app.optimizer import SimulatedPriceOptimizer
    from app.optimizer.factory import OptimizerStatus

    ctx = AppContext(settings=get_settings(), sessions=sessions, optimizer=SimulatedPriceOptimizer(),
                     optimizer_status=OptimizerStatus(), llm=None, persona_llm=None, validation=ValidationService())
    with sessions() as s:
        if reset:
            wipe(s)
        elif s.scalars(select(DomainRow)).first() is not None:
            return []
    created = []
    for name in SEEDED:
        bundle = load_example(name)
        svc.create_domain(ctx, ANALYST, bundle)
        result = svc.validate_draft(ctx, ANALYST, bundle["domain"]["id"])
        if not result["report"]["ok"]:
            raise RuntimeError(f"seed strategy for {name} does not validate: {result['report']['issues']}")
        version = svc.approve(ctx, MANAGER, bundle["domain"]["id"], notes="Seeded demo strategy (synthetic data).")
        svc.publish(ctx, MANAGER, bundle["domain"]["id"], version["id"])
        created.append(bundle["domain"]["id"])
    with sessions() as s:
        for c in CUSTOMERS:
            s.merge(CustomerRow(**c))
        s.commit()
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="wipe all data before seeding")
    args = parser.parse_args()
    settings = get_settings()
    run_migrations(settings.database_url)
    sessions = make_session_factory(make_engine(settings.database_url))
    created = seed(sessions, reset=args.reset)
    print(f"Seeded: {', '.join(created)}" if created else "Database already has data (use --reset to reseed).")


if __name__ == "__main__":
    main()
