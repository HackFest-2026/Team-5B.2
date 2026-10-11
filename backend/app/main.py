"""FastAPI application factory. Run: ``uvicorn app.main:app --reload --port 8000`` from backend/."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.db import make_engine, make_session_factory, run_migrations
from app.errors import install_handlers
from app.llm import build_llm
from app.market import build_researcher
from app.optimizer import OptimizerStatus, build_optimizer
from app.routers import domains, pricing
from app.services.common import AppContext
from app.validation import ValidationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _noisy in ("alembic", "httpx", "httpx2", "google_genai"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def create_app(settings: Optional[Settings] = None, optimizer=None, llm=None, seed: Optional[bool] = None, market=None) -> FastAPI:
    settings = settings or get_settings()
    run_migrations(settings.database_url)
    sessions = make_session_factory(make_engine(settings.database_url))
    status = OptimizerStatus()
    ctx = AppContext(
        settings=settings,
        sessions=sessions,
        optimizer=optimizer or build_optimizer(settings, status),
        optimizer_status=status,
        llm=llm if llm is not None else build_llm(settings),
        persona_llm=None if settings.persona_text_provider == "none" else build_llm(settings, settings.persona_text_provider),
        validation=ValidationService(seed=settings.validation_seed),
        market=market if market is not None else build_researcher(settings),
    )
    if settings.seed_on_startup if seed is None else seed:
        from app.seed.seed import seed as run_seed

        run_seed(sessions)

    app = FastAPI(title="Monetize360 API", version="1.0.0",
                  description="Universal configurable pricing engine. Auth is development-grade (Bearer dev:<role>:<user>).")
    app.state.ctx = ctx
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list(), allow_methods=["*"], allow_headers=["*"])
    install_handlers(app)
    app.include_router(domains.router)
    app.include_router(pricing.router)
    return app


def __getattr__(name: str):  # lazily build the module-level app for `uvicorn app.main:app`
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
