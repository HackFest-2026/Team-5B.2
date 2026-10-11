"""API test fixtures: a fresh SQLite file per test, seeded with the three demo domains."""

from __future__ import annotations

from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.optimizer import SimulatedPriceOptimizer


def H(role: str, user: str = "") -> dict[str, str]:
    return {"Authorization": f"Bearer dev:{role}:{user or role + '1'}"}


ANALYST, ANALYST2, MANAGER, MANAGER2, ADMIN = H("analyst", "ana"), H("analyst", "bob"), H("manager", "maya"), H("manager", "max"), H("admin", "root")
CUSTOMER = H("customer", "guest-gold")


@pytest.fixture
def make_client(tmp_path) -> Callable[..., TestClient]:
    def factory(validation_mode: str = "off", llm: Any = None, optimizer: Any = None, seed: bool = True, db: str = "test.db", market: Any = None,
                **settings: Any) -> TestClient:
        s = Settings(database_url=f"sqlite:///{(tmp_path / db).as_posix()}", validation_mode=validation_mode,
                     validation_pop_quote=150, validation_pop_sim=200, _env_file=None, **settings)
        app = create_app(s, optimizer=optimizer or SimulatedPriceOptimizer(), llm=llm, seed=seed, market=market)
        return TestClient(app)

    return factory


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def hotel_quote(**over: Any) -> dict[str, Any]:
    body = {"domain_id": "hospitality", "product_id": "DLX", "attributes": {"check_in": "2026-10-24", "nights": 2},
            "as_of": "2026-10-11"}
    body.update(over)
    return body
