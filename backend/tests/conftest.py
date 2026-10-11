"""Shared test fixtures.

Engine tests use a deliberately abstract "widgets" domain so they never depend on an industry.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from app.schemas import (
    CandidateEvaluation,
    DomainBundle,
    DomainConfig,
    ModelInfo,
    OptimizerRequest,
    OptimizerResponse,
    PriceRequest,
    Product,
    Rule,
    StrategyConfig,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "domains"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle(name: str) -> DomainBundle:
    return DomainBundle.model_validate(load_json(EXAMPLES / f"{name}.json"))


WIDGET_DOMAIN = {
    "id": "widgets",
    "name": "Widgets",
    "currency": "USD",
    "output_unit": {"label": "USD", "prefix": "$", "decimals": 2},
    "pipeline": ["base", "demand", "adjust"],
    "attributes": [
        {"key": "qty", "label": "Quantity", "type": "number", "source": "request", "default": 1, "min": 1, "max": 100},
        {"key": "colour", "label": "Colour", "type": "enum", "allowed_values": ["red", "blue", "green"], "source": "request", "default": "red"},
        {"key": "express", "label": "Express", "type": "boolean", "source": "request", "default": False},
        {"key": "start", "label": "Start date", "type": "date", "source": "request", "default": "2026-02-01"},
        {"key": "stock", "label": "Stock", "type": "number", "source": "inventory", "default": 50, "semantic": "scarcity",
         "normalize": {"min": 0, "max": 100, "invert": True}},
        {"key": "tier", "label": "Tier", "type": "enum", "allowed_values": ["Standard", "Gold"], "source": "profile", "default": "Standard"},
        {"key": "lead", "label": "Lead days", "type": "number", "source": "computed", "expression": "daysBetween(as_of, start)"},
        {"key": "lead_weeks", "label": "Lead weeks", "type": "number", "source": "computed", "expression": "floor(lead / 7)"},
        {"key": "rival", "label": "Rival price", "type": "number", "source": "inventory", "semantic": "competitorPrice"},
    ],
    "products": [
        {"id": "W1", "name": "Widget", "base_price": 100, "cost": 40, "inventory": {"stock": 5}, "previous_price": 100},
    ],
}


def rule(id: str, condition: Optional[dict] = None, action: Optional[dict] = None, **over: Any) -> Rule:
    data = {
        "id": id, "name": over.pop("name", id), "priority": over.pop("priority", 10), "stage": over.pop("stage", "adjust"),
        "condition": condition if condition is not None else {"all": []},
        "action": action or {"type": "none"},
        **over,
    }
    return Rule.model_validate(data)


BASE_RULES = [
    {"id": "bulk", "name": "bulk", "condition": {"attr": "qty", "op": ">=", "value": 10}, "action": {"type": "percent", "value": -10}, "stage": "adjust", "priority": 10},
    {"id": "scarce", "name": "scarce", "condition": {"attr": "stock", "op": "<", "value": 10}, "action": {"type": "percent", "value": 20}, "stage": "demand", "priority": 10},
    {"id": "express", "name": "express", "condition": {"attr": "express", "op": "==", "value": True}, "action": {"type": "fixed", "value": 15}, "stage": "adjust", "priority": 20},
    {"id": "gold", "name": "gold", "condition": {"attr": "tier", "op": "==", "value": "Gold"}, "action": {"type": "percent", "value": -5}, "stage": "adjust", "priority": 30},
]


@pytest.fixture
def domain() -> DomainConfig:
    return DomainConfig.model_validate(WIDGET_DOMAIN)


@pytest.fixture
def product(domain: DomainConfig) -> Product:
    return domain.products[0]


def make_strategy(**over: Any) -> StrategyConfig:
    data: dict[str, Any] = {
        "domain_id": "widgets",
        "name": "Widget strategy",
        "rules": BASE_RULES,
        "constraints": [{"type": "rounding", "value": 0.5}],
        "conflict_policy": {"mode": "stack", "combine": "multiplicative"},
    }
    data.update(over)
    return StrategyConfig.model_validate(data)


@pytest.fixture
def strategy() -> StrategyConfig:
    return make_strategy()


def req(attributes: Optional[dict] = None, **over: Any) -> PriceRequest:
    return PriceRequest.model_validate(
        {"domain_id": "widgets", "product_id": "W1", "attributes": attributes or {}, "as_of": "2026-01-01", **over}
    )


class FakeOptimizer:
    """Deterministic, transparent stand-in for a PriceOptimizer (tests only)."""

    info = ModelInfo(name="fake-linear", version="test-1", kind="simulated")

    def __init__(self, p_buy: Optional[Callable[[float, float], float]] = None, pick: Optional[Callable] = None):
        self.p_buy = p_buy or (lambda price, base: max(0.0, min(1.0, 1.6 - 1.0 * price / base)))
        self.pick = pick

    def evaluate(self, r: OptimizerRequest) -> OptimizerResponse:
        evals = []
        for c in r.candidates:
            p = self.p_buy(c.price, r.base_price)
            evals.append(CandidateEvaluation(
                price=c.price, p_buy=p, expected_revenue=p * c.price, expected_profit=p * (c.price - r.unit_cost),
                eligible=c.eligible, selection_status="INELIGIBLE" if not c.eligible else "NOT_SELECTED",
            ))
        eligible = [e for e in evals if e.eligible and e.p_buy >= r.conversion_floor]
        selected = max(eligible, key=lambda e: (e.expected_profit, -e.price)).price if eligible else None
        if self.pick is not None:
            selected = self.pick(evals)
        return OptimizerResponse(evaluations=evals, selected_price=selected, model=self.info)


def logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))
