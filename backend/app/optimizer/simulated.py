"""Transparent, deterministic stand-in for the ML model (fallback and tests).

Uses the same 14 features as the real model and a hand-written logistic response, so the two are
interchangeable behind the PriceOptimizer interface. Its outputs are simulated, never presented
as a trained model or as real conversion rates.
"""

from __future__ import annotations

import math

from app.schemas.optimizer import CandidateEvaluation, ModelInfo, OptimizerRequest, OptimizerResponse

from .base import finish
from .features import build_features

COEFFICIENTS = {
    "intercept": 1.2,
    "sensitivity_x_log_ratio": -4.0,
    "over_competitor": -2.5,
    "urgency": 0.6,
    "scarcity": 0.4,
    "demand": 0.3,
    "loyalty": 0.5,
    "value_gap": 1.5,
    "promotion": 0.3,
}


class SimulatedPriceOptimizer:
    info = ModelInfo(name="simulated-logit", version="sim-1.0", kind="simulated", training_data="n/a")

    def evaluate(self, req: OptimizerRequest, overrides: dict[str, float] | None = None) -> OptimizerResponse:
        batch = build_features(req, overrides)
        k = COEFFICIENTS
        by_price = {c.price: c for c in req.candidates}
        evals = []
        for price, f in zip(batch.prices, batch.rows):
            sens = max(f.customer_sensitivity, 0.05)
            z = (k["intercept"]
                 + k["sensitivity_x_log_ratio"] * (0.5 + sens) * f.log_price_ratio
                 + k["over_competitor"] * max(0.0, f.price_vs_competitor - 1.0)
                 + k["urgency"] * f.urgency_score
                 + k["scarcity"] * f.scarcity_score
                 + k["demand"] * (f.demand_pressure - 1.0)
                 + k["loyalty"] * f.customer_loyalty_score
                 + k["value_gap"] * f.perceived_value_gap
                 + k["promotion"] * f.is_promotion_active)
            p = round(min(max(1.0 / (1.0 + math.exp(-z)), 0.001), 0.999), 6)
            evals.append(CandidateEvaluation(
                price=price, features=f, p_buy=p,
                expected_revenue=round(p * price, 6), expected_profit=round(p * (price - req.unit_cost), 6),
                eligible=by_price[price].eligible, selection_status="NOT_SELECTED",
                explanation=f"Simulated: P(buy) {p:.0%} at {f.price_ratio:.2f}× base, {f.price_vs_competitor:.2f}× competitor",
            ))
        return finish(evals, req, self.info, batch.warnings)
