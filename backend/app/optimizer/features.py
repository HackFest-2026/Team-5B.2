"""Build the 14 ML_Updated features from an OptimizerRequest (the single place this happens).

Formulas mirror ``ml_updated/features.py::build_candidate_features`` exactly (a test compares
them). Rank is computed over the full candidate set, eligible or not, matching training where
every quote had a fixed candidate list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.schemas.optimizer import ML_FEATURE_COLUMNS, MLFeatures, OptimizerRequest

# Defaults used by ml_updated/predict_best_price.py when a signal is unknown.
SIGNAL_DEFAULTS = {
    "customerSensitivity": 0.5,
    "customerLoyalty": 0.3,
    "urgency": 0.3,
    "scarcity": 0.3,
    "demandPressure": 1.0,
    "competitorDensity": 0.5,
    "promotionActive": 0.0,
    "timeDecay": 0.5,
}
# Ranges seen in training (README dataset schema). Values outside get an OOD warning.
TRAINING_RANGES = {
    "price_ratio": (0.6, 1.5),
    "price_vs_competitor": (0.6, 1.5),
    "customer_sensitivity": (0.0, 1.0),
    "customer_loyalty_score": (0.0, 1.0),
    "urgency_score": (0.0, 1.0),
    "scarcity_score": (0.0, 1.0),
    "demand_pressure": (0.2, 2.0),
    "competitor_density": (0.0, 1.0),
    "time_decay_factor": (0.0, 1.0),
}


@dataclass
class FeatureBatch:
    prices: list[float]
    rows: list[MLFeatures]
    warnings: list[str] = field(default_factory=list)

    def matrix(self) -> list[list[float]]:
        return [[float(getattr(r, c)) for c in ML_FEATURE_COLUMNS] for r in self.rows]


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def candidate_features(
    candidate_price: float, base_price: float, unit_cost: float, competitor_price: float, rank: float,
    sensitivity: float, loyalty: float, urgency: float, scarcity: float, demand: float, density: float,
    promo: int, time_decay: float, perceived_value_gap: float | None,
) -> MLFeatures:
    bp = max(base_price, 0.01)
    cp = max(competitor_price, 0.01)
    price_ratio = _clip(candidate_price / bp, 0.1, 10.0)
    if perceived_value_gap is None:
        perceived_value = bp * (1.0 + 0.15 * loyalty + 0.10 * (demand - 1.0))
        perceived_value_gap = (perceived_value - candidate_price) / bp
    return MLFeatures(
        price_ratio=round(price_ratio, 4),
        log_price_ratio=round(math.log(price_ratio), 4),
        price_vs_competitor=round(_clip(candidate_price / cp, 0.1, 10.0), 4),
        margin_ratio=round((candidate_price - unit_cost) / bp, 4),
        price_rank_normalized=round(rank, 4),
        customer_sensitivity=round(sensitivity, 4),
        customer_loyalty_score=round(loyalty, 4),
        urgency_score=round(urgency, 4),
        scarcity_score=round(scarcity, 4),
        demand_pressure=round(demand, 4),
        competitor_density=round(density, 4),
        is_promotion_active=int(promo),
        time_decay_factor=round(time_decay, 4),
        perceived_value_gap=round(perceived_value_gap, 4),
    )


def build_features(req: OptimizerRequest, overrides: dict[str, float] | None = None) -> FeatureBatch:
    """Features for every candidate. ``overrides`` replaces signals (e.g. a persona's sensitivity)."""
    sig = {**req.signals, **(overrides or {})}
    warnings: list[str] = []
    defaulted = []

    def get(name: str) -> float:
        if name in sig and sig[name] is not None:
            return float(sig[name])
        defaulted.append(name)
        return SIGNAL_DEFAULTS[name]

    sensitivity, loyalty = get("customerSensitivity"), get("customerLoyalty")
    urgency, scarcity, demand = get("urgency"), get("scarcity"), get("demandPressure")
    density, time_decay = get("competitorDensity"), get("timeDecay")
    promo = 1 if get("promotionActive") >= 0.5 else 0
    competitor = req.competitor_price
    if competitor is None or competitor <= 0:
        competitor = req.base_price
        defaulted.append("competitorPrice")
    if defaulted:
        warnings.append("DEFAULT_SIGNALS:" + ",".join(sorted(set(defaulted))))
    if req.perceived_value is None:
        warnings.append("DEFAULT_PERCEIVED_VALUE")

    prices = sorted(c.price for c in req.candidates)
    n = len(prices)
    rows = []
    for i, price in enumerate(prices):
        gap = None if req.perceived_value is None else (req.perceived_value - price) / max(req.base_price, 0.01)
        rows.append(candidate_features(
            price, req.base_price, req.unit_cost, competitor, i / max(n - 1, 1),
            sensitivity, loyalty, urgency, scarcity, demand, density, promo, time_decay, gap,
        ))
    ood = sorted({
        name for r in rows for name, (lo, hi) in TRAINING_RANGES.items()
        if not (lo - 1e-9 <= float(getattr(r, name)) <= hi + 1e-9)
    })
    warnings.extend(f"OOD_FEATURE:{name}" for name in ood)
    return FeatureBatch(prices=prices, rows=rows, warnings=warnings)
