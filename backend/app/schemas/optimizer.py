"""Contract between the pricing engine and any PriceOptimizer (ML model or simulator).

The engine sends checked candidates; the optimizer returns P(buy) and expected values for each.
The engine re-verifies the optimizer's choice against hard constraints, so the optimizer can
never override them.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from .common import Id, StrictModel
from .pricing_parts import CheckedCandidate
from .strategy import Objective

# Exactly the 14 feature columns of the ML_Updated model, in training order.
ML_FEATURE_COLUMNS: tuple[str, ...] = (
    "price_ratio", "log_price_ratio", "price_vs_competitor", "margin_ratio", "price_rank_normalized",
    "customer_sensitivity", "customer_loyalty_score", "urgency_score", "scarcity_score", "demand_pressure",
    "competitor_density", "is_promotion_active", "time_decay_factor", "perceived_value_gap",
)


class MLFeatures(StrictModel):
    price_ratio: float
    log_price_ratio: float
    price_vs_competitor: float
    margin_ratio: float
    price_rank_normalized: float
    customer_sensitivity: float
    customer_loyalty_score: float
    urgency_score: float
    scarcity_score: float
    demand_pressure: float
    competitor_density: float
    is_promotion_active: int
    time_decay_factor: float
    perceived_value_gap: float


class OptimizerRequest(StrictModel):
    domain_id: Id
    strategy_version_id: str
    base_price: float = Field(gt=0)
    unit_cost: float = Field(ge=0)
    competitor_price: Optional[float] = None
    perceived_value: Optional[float] = None
    signals: dict[str, float] = Field(default_factory=dict)
    candidates: list[CheckedCandidate]
    objective: Objective = "max_expected_profit"
    conversion_floor: float = Field(default=0.10, ge=0, le=1)


SelectionStatus = Literal["SELECTED", "NOT_SELECTED", "INELIGIBLE", "BELOW_CONVERSION_FLOOR"]


class CandidateEvaluation(StrictModel):
    price: float
    features: Optional[MLFeatures] = None
    p_buy: float = Field(ge=0, le=1)
    expected_revenue: float
    expected_profit: float
    eligible: bool
    selection_status: SelectionStatus
    explanation: str = ""


class ModelInfo(StrictModel):
    name: str
    version: str
    kind: Literal["ml", "simulated"]
    training_data: Literal["synthetic", "n/a"] = "n/a"
    artifact_sha256: Optional[str] = None


class OptimizerResponse(StrictModel):
    evaluations: list[CandidateEvaluation]
    selected_price: Optional[float] = None
    model: ModelInfo
    warnings: list[str] = Field(default_factory=list)
