"""
models.py -- Domain-neutral data structures and typed models for customer-agent simulation.

Defines schemas for:
  - Offer input data and validation
  - Customer agent traits and dynamic state
  - Decision outcomes and probability distributions
  - Simulation metrics and confidence intervals
  - Validation configuration, status, and reporting
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DecisionOutcome(str, Enum):
    """Possible customer agent decision outcomes."""
    BUY = "BUY"
    REJECT = "REJECT"
    DEFER = "DEFER"


class ValidationStatus(str, Enum):
    """Validation decision gate outcome."""
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass
class Offer:
    """
    Domain-neutral representation of a pricing candidate and its environment.

    Attributes:
        price: Candidate price evaluated by the agent (> 0).
        base_price: Stated catalog, reference, or anchor price (> 0).
        perceived_value: Perceived utility or customer valuation in monetary units (> 0).
        competitor_price: Optional prevailing market competitor price (> 0 if specified).
        urgency: Urgency signal in [0.0, 1.0] (0 = no time pressure, 1 = immediate need).
        scarcity: Scarcity signal in [0.0, 1.0] (0 = abundant supply, 1 = near stockout).
        unit_cost: Optional unit cost for profit calculation (>= 0 if specified).
        context: Optional generic key-value metadata for downstream adapters.
    """
    price: float
    base_price: float
    perceived_value: float
    competitor_price: Optional[float] = None
    urgency: float = 0.0
    scarcity: float = 0.0
    unit_cost: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"Offer.price must be strictly positive, got {self.price}")
        if self.base_price <= 0:
            raise ValueError(f"Offer.base_price must be strictly positive, got {self.base_price}")
        if self.perceived_value < 0:
            raise ValueError(f"Offer.perceived_value cannot be negative, got {self.perceived_value}")
        if self.competitor_price is not None and self.competitor_price <= 0:
            raise ValueError(f"Offer.competitor_price must be strictly positive when provided, got {self.competitor_price}")
        if not (0.0 <= self.urgency <= 1.0):
            raise ValueError(f"Offer.urgency must be within [0.0, 1.0], got {self.urgency}")
        if not (0.0 <= self.scarcity <= 1.0):
            raise ValueError(f"Offer.scarcity must be within [0.0, 1.0], got {self.scarcity}")
        if self.unit_cost is not None and self.unit_cost < 0:
            raise ValueError(f"Offer.unit_cost cannot be negative, got {self.unit_cost}")


@dataclass
class AgentTraits:
    """
    Stable customer behavioral traits that remain persistent across interactions.

    Attributes:
        price_sensitivity: Elasticity / responsiveness to price changes (>= 0).
        value_orientation: Weight placed on perceived surplus / quality (>= 0).
        risk_aversion: Hesitation when faced with high commitments or uncertainties (>= 0).
        patience: Willingness to postpone or defer a purchase for a better opportunity (>= 0).
        loyalty: Preference for the focal vendor mitigating competitor alternatives (>= 0).
        decision_noise: Degree of stochastic temperature in evaluation (>= 0.01).
    """
    price_sensitivity: float = 1.0
    value_orientation: float = 1.0
    risk_aversion: float = 1.0
    patience: float = 1.0
    loyalty: float = 1.0
    decision_noise: float = 0.40

    def __post_init__(self) -> None:
        for trait_name in ("price_sensitivity", "value_orientation", "risk_aversion", "patience", "loyalty"):
            val = getattr(self, trait_name)
            if val < 0.0:
                raise ValueError(f"Trait '{trait_name}' must be non-negative, got {val}")
        if self.decision_noise <= 0.0:
            raise ValueError(f"Trait 'decision_noise' must be strictly positive, got {self.decision_noise}")


@dataclass
class AgentState:
    """
    Dynamic state of a customer at the moment of evaluation.

    Attributes:
        budget_ratio: Ratio of customer's available budget to the reference price (> 0).
        need_intensity: Instantaneous strength of need (>= 0).
        reference_price_bias: Subjective reference anchor deviation as fraction (-1.0 to 2.0).
        memory: Optional extensible purchase history or memory record.
    """
    budget_ratio: float = 1.20
    need_intensity: float = 1.0
    reference_price_bias: float = 0.0
    memory: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.budget_ratio <= 0.0:
            raise ValueError(f"State.budget_ratio must be strictly positive, got {self.budget_ratio}")
        if self.need_intensity < 0.0:
            raise ValueError(f"State.need_intensity cannot be negative, got {self.need_intensity}")
        if not (-1.0 <= self.reference_price_bias <= 2.0):
            raise ValueError(f"State.reference_price_bias must be within [-1.0, 2.0], got {self.reference_price_bias}")


@dataclass
class AgentDecision:
    """Detailed record of an individual agent's decision."""
    agent_id: str
    outcome: DecisionOutcome
    p_buy: float
    p_defer: float
    p_reject: float
    utility_buy: float
    utility_defer: float


@dataclass
class SimulationMetrics:
    """Aggregate outcomes across a simulated population."""
    population_size: int
    n_purchases: int
    n_defers: int
    n_rejects: int
    conversion_rate: float
    ci_lower: float
    ci_upper: float
    deferral_rate: float
    rejection_rate: float
    mean_buy_probability: float
    probability_variance: float
    expected_revenue: float = 0.0
    expected_profit: Optional[float] = None


@dataclass
class CandidatePriceResult:
    """Simulation and ML metrics for a specific candidate price."""
    price: float
    sim_conversion: float
    sim_ci_lower: float
    sim_ci_upper: float
    sim_expected_profit: Optional[float] = None
    sim_expected_revenue: Optional[float] = None
    ranking_metric_name: str = "profit"  # "profit" or "revenue"
    ranking_metric_value: float = 0.0
    ml_probability: Optional[float] = None
    ml_expected_profit: Optional[float] = None
    ml_expected_revenue: Optional[float] = None
    probability_gap: Optional[float] = None
    rank_ml: Optional[int] = None
    rank_sim: Optional[int] = None


@dataclass
class ValidationConfig:
    """
    Configuration parameters and thresholds for the validation decision gate.
    """
    population_size: int = 1000
    seed: int = 42
    minimum_simulated_conversion: float = 0.05
    max_allowed_probability_gap: float = 0.15
    confidence_level: float = 0.95
    ranking_tolerance_rank_drop: int = 2
    ranking_tolerance_profit_gap: float = 0.12
    min_heterogeneity_variance: float = 0.002
    max_retry_attempts: int = 3
    scenarios: List[str] = field(default_factory=lambda: [
        "budget_squeeze",
        "competitor_undercut",
        "high_urgency",
        "perceived_value_drop",
    ])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """
    Structured outcome of the customer-agent validation process.
    """
    status: ValidationStatus
    offer: Offer
    ml_probability: float
    simulated_conversion: float
    probability_gap: float
    ci_lower: float
    ci_upper: float
    population_size: int
    n_purchases: int
    consistency_results: Dict[str, Any]
    heterogeneity_results: Dict[str, Any]
    scenario_results: Dict[str, Any]
    ranking_results: Optional[Dict[str, Any]]
    discrepancy_results: Dict[str, Any]
    warnings: List[str]
    recommendations: List[str]
    reason: str
    minimum_simulated_conversion: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def summary_text(self) -> str:
        """Human-readable multi-line summary of the validation report."""
        lines = [
            f"=== Monetize360 Customer-Agent Validation Gate ===",
            f"Status: {self.status.value}",
            f"Reason: {self.reason}",
            f"Candidate Price: ${self.offer.price:.2f} (Base: ${self.offer.base_price:.2f})",
            f"ML Predicted Probability: {self.ml_probability:.2%}",
            f"Simulated Conversion:     {self.simulated_conversion:.2%} "
            f"[95% CI: {self.ci_lower:.2%} - {self.ci_upper:.2%}]",
            f"Probability Gap:          {self.probability_gap:.2%} "
            f"({'EXCEEDS TOLERANCE' if self.discrepancy_results.get('gap_exceeded') else 'WITHIN TOLERANCE'})",
            f"Population Size:          {self.population_size} ({self.n_purchases} purchases)",
        ]
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if self.recommendations:
            lines.append("Recommendations:")
            for r in self.recommendations:
                lines.append(f"  - {r}")
        return "\n".join(lines)
