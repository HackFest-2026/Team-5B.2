"""
customer_agent_sim -- Standalone, domain-agnostic customer-agent validation module.

Validates ML pricing recommendations against simulated heterogeneous consumer populations.
Pipeline:
  Pricing Engine -> ML Optimizer -> Customer-Agent Validation -> PASS / FAIL / REVIEW_REQUIRED
"""

from __future__ import annotations

from typing import Dict, List, Optional

from customer_agent_sim.agent import CustomerAgent
from customer_agent_sim.decision_policy import BaseDecisionPolicy, StandardDecisionPolicy
from customer_agent_sim.feedback import (
    FeedbackDiagnostic,
    bounded_validation_pipeline,
    extract_feedback_diagnostics,
)
from customer_agent_sim.models import (
    AgentDecision,
    AgentState,
    AgentTraits,
    CandidatePriceResult,
    DecisionOutcome,
    Offer,
    SimulationMetrics,
    ValidationConfig,
    ValidationReport,
    ValidationStatus,
)
from customer_agent_sim.population import PopulationGenerator, split_population_by_trait
from customer_agent_sim.runner import (
    simulate_candidates,
    simulate_offer,
    simulate_population_decisions,
)
from customer_agent_sim.scenarios import (
    STANDARD_SCENARIOS,
    ScenarioDefinition,
    apply_scenario,
    evaluate_scenarios,
)
from customer_agent_sim.validation import (
    evaluate_decision_gate,
    validate_behavioural_consistency,
    validate_ml_discrepancy,
    validate_population_heterogeneity,
    validate_ranking_stability,
    validate_scenario_robustness,
)


def validate_offer(
    offer: Offer,
    ml_probability: float,
    candidate_prices: Optional[List[float]] = None,
    ml_probabilities: Optional[Dict[float, float]] = None,
    config: Optional[ValidationConfig] = None,
    population: Optional[List[CustomerAgent]] = None,
    policy: Optional[BaseDecisionPolicy] = None,
) -> ValidationReport:
    """
    Primary public entrypoint for Monetize360 customer-agent validation.

    Args:
        offer: Domain-neutral Offer candidate.
        ml_probability: Predicted purchase probability from Monetize360's ML model.
        candidate_prices: Optional list of alternative candidate prices evaluated by optimizer.
        ml_probabilities: Optional mapping of candidate price -> ML probability.
        config: Optional validation configuration specifying thresholds and scenarios.
        population: Optional pre-generated customer population (generated if None).
        policy: Optional behavioral decision policy (defaults to StandardDecisionPolicy).

    Returns:
        ValidationReport containing decision gate status (PASS/FAIL/REVIEW_REQUIRED)
        and diagnostics across all five validation capabilities.
    """
    return evaluate_decision_gate(
        offer=offer,
        ml_probability=ml_probability,
        candidate_prices=candidate_prices,
        ml_probabilities=ml_probabilities,
        config=config,
        population=population,
        policy=policy,
    )


__all__ = [
    "Offer",
    "AgentTraits",
    "AgentState",
    "CustomerAgent",
    "PopulationGenerator",
    "split_population_by_trait",
    "BaseDecisionPolicy",
    "StandardDecisionPolicy",
    "DecisionOutcome",
    "ValidationStatus",
    "ValidationConfig",
    "ValidationReport",
    "SimulationMetrics",
    "CandidatePriceResult",
    "FeedbackDiagnostic",
    "ScenarioDefinition",
    "STANDARD_SCENARIOS",
    "validate_offer",
    "simulate_offer",
    "simulate_candidates",
    "simulate_population_decisions",
    "apply_scenario",
    "evaluate_scenarios",
    "validate_behavioural_consistency",
    "validate_population_heterogeneity",
    "validate_scenario_robustness",
    "validate_ranking_stability",
    "validate_ml_discrepancy",
    "extract_feedback_diagnostics",
    "bounded_validation_pipeline",
]
