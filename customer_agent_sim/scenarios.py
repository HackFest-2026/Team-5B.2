"""
scenarios.py -- Configurable market scenarios and stress tests for robustness evaluation.

Ensures ceteris-paribus comparisons by reusing the exact same customer agent population
while applying explicit, reproducible shifts in budgets, needs, competitor prices,
and urgency/scarcity dynamics.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from customer_agent_sim.agent import CustomerAgent
from customer_agent_sim.models import Offer, SimulationMetrics
from customer_agent_sim.runner import simulate_offer


@dataclass
class ScenarioDefinition:
    """Explicit parameters governing a scenario transformation."""
    name: str
    description: str
    budget_factor: float = 1.0
    need_factor: float = 1.0
    urgency_override: Optional[float] = None
    scarcity_override: Optional[float] = None
    competitor_factor: Optional[float] = None
    perceived_value_factor: float = 1.0


# Registry of predefined market and consumer scenarios
STANDARD_SCENARIOS: Dict[str, ScenarioDefinition] = {
    "budget_squeeze": ScenarioDefinition(
        name="budget_squeeze",
        description="Economic downturn or inflationary pressure: customer budgets reduced by 25%.",
        budget_factor=0.75,
    ),
    "budget_expansion": ScenarioDefinition(
        name="budget_expansion",
        description="Payday or stimulus boom: customer budgets increased by 25%.",
        budget_factor=1.25,
    ),
    "high_urgency": ScenarioDefinition(
        name="high_urgency",
        description="High urgency / immediate requirement: urgency raised to 0.90, need intensity +25%.",
        urgency_override=0.90,
        need_factor=1.25,
    ),
    "low_urgency": ScenarioDefinition(
        name="low_urgency",
        description="Casual comparison shopping: urgency dropped to 0.10, need intensity -30%.",
        urgency_override=0.10,
        need_factor=0.70,
    ),
    "competitor_undercut": ScenarioDefinition(
        name="competitor_undercut",
        description="Aggressive competitor price discount: competitor price drops by 20%.",
        competitor_factor=0.80,
    ),
    "competitor_premium": ScenarioDefinition(
        name="competitor_premium",
        description="Competitor raises prices by 20% due to cost increases or supply crunch.",
        competitor_factor=1.20,
    ),
    "high_scarcity": ScenarioDefinition(
        name="high_scarcity",
        description="Stock scarcity alert: scarcity signal set to 0.85.",
        scarcity_override=0.85,
    ),
    "perceived_value_drop": ScenarioDefinition(
        name="perceived_value_drop",
        description="Perceived product value drops by 20% (e.g. brand degradation, negative reviews).",
        perceived_value_factor=0.80,
    ),
    "perceived_value_boost": ScenarioDefinition(
        name="perceived_value_boost",
        description="Perceived value boosts by 20% (e.g. positive PR, product redesign).",
        perceived_value_factor=1.20,
    ),
    "stagflation_stress": ScenarioDefinition(
        name="stagflation_stress",
        description="Combined adverse stress test: budgets -25%, competitor drops 15%, perceived value drops 15%.",
        budget_factor=0.75,
        competitor_factor=0.85,
        perceived_value_factor=0.85,
    ),
}


def apply_scenario(
    population: List[CustomerAgent],
    offer: Offer,
    scenario: Union[ScenarioDefinition, str],
) -> Tuple[List[CustomerAgent], Offer]:
    """
    Transform the population state and offer according to a scenario specification.

    Crucially, the underlying agents' stable traits are preserved intact, preventing
    sampling artifacts from confounding scenario sensitivity.

    Args:
        population: Baseline agent population.
        offer: Baseline offer.
        scenario: ScenarioDefinition instance or name of standard scenario.

    Returns:
        (transformed_population, transformed_offer)
    """
    if isinstance(scenario, str):
        if scenario not in STANDARD_SCENARIOS:
            raise KeyError(f"Unknown scenario '{scenario}'. Available: {list(STANDARD_SCENARIOS.keys())}")
        defn = STANDARD_SCENARIOS[scenario]
    else:
        defn = scenario

    # 1. Transform agent population state
    transformed_population: List[CustomerAgent] = []
    for agent in population:
        new_budget = agent.state.budget_ratio * defn.budget_factor
        new_need = agent.state.need_intensity * defn.need_factor
        new_agent = agent.with_state_update(
            budget_ratio=new_budget,
            need_intensity=new_need,
        )
        transformed_population.append(new_agent)

    # 2. Transform offer
    new_urgency = defn.urgency_override if defn.urgency_override is not None else offer.urgency
    new_scarcity = defn.scarcity_override if defn.scarcity_override is not None else offer.scarcity
    new_perceived_value = offer.perceived_value * defn.perceived_value_factor

    new_comp_price = offer.competitor_price
    if defn.competitor_factor is not None:
        if new_comp_price is not None:
            new_comp_price = new_comp_price * defn.competitor_factor
        else:
            # Do NOT invent a competitor price if none exists in the market
            new_comp_price = None

    transformed_offer = Offer(
        price=offer.price,
        base_price=offer.base_price,
        perceived_value=new_perceived_value,
        competitor_price=new_comp_price,
        urgency=new_urgency,
        scarcity=new_scarcity,
        unit_cost=offer.unit_cost,
        context=copy.deepcopy(offer.context),
    )

    return transformed_population, transformed_offer


def evaluate_scenarios(
    population: List[CustomerAgent],
    offer: Offer,
    scenario_names: List[str],
    seed: Optional[int] = 42,
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate an offer across multiple market scenarios against baseline using Common Random Numbers.

    Returns:
        Dictionary mapping scenario name -> detailed outcome metrics and delta from baseline.
    """
    # Evaluate baseline
    base_metrics = simulate_offer(
        population=population,
        offer=offer,
        seed=seed,
        confidence_level=confidence_level,
    )

    results: Dict[str, Dict[str, Any]] = {
        "baseline": {
            "description": "Unperturbed baseline market environment",
            "applicable": True,
            "conversion_rate": base_metrics.conversion_rate,
            "ci_lower": base_metrics.ci_lower,
            "ci_upper": base_metrics.ci_upper,
            "delta_from_baseline": 0.0,
            "pct_change": 0.0,
            "purchases": base_metrics.n_purchases,
        }
    }

    for name in scenario_names:
        defn = STANDARD_SCENARIOS.get(name)
        desc = defn.description if defn else f"Custom scenario {name}"

        # If scenario specifically tests competitor price shifts but offer has no competitor, mark inapplicable
        if defn and defn.competitor_factor is not None and offer.competitor_price is None:
            if (
                defn.budget_factor == 1.0
                and defn.need_factor == 1.0
                and defn.perceived_value_factor == 1.0
                and defn.urgency_override is None
                and defn.scarcity_override is None
            ):
                results[name] = {
                    "description": desc,
                    "applicable": False,
                    "reason": f"Scenario '{name}' requires competitor_price, but offer has no competitor benchmark.",
                    "conversion_rate": base_metrics.conversion_rate,
                    "ci_lower": base_metrics.ci_lower,
                    "ci_upper": base_metrics.ci_upper,
                    "delta_from_baseline": 0.0,
                    "pct_change": 0.0,
                    "purchases": base_metrics.n_purchases,
                }
                continue

        scen_pop, scen_offer = apply_scenario(population, offer, name)
        scen_metrics = simulate_offer(
            population=scen_pop,
            offer=scen_offer,
            seed=seed,  # Common Random Numbers
            confidence_level=confidence_level,
        )

        delta = scen_metrics.conversion_rate - base_metrics.conversion_rate
        pct_change = (delta / base_metrics.conversion_rate) if base_metrics.conversion_rate > 0 else 0.0

        results[name] = {
            "description": desc,
            "applicable": True,
            "conversion_rate": scen_metrics.conversion_rate,
            "ci_lower": scen_metrics.ci_lower,
            "ci_upper": scen_metrics.ci_upper,
            "delta_from_baseline": delta,
            "pct_change": pct_change,
            "purchases": scen_metrics.n_purchases,
        }

    return results
