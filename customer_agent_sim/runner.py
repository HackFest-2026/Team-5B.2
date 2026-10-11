"""
runner.py -- Simulation orchestrator and batch evaluation runner.

Executes controlled Monte Carlo experiments across populations of CustomerAgents.
Supports Common Random Numbers (CRN) for fair candidate-price and scenario comparisons.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple
import numpy as np

from customer_agent_sim.agent import CustomerAgent
from customer_agent_sim.decision_policy import BaseDecisionPolicy, StandardDecisionPolicy
from customer_agent_sim.metrics import calculate_variance, wilson_score_interval
from customer_agent_sim.models import (
    AgentDecision,
    CandidatePriceResult,
    DecisionOutcome,
    Offer,
    SimulationMetrics,
)


def simulate_population_decisions(
    population: List[CustomerAgent],
    offer: Offer,
    policy: Optional[BaseDecisionPolicy] = None,
    seed: Optional[int] = None,
) -> List[AgentDecision]:
    """
    Simulate individual agent choices on a single offer.

    Args:
        population: Customer agents.
        offer: Offer to evaluate.
        policy: Behavioral decision policy.
        seed: Random seed for stochastic sampling.

    Returns:
        List of AgentDecision records.
    """
    if policy is None:
        policy = StandardDecisionPolicy()

    rng = np.random.default_rng(seed)
    decisions: List[AgentDecision] = []

    for agent in population:
        decision = agent.evaluate(offer, policy=policy, rng=rng)
        decisions.append(decision)

    return decisions


def simulate_offer(
    population: List[CustomerAgent],
    offer: Offer,
    policy: Optional[BaseDecisionPolicy] = None,
    seed: Optional[int] = None,
    confidence_level: float = 0.95,
) -> SimulationMetrics:
    """
    Execute simulation across population and calculate aggregate performance metrics.

    Args:
        population: Customer agents.
        offer: Offer to evaluate.
        policy: Behavioral decision policy.
        seed: Random seed for reproducible Monte Carlo sampling.
        confidence_level: Confidence level for Wilson score interval.

    Returns:
        SimulationMetrics with conversion rate, confidence intervals, and breakdown.
    """
    decisions = simulate_population_decisions(population, offer, policy=policy, seed=seed)
    n_total = len(decisions)
    if n_total == 0:
        raise ValueError("Cannot simulate an empty population.")

    n_purchases = sum(1 for d in decisions if d.outcome == DecisionOutcome.BUY)
    n_defers = sum(1 for d in decisions if d.outcome == DecisionOutcome.DEFER)
    n_rejects = sum(1 for d in decisions if d.outcome == DecisionOutcome.REJECT)

    conversion_rate = n_purchases / n_total
    deferral_rate = n_defers / n_total
    rejection_rate = n_rejects / n_total

    ci_lower, ci_upper = wilson_score_interval(
        successes=n_purchases,
        total=n_total,
        confidence=confidence_level,
    )

    buy_probs = [d.p_buy for d in decisions]
    mean_buy_p = float(np.mean(buy_probs))
    var_buy_p = calculate_variance(buy_probs)

    expected_revenue = offer.price * conversion_rate
    expected_profit: Optional[float] = None
    if offer.unit_cost is not None:
        unit_margin = offer.price - offer.unit_cost
        expected_profit = unit_margin * conversion_rate

    return SimulationMetrics(
        population_size=n_total,
        n_purchases=n_purchases,
        n_defers=n_defers,
        n_rejects=n_rejects,
        conversion_rate=conversion_rate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        deferral_rate=deferral_rate,
        rejection_rate=rejection_rate,
        mean_buy_probability=mean_buy_p,
        probability_variance=var_buy_p,
        expected_revenue=expected_revenue,
        expected_profit=expected_profit,
    )


def simulate_candidates(
    population: List[CustomerAgent],
    candidate_prices: List[float],
    base_offer: Offer,
    ml_probabilities: Optional[Dict[float, float]] = None,
    policy: Optional[BaseDecisionPolicy] = None,
    seed: Optional[int] = 42,
    confidence_level: float = 0.95,
) -> List[CandidatePriceResult]:
    """
    Simulate a set of candidate prices against the same population using Common Random Numbers.

    Ranking Objective:
      - If `unit_cost` is available: candidates are ranked by expected unit profit:
          E[Profit] = (price - unit_cost) * simulated_conversion
      - If `unit_cost` is unavailable: candidates are ranked by expected revenue:
          E[Revenue] = price * simulated_conversion
      Compatible metrics are computed and compared for ML probabilities.

    Args:
        population: Customer agents.
        candidate_prices: List of prices to evaluate.
        base_offer: Baseline offer template (metadata, unit cost, etc.).
        ml_probabilities: Optional map of price -> ML purchase probability.
        policy: Behavioral decision policy.
        seed: Fixed seed for Common Random Numbers across candidates.
        confidence_level: Wilson interval confidence.

    Returns:
        List of CandidatePriceResult with simulation conversion, profit/revenue, and rankings.
    """
    results: List[CandidatePriceResult] = []
    has_cost = base_offer.unit_cost is not None
    metric_name = "profit" if has_cost else "revenue"

    for price in candidate_prices:
        # Create an offer copy with this candidate price
        cand_offer = Offer(
            price=float(price),
            base_price=base_offer.base_price,
            perceived_value=base_offer.perceived_value,
            competitor_price=base_offer.competitor_price,
            urgency=base_offer.urgency,
            scarcity=base_offer.scarcity,
            unit_cost=base_offer.unit_cost,
            context=copy.deepcopy(base_offer.context),
        )

        metrics = simulate_offer(
            population=population,
            offer=cand_offer,
            policy=policy,
            seed=seed,  # Common Random Numbers
            confidence_level=confidence_level,
        )

        ml_prob = ml_probabilities.get(price) if ml_probabilities else None
        ml_profit = None
        ml_revenue = None
        if ml_prob is not None:
            ml_revenue = price * ml_prob
            if has_cost:
                ml_profit = (price - base_offer.unit_cost) * ml_prob

        prob_gap = abs(ml_prob - metrics.conversion_rate) if ml_prob is not None else None
        sim_metric_val = metrics.expected_profit if has_cost else metrics.expected_revenue

        results.append(
            CandidatePriceResult(
                price=float(price),
                sim_conversion=metrics.conversion_rate,
                sim_ci_lower=metrics.ci_lower,
                sim_ci_upper=metrics.ci_upper,
                sim_expected_profit=metrics.expected_profit,
                sim_expected_revenue=metrics.expected_revenue,
                ranking_metric_name=metric_name,
                ranking_metric_value=float(sim_metric_val if sim_metric_val is not None else 0.0),
                ml_probability=ml_prob,
                ml_expected_profit=ml_profit,
                ml_expected_revenue=ml_revenue,
                probability_gap=prob_gap,
            )
        )

    # Compute rankings
    # Simulation ranks (1 = best)
    sim_sort_key = (
        lambda r: r.sim_expected_profit if has_cost else (r.sim_expected_revenue or 0.0)
    )
    sorted_by_sim = sorted(results, key=sim_sort_key, reverse=True)
    for rank, res in enumerate(sorted_by_sim, start=1):
        res.rank_sim = rank

    # ML ranks if ML probabilities are available
    if ml_probabilities:
        ml_sort_key = (
            lambda r: (r.ml_expected_profit if r.ml_expected_profit is not None else -float("inf"))
            if has_cost
            else (r.ml_expected_revenue if r.ml_expected_revenue is not None else -float("inf"))
        )
        sorted_by_ml = sorted(results, key=ml_sort_key, reverse=True)
        for rank, res in enumerate(sorted_by_ml, start=1):
            res.rank_ml = rank

    return results
