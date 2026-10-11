"""
validation.py -- Core validation suite and decision gate for pricing recommendations.

Implements the five validation capabilities:
  A. Behavioural consistency (law of demand, perceived value, competitor contrast, urgency/scarcity)
  B. Population heterogeneity (subgroup purchase rates, variance, diversity detection)
  C. Scenario robustness (stress testing across market shocks with common populations)
  D. Ranking stability (candidate grid evaluation and rank drop diagnostics)
  E. ML disagreement and anomaly detection (probability gap, divergence factor analysis)

Synthesizes results into a deterministic decision gate: PASS, FAIL, or REVIEW_REQUIRED.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from customer_agent_sim.agent import CustomerAgent
from customer_agent_sim.decision_policy import BaseDecisionPolicy, StandardDecisionPolicy
from customer_agent_sim.models import (
    CandidatePriceResult,
    Offer,
    SimulationMetrics,
    ValidationConfig,
    ValidationReport,
    ValidationStatus,
)
from customer_agent_sim.population import PopulationGenerator, split_population_by_trait
from customer_agent_sim.runner import simulate_candidates, simulate_offer
from customer_agent_sim.scenarios import apply_scenario, evaluate_scenarios


def validate_behavioural_consistency(
    population: List[CustomerAgent],
    offer: Offer,
    policy: Optional[BaseDecisionPolicy] = None,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Test A: Behavioural consistency.

    Verifies whether simulated agents respond rationally and monotonically to controlled
    price, value, competitor, and urgency perturbations.
    """
    results: Dict[str, Any] = {
        "passed": True,
        "checks": {},
        "diagnostics": [],
    }

    # 1. Price Ladder Check (Law of demand / negative elasticity)
    # Test at 0.8x, 0.9x, 1.0x, 1.1x, 1.2x of candidate price
    multipliers = [0.80, 0.90, 1.00, 1.10, 1.20]
    prices = [round(offer.price * m, 2) for m in multipliers]
    ladder_results = simulate_candidates(
        population=population,
        candidate_prices=prices,
        base_offer=offer,
        policy=policy,
        seed=seed,
    )
    ladder_convs = [r.sim_conversion for r in ladder_results]

    # In standard settings, conversion should generally decrease (or not strictly increase) as price rises
    price_inversions = 0
    for i in range(len(ladder_convs) - 1):
        # Allow small statistical tolerance (+0.03) for Monte Carlo sampling variance
        if ladder_convs[i + 1] > ladder_convs[i] + 0.03:
            price_inversions += 1

    price_monotonic = (price_inversions == 0) and (ladder_convs[0] >= ladder_convs[-1])
    results["checks"]["price_monotonicity"] = {
        "passed": price_monotonic,
        "prices": prices,
        "conversions": [round(c, 4) for c in ladder_convs],
        "inversions": price_inversions,
    }
    if not price_monotonic:
        results["passed"] = False
        results["diagnostics"].append(
            f"Price monotonicity violated: demand increased with higher prices across {price_inversions} steps."
        )

    # 2. Perceived Value Sensitivity Check
    high_val_offer = Offer(
        price=offer.price,
        base_price=offer.base_price,
        perceived_value=offer.perceived_value * 1.30,
        competitor_price=offer.competitor_price,
        urgency=offer.urgency,
        scarcity=offer.scarcity,
        unit_cost=offer.unit_cost,
    )
    m_high_val = simulate_offer(population, high_val_offer, policy=policy, seed=seed)
    m_base = simulate_offer(population, offer, policy=policy, seed=seed)

    val_sensitive = m_high_val.conversion_rate >= (m_base.conversion_rate - 0.02)
    results["checks"]["perceived_value_sensitivity"] = {
        "passed": val_sensitive,
        "base_conversion": round(m_base.conversion_rate, 4),
        "high_value_conversion": round(m_high_val.conversion_rate, 4),
    }
    if not val_sensitive:
        results["diagnostics"].append("Perceived value sensitivity anomaly: higher value did not preserve conversion.")

    # 3. Competitor Contrast Check
    if offer.competitor_price is not None:
        comp_cheap_offer = Offer(
            price=offer.price,
            base_price=offer.base_price,
            perceived_value=offer.perceived_value,
            competitor_price=offer.competitor_price * 0.75,
            urgency=offer.urgency,
            scarcity=offer.scarcity,
            unit_cost=offer.unit_cost,
        )
        m_comp_cheap = simulate_offer(population, comp_cheap_offer, policy=policy, seed=seed)
        comp_sensitive = m_comp_cheap.conversion_rate <= (m_base.conversion_rate + 0.03)
        results["checks"]["competitor_sensitivity"] = {
            "passed": comp_sensitive,
            "base_conversion": round(m_base.conversion_rate, 4),
            "competitor_discount_conversion": round(m_comp_cheap.conversion_rate, 4),
        }
        if not comp_sensitive:
            results["diagnostics"].append("Competitor sensitivity anomaly: cheaper competitor did not reduce conversion.")

    # 4. Urgency & Scarcity Check
    urgent_offer = Offer(
        price=offer.price,
        base_price=offer.base_price,
        perceived_value=offer.perceived_value,
        competitor_price=offer.competitor_price,
        urgency=min(1.0, offer.urgency + 0.50),
        scarcity=min(1.0, offer.scarcity + 0.50),
        unit_cost=offer.unit_cost,
    )
    m_urgent = simulate_offer(population, urgent_offer, policy=policy, seed=seed)
    urgency_sensitive = m_urgent.conversion_rate >= (m_base.conversion_rate - 0.02)
    results["checks"]["urgency_scarcity_sensitivity"] = {
        "passed": urgency_sensitive,
        "base_conversion": round(m_base.conversion_rate, 4),
        "urgent_conversion": round(m_urgent.conversion_rate, 4),
    }

    return results


def validate_population_heterogeneity(
    population: List[CustomerAgent],
    offer: Offer,
    policy: Optional[BaseDecisionPolicy] = None,
    min_variance: float = 0.002,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Test B: Population heterogeneity.

    Ensures that customer traits generate meaningful variance in behavior rather than
    behaving as an identical cloned cohort.
    """
    # Check overall population probability variance first
    overall = simulate_offer(population, offer, policy=policy, seed=seed)
    variance = overall.probability_variance
    is_heterogeneous = bool(variance >= min_variance)

    subgroups = {}

    # Check subgroups by price sensitivity if population can be divided
    low_ps, high_ps = split_population_by_trait(population, "price_sensitivity", 0.50)
    if len(low_ps) > 0 and len(high_ps) > 0:
        m_low_ps = simulate_offer(low_ps, offer, policy=policy, seed=seed)
        m_high_ps = simulate_offer(high_ps, offer, policy=policy, seed=seed)
        subgroups["price_sensitivity"] = {
            "low_group_conversion": round(m_low_ps.conversion_rate, 4),
            "high_group_conversion": round(m_high_ps.conversion_rate, 4),
            "delta": round(m_low_ps.conversion_rate - m_high_ps.conversion_rate, 4),
        }

    # Check subgroups by budget ratio if population can be divided
    low_b, high_b = split_population_by_trait(population, "budget_ratio", 0.50)
    if len(low_b) > 0 and len(high_b) > 0:
        m_low_b = simulate_offer(low_b, offer, policy=policy, seed=seed)
        m_high_b = simulate_offer(high_b, offer, policy=policy, seed=seed)
        subgroups["budget_ratio"] = {
            "low_group_conversion": round(m_low_b.conversion_rate, 4),
            "high_group_conversion": round(m_high_b.conversion_rate, 4),
            "delta": round(m_high_b.conversion_rate - m_low_b.conversion_rate, 4),
        }

    return {
        "passed": is_heterogeneous,
        "probability_variance": round(variance, 6),
        "min_required_variance": min_variance,
        "subgroups": subgroups,
        "warning": None if is_heterogeneous else "Population shows near-zero variance: risk of cloned customer behavior.",
    }


def validate_scenario_robustness(
    population: List[CustomerAgent],
    offer: Offer,
    scenarios: List[str],
    min_conversion_threshold: float = 0.05,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Test C: Scenario robustness.

    Evaluates the recommended price across specified scenarios and detects fragile collapses.
    """
    scen_data = evaluate_scenarios(population, offer, scenarios, seed=seed)
    base_conv = scen_data["baseline"]["conversion_rate"]

    passed = True
    vulnerabilities = []

    for name, data in scen_data.items():
        if name == "baseline" or not data.get("applicable", True):
            continue
        conv = data["conversion_rate"]
        # Check if conversion falls below hard threshold or drops by more than 65% relatively
        if conv < min_conversion_threshold:
            passed = False
            vulnerabilities.append(f"Scenario '{name}' collapsed conversion below minimum threshold ({conv:.2%}).")
        elif base_conv > 0.10 and conv < 0.35 * base_conv:
            vulnerabilities.append(f"Scenario '{name}' triggered severe conversion drop (>65% drop to {conv:.2%}).")

    return {
        "passed": passed,
        "baseline_conversion": round(base_conv, 4),
        "scenarios": scen_data,
        "vulnerabilities": vulnerabilities,
    }


def validate_ranking_stability(
    population: List[CustomerAgent],
    candidate_prices: Optional[List[float]],
    base_offer: Offer,
    ml_probabilities: Optional[Dict[float, float]] = None,
    tolerance_rank_drop: int = 2,
    tolerance_profit_gap: float = 0.12,
    policy: Optional[BaseDecisionPolicy] = None,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Test D: Ranking stability.

    Assesses whether the ML-selected price remains competitive against alternative candidate prices.
    Evaluates both rank drop and relative profit/revenue gap against configured tolerances.
    If only a single price was provided, marks ranking stability as unavailable.
    """
    if not candidate_prices or len(candidate_prices) <= 1:
        return {
            "available": False,
            "reason": "Single-price validation: no candidate alternatives provided.",
        }

    # Ensure selected price is in candidate prices
    if base_offer.price not in candidate_prices:
        all_candidates = sorted(list(set(candidate_prices + [base_offer.price])))
    else:
        all_candidates = sorted(candidate_prices)

    results = simulate_candidates(
        population=population,
        candidate_prices=all_candidates,
        base_offer=base_offer,
        ml_probabilities=ml_probabilities,
        policy=policy,
        seed=seed,
    )

    # Find the candidate matching the recommended price
    selected_result = next((r for r in results if abs(r.price - base_offer.price) < 1e-4), None)
    if not selected_result:
        return {"available": False, "reason": "Recommended price not found in candidates."}

    best_result = next(r for r in results if r.rank_sim == 1)
    sim_rank = selected_result.rank_sim or 1
    ml_rank = selected_result.rank_ml or 1
    rank_drop = sim_rank - ml_rank

    metric_name = selected_result.ranking_metric_name
    best_metric = best_result.ranking_metric_value
    selected_metric = selected_result.ranking_metric_value

    # Safe calculation of relative metric gap
    if best_metric > 1e-9:
        metric_gap = max(0.0, (best_metric - selected_metric) / best_metric)
    elif abs(best_metric) <= 1e-9:
        metric_gap = 0.0 if selected_metric >= -1e-9 else 1.0
    else:
        # Both best and selected candidate have negative values
        metric_gap = max(0.0, (best_metric - selected_metric) / abs(best_metric))

    rank_drop_passed = (sim_rank <= tolerance_rank_drop + 1) and (rank_drop <= tolerance_rank_drop)
    gap_passed = (metric_gap <= tolerance_profit_gap)
    stable = rank_drop_passed and gap_passed

    return {
        "available": True,
        "passed": stable,
        "recommended_price": base_offer.price,
        "ranking_metric_name": metric_name,
        "sim_rank": sim_rank,
        "ml_rank": ml_rank,
        "rank_drop": rank_drop,
        "tolerance_rank_drop": tolerance_rank_drop,
        "rank_drop_passed": rank_drop_passed,
        "best_sim_price": best_result.price,
        "best_sim_metric": round(best_metric, 4),
        "selected_sim_metric": round(selected_metric, 4),
        "metric_gap": round(metric_gap, 4),
        "tolerance_profit_gap": tolerance_profit_gap,
        "gap_passed": gap_passed,
        "candidates": [
            {
                "price": r.price,
                "sim_conversion": round(r.sim_conversion, 4),
                "sim_expected_profit": round(r.sim_expected_profit, 2) if r.sim_expected_profit is not None else None,
                "sim_expected_revenue": round(r.sim_expected_revenue, 2) if r.sim_expected_revenue is not None else None,
                "ranking_metric_name": r.ranking_metric_name,
                "ranking_metric_value": round(r.ranking_metric_value, 2),
                "rank_sim": r.rank_sim,
                "ml_probability": round(r.ml_probability, 4) if r.ml_probability is not None else None,
                "rank_ml": r.rank_ml,
            }
            for r in results
        ],
    }


def validate_ml_discrepancy(
    ml_probability: float,
    sim_conversion: float,
    max_allowed_gap: float = 0.15,
) -> Dict[str, Any]:
    """
    Test E: ML disagreement and anomaly detection.

    Quantifies the discrepancy between ML probability and simulated purchase rate.
    """
    gap = abs(ml_probability - sim_conversion)
    exceeded = gap > max_allowed_gap
    severe = gap > (2.0 * max_allowed_gap)

    contributing_factors = []
    if ml_probability > sim_conversion + 0.10:
        contributing_factors.append(
            "ML model expects higher conversion than agent simulation. Potential customer budget constraints "
            "or price sensitivity may be underestimated by ML."
        )
    elif sim_conversion > ml_probability + 0.10:
        contributing_factors.append(
            "Agent simulation yields higher conversion than ML model. Perceived value surplus or urgency "
            "may be cushioning customer willingness to pay."
        )

    return {
        "ml_probability": round(ml_probability, 4),
        "simulated_conversion": round(sim_conversion, 4),
        "gap": round(gap, 4),
        "gap_percentage_points": round(gap * 100, 2),
        "max_allowed_gap": max_allowed_gap,
        "gap_exceeded": exceeded,
        "severe_discrepancy": severe,
        "contributing_factors": contributing_factors,
    }


def evaluate_decision_gate(
    offer: Offer,
    ml_probability: float,
    candidate_prices: Optional[List[float]] = None,
    ml_probabilities: Optional[Dict[float, float]] = None,
    config: Optional[ValidationConfig] = None,
    population: Optional[List[CustomerAgent]] = None,
    policy: Optional[BaseDecisionPolicy] = None,
) -> ValidationReport:
    """
    Unified validation decision gate.

    Synthesizes the 5 validation tests against configured thresholds and returns
    a PASS, FAIL, or REVIEW_REQUIRED status with comprehensive diagnostics.
    """
    if config is None:
        config = ValidationConfig()

    # Input validation
    if not (0.0 <= ml_probability <= 1.0):
        raise ValueError(f"ml_probability must be in [0.0, 1.0], got {ml_probability}")

    # Generate population if not provided
    if population is None:
        generator = PopulationGenerator()
        population = generator.generate(size=config.population_size, seed=config.seed)

    if policy is None:
        policy = StandardDecisionPolicy()

    # Baseline simulation
    baseline_metrics = simulate_offer(
        population=population,
        offer=offer,
        policy=policy,
        seed=config.seed,
        confidence_level=config.confidence_level,
    )

    # 1. Behavioural consistency
    consistency_res = validate_behavioural_consistency(population, offer, policy=policy, seed=config.seed)

    # 2. Population heterogeneity
    heterogeneity_res = validate_population_heterogeneity(
        population,
        offer,
        policy=policy,
        min_variance=config.min_heterogeneity_variance,
        seed=config.seed,
    )

    # 3. Scenario robustness
    scenario_res = validate_scenario_robustness(
        population,
        offer,
        scenarios=config.scenarios,
        min_conversion_threshold=config.minimum_simulated_conversion,
        seed=config.seed,
    )

    # 4. Ranking stability
    ranking_res = validate_ranking_stability(
        population=population,
        candidate_prices=candidate_prices,
        base_offer=offer,
        ml_probabilities=ml_probabilities,
        tolerance_rank_drop=config.ranking_tolerance_rank_drop,
        tolerance_profit_gap=config.ranking_tolerance_profit_gap,
        policy=policy,
        seed=config.seed,
    )

    # 5. ML Discrepancy
    discrepancy_res = validate_ml_discrepancy(
        ml_probability=ml_probability,
        sim_conversion=baseline_metrics.conversion_rate,
        max_allowed_gap=config.max_allowed_probability_gap,
    )
    discrepancy_res["minimum_simulated_conversion"] = config.minimum_simulated_conversion

    # Gate decision logic
    warnings: List[str] = []
    recommendations: List[str] = []
    status = ValidationStatus.PASS
    reason_parts = []

    # FAIL checks
    if baseline_metrics.ci_upper < config.minimum_simulated_conversion:
        status = ValidationStatus.FAIL
        reason_parts.append(
            f"Simulated conversion ({baseline_metrics.conversion_rate:.2%}) is definitively below the minimum "
            f"threshold ({config.minimum_simulated_conversion:.2%})."
        )
        recommendations.append("Decrease candidate price to restore economic feasibility.")

    if not consistency_res["passed"]:
        status = ValidationStatus.FAIL
        reason_parts.append("Failed behavioural consistency (price monotonicity / demand response violated).")
        recommendations.append("Investigate market signals or calibrate behavioral policy parameters.")

    if discrepancy_res["severe_discrepancy"]:
        status = ValidationStatus.FAIL
        reason_parts.append(
            f"Severe discrepancy between ML probability ({ml_probability:.2%}) and simulation ({baseline_metrics.conversion_rate:.2%}): "
            f"gap {discrepancy_res['gap']:.2%} exceeds 2x allowed tolerance."
        )
        recommendations.append("Halt automated price application; conduct model review.")

    # REVIEW_REQUIRED checks (only if not already FAIL)
    if status != ValidationStatus.FAIL:
        if discrepancy_res["gap_exceeded"]:
            status = ValidationStatus.REVIEW_REQUIRED
            reason_parts.append(
                f"Probability gap ({discrepancy_res['gap']:.2%}) exceeds allowed tolerance ({config.max_allowed_probability_gap:.2%})."
            )
            warnings.append(f"ML and simulation estimate divergence of {discrepancy_res['gap_percentage_points']} pp.")
            recommendations.append("Request pricing analyst review to reconcile behavioral assumptions with ML signals.")

        if baseline_metrics.conversion_rate >= config.minimum_simulated_conversion and baseline_metrics.ci_lower < config.minimum_simulated_conversion:
            status = ValidationStatus.REVIEW_REQUIRED
            reason_parts.append(
                f"Simulated conversion is borderline: lower 95% CI ({baseline_metrics.ci_lower:.2%}) falls below "
                f"minimum threshold ({config.minimum_simulated_conversion:.2%})."
            )
            warnings.append("Borderline conversion with uncertainty overlapping safety threshold.")

        if not scenario_res["passed"] or len(scenario_res["vulnerabilities"]) > 0:
            if status != ValidationStatus.FAIL:
                status = ValidationStatus.REVIEW_REQUIRED
            for vuln in scenario_res["vulnerabilities"]:
                warnings.append(f"Scenario vulnerability: {vuln}")
            reason_parts.append("Recommended price demonstrates fragility under stress scenarios.")
            recommendations.append("Evaluate more defensive pricing alternatives robust against market shifts.")

        if ranking_res.get("available") and not ranking_res.get("passed"):
            status = ValidationStatus.REVIEW_REQUIRED
            fail_reasons = []
            if not ranking_res.get("rank_drop_passed", True):
                fail_reasons.append(
                    f"rank drop of {ranking_res['rank_drop']} exceeded tolerance ({ranking_res['tolerance_rank_drop']})"
                )
            if not ranking_res.get("gap_passed", True):
                fail_reasons.append(
                    f"{ranking_res.get('ranking_metric_name', 'metric')} gap of {ranking_res['metric_gap']:.2%} "
                    f"exceeded tolerance ({ranking_res['tolerance_profit_gap']:.2%})"
                )
            fail_desc = "; ".join(fail_reasons)
            reason_parts.append(
                f"Ranking instability: ML-selected price dropped to rank #{ranking_res['sim_rank']} "
                f"in customer simulation ({fail_desc}; best simulated candidate: ${ranking_res['best_sim_price']:.2f})."
            )
            warnings.append(f"Simulation ranking flags instability: {fail_desc}.")
            recommendations.append(
                f"Consider evaluating candidate price ${ranking_res['best_sim_price']:.2f} for improved simulated {ranking_res.get('ranking_metric_name', 'outcome')}."
            )

        if not heterogeneity_res["passed"]:
            warnings.append(heterogeneity_res["warning"])

    if status == ValidationStatus.PASS:
        reason = "All configured validation checks passed successfully."
        recommendations.append("Price recommendation validated for downstream deployment.")
    else:
        reason = " | ".join(reason_parts) if reason_parts else "Validation requires attention."

    return ValidationReport(
        status=status,
        offer=offer,
        ml_probability=ml_probability,
        simulated_conversion=baseline_metrics.conversion_rate,
        probability_gap=discrepancy_res["gap"],
        ci_lower=baseline_metrics.ci_lower,
        ci_upper=baseline_metrics.ci_upper,
        population_size=baseline_metrics.population_size,
        n_purchases=baseline_metrics.n_purchases,
        consistency_results=consistency_res,
        heterogeneity_results=heterogeneity_res,
        scenario_results=scenario_res,
        ranking_results=ranking_res,
        discrepancy_results=discrepancy_res,
        warnings=warnings,
        recommendations=recommendations,
        reason=reason,
        minimum_simulated_conversion=config.minimum_simulated_conversion,
    )
