"""
run_demo.py -- Runnable demonstration of Monetize360 Customer-Agent Validation Module.

Demonstrates:
  1. Input offer specification and simulation configuration.
  2. Diverse customer agent population generation.
  3. Monte Carlo decision evaluation with individual choices (BUY/DEFER/REJECT).
  4. Aggregate conversion and Wilson confidence interval estimation.
  5. The 5 validation capability checks:
     A. Behavioural consistency
     B. Population heterogeneity
     C. Scenario robustness
     D. Ranking stability across candidates
     E. ML disagreement analysis
  6. Final decision gate (PASS / FAIL / REVIEW_REQUIRED) and structured feedback loop.
  7. Single-price mode and alternative scenario tests.

Usage:
  python customer_agent_sim/examples/run_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from customer_agent_sim import (
    Offer,
    PopulationGenerator,
    StandardDecisionPolicy,
    ValidationConfig,
    ValidationStatus,
    bounded_validation_pipeline,
    extract_feedback_diagnostics,
    simulate_offer,
    simulate_population_decisions,
    validate_offer,
)


def print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def run_demo() -> None:
    print_banner("Monetize360 Customer-Agent Validation Module -- Live Demonstration")
    print("Disclaimer: All simulated customer agents, preferences, and probabilities")
    print("are synthetic programmatic models for offline algorithmic verification.\n")

    # ---------------------------------------------------------
    # 1. Offer & Configuration Setup
    # ---------------------------------------------------------
    print(">>> 1. Setting up Offer & Validation Configuration...")
    offer = Offer(
        price=105.00,             # ML Optimizer selected price
        base_price=100.00,        # Catalog reference price
        perceived_value=120.00,   # Customer perceived utility
        competitor_price=108.00,  # Competitor price
        urgency=0.35,             # Moderate urgency
        scarcity=0.25,            # Moderate supply pressure
        unit_cost=60.00,          # Unit cost for profit calculations
        context={"channel": "direct", "region": "metro_west"},
    )
    ml_probability = 0.65       # ML model predicted purchase probability

    candidate_prices = [85.00, 95.00, 100.00, 105.00, 115.00, 125.00]
    ml_probabilities = {
        85.00: 0.82,
        95.00: 0.74,
        100.00: 0.69,
        105.00: 0.65,
        115.00: 0.52,
        125.00: 0.38,
    }

    config = ValidationConfig(
        population_size=1000,
        seed=42,
        minimum_simulated_conversion=0.08,
        max_allowed_probability_gap=0.15,
        ranking_tolerance_rank_drop=2,
    )
    print(f"  Candidate Price:       ${offer.price:.2f} (Base: ${offer.base_price:.2f})")
    print(f"  Perceived Value:       ${offer.perceived_value:.2f}")
    print(f"  Competitor Price:      ${offer.competitor_price:.2f}")
    print(f"  ML Predicted P(buy):   {ml_probability:.2%}")
    print(f"  Candidate Grid:        {candidate_prices}")
    print(f"  Population Size:       {config.population_size}")

    # ---------------------------------------------------------
    # 2. Population Generation
    # ---------------------------------------------------------
    print("\n>>> 2. Generating Heterogeneous Customer Agent Population...")
    gen = PopulationGenerator()
    population = gen.generate(size=config.population_size, seed=config.seed)
    sample_agents = population[:3]
    for a in sample_agents:
        print(f"  [{a.agent_id}] Traits: PriceSens={a.traits.price_sensitivity:.2f}, "
              f"ValOrient={a.traits.value_orientation:.2f}, RiskAvers={a.traits.risk_aversion:.2f}, "
              f"Patience={a.traits.patience:.2f} | State: BudgetRatio={a.state.budget_ratio:.2f}")

    # ---------------------------------------------------------
    # 3. Individual Agent Decision Simulation
    # ---------------------------------------------------------
    print("\n>>> 3. Sampling Agent Decisions (BUY / DEFER / REJECT)...")
    decisions = simulate_population_decisions(population[:5], offer, seed=config.seed)
    for d in decisions:
        print(f"  [{d.agent_id}] -> Outcome: {d.outcome.value:<6} "
              f"(P_buy={d.p_buy:.3f}, P_defer={d.p_defer:.3f}, P_reject={d.p_reject:.3f})")

    # ---------------------------------------------------------
    # 4. Aggregate Metrics & Uncertainty
    # ---------------------------------------------------------
    print("\n>>> 4. Computing Aggregate Metrics & Monte Carlo Uncertainty...")
    metrics = simulate_offer(population, offer, seed=config.seed)
    print(f"  Simulated Purchases:  {metrics.n_purchases} / {metrics.population_size}")
    print(f"  Conversion Rate:      {metrics.conversion_rate:.2%}")
    print(f"  Wilson 95% CI:        [{metrics.ci_lower:.2%} - {metrics.ci_upper:.2%}]")
    print(f"  Deferral Rate:        {metrics.deferral_rate:.2%}")
    print(f"  Rejection Rate:       {metrics.rejection_rate:.2%}")
    print(f"  Expected Unit Profit: ${metrics.expected_profit:.2f}")

    # ---------------------------------------------------------
    # 5. Full Validation Gate Execution
    # ---------------------------------------------------------
    print("\n>>> 5. Executing Validation Suite across all 5 checks...")
    report = validate_offer(
        offer=offer,
        ml_probability=ml_probability,
        candidate_prices=candidate_prices,
        ml_probabilities=ml_probabilities,
        config=config,
        population=population,
    )

    print(f"\nCheck A (Consistency):    {'PASSED' if report.consistency_results['passed'] else 'FAILED'}")
    print(f"  Price Ladder Monotonicity: {report.consistency_results['checks']['price_monotonicity']['passed']}")
    print(f"  Value Sensitivity:         {report.consistency_results['checks']['perceived_value_sensitivity']['passed']}")

    print(f"\nCheck B (Heterogeneity):  {'PASSED' if report.heterogeneity_results['passed'] else 'FAILED'}")
    print(f"  Choice Prob Variance:      {report.heterogeneity_results['probability_variance']}")
    ps_sub = report.heterogeneity_results["subgroups"]["price_sensitivity"]
    print(f"  Low Price Sens Conv:       {ps_sub['low_group_conversion']:.2%}")
    print(f"  High Price Sens Conv:      {ps_sub['high_group_conversion']:.2%}")

    print(f"\nCheck C (Scenarios):      {'PASSED' if report.scenario_results['passed'] else 'FAILED'}")
    for sc_name, sc_data in report.scenario_results["scenarios"].items():
        if sc_name != "baseline":
            print(f"  - {sc_name:<22}: Conv={sc_data['conversion_rate']:.2%} (Delta: {sc_data['delta_from_baseline']:+.2%})")

    print(f"\nCheck D (Ranking):        {'PASSED' if report.ranking_results['passed'] else 'FAILED'}")
    print(f"  ML Rank: #{report.ranking_results['ml_rank']} | Sim Rank: #{report.ranking_results['sim_rank']} "
          f"(Best Sim Price: ${report.ranking_results['best_sim_price']:.2f})")
    print(f"  Metric: {report.ranking_results.get('ranking_metric_name', 'metric')} | "
          f"Relative Gap: {report.ranking_results.get('metric_gap', 0.0):.2%} "
          f"(Tolerance: {report.ranking_results.get('tolerance_profit_gap', 0.12):.2%})")

    print(f"\nCheck E (ML Discrepancy): {'PASSED' if not report.discrepancy_results['gap_exceeded'] else 'EXCEEDED'}")
    print(f"  ML Probability:   {report.discrepancy_results['ml_probability']:.2%}")
    print(f"  Sim Conversion:   {report.discrepancy_results['simulated_conversion']:.2%}")
    print(f"  Probability Gap:  {report.discrepancy_results['gap_percentage_points']} pp "
          f"(Tolerance: {report.discrepancy_results['max_allowed_gap'] * 100:.1f} pp)")

    # ---------------------------------------------------------
    # 6. Final Gate Decision & Structured Feedback
    # ---------------------------------------------------------
    print_banner("Validation Gate Final Decision")
    print(report.summary_text())

    feedback = extract_feedback_diagnostics(report)
    print("\nStructured Feedback Diagnostic for Monetize360 Pipeline:")
    print(json.dumps(feedback.to_dict(), indent=2))

    # ---------------------------------------------------------
    # 7. Edge Case: Single-Price Validation & Inapplicable Competitor Scenario
    # ---------------------------------------------------------
    print_banner("Diagnostic Case 2: Single-Price Mode & Inapplicable Competitor Scenario")
    print("Evaluating candidate price without competitor benchmark or candidate grid...")
    no_comp_offer = Offer(
        price=105.00,
        base_price=100.00,
        perceived_value=120.00,
        competitor_price=None,  # No competitor benchmark available
        urgency=0.35,
        scarcity=0.25,
        unit_cost=60.00,
    )
    single_report = validate_offer(
        offer=no_comp_offer,
        ml_probability=ml_probability,
        candidate_prices=None,  # No candidates
        config=config,
        population=population,
    )
    print(f"Status: {single_report.status.value}")
    print(f"Ranking Stability Availability: {single_report.ranking_results.get('available')}")
    print(f"Ranking Reason: {single_report.ranking_results.get('reason')}")
    comp_scenario_info = single_report.scenario_results["scenarios"].get("competitor_undercut", {})
    print(f"Competitor Scenario Applicable: {comp_scenario_info.get('applicable')} "
          f"({comp_scenario_info.get('reason', 'N/A')})")

    # ---------------------------------------------------------
    # 8. Edge Case: Overpriced Recommendation triggering FAIL / REVIEW_REQUIRED
    # ---------------------------------------------------------
    print_banner("Diagnostic Case 3: Overpriced Candidate triggering REVIEW_REQUIRED / FAIL")
    overpriced_offer = Offer(
        price=185.00,
        base_price=100.00,
        perceived_value=110.00,
        competitor_price=95.00,
        urgency=0.10,
        scarcity=0.05,
        unit_cost=60.00,
    )
    stress_report = validate_offer(
        offer=overpriced_offer,
        ml_probability=0.75,  # Wildly over-optimistic ML prediction
        candidate_prices=[90.0, 100.0, 120.0, 150.0, 185.0],
        config=config,
        population=population,
    )
    print(stress_report.summary_text())
    stress_feedback = extract_feedback_diagnostics(stress_report)
    print(f"Suggested Action: {stress_feedback.suggested_action}")
    print(f"Alternative Candidate: ${stress_feedback.alternative_candidate_price}")

    print_banner("Demo Completed Successfully")


if __name__ == "__main__":
    run_demo()
