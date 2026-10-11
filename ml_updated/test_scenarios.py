"""
test_scenarios.py - Cross-Domain and Abnormal / Edge-Case Evaluation Suite

Tests:
  Part 1: New Diverse Domains
    - Cloud / AI GPU Compute
    - Airline Dynamic Seat Pricing
    - EdTech Subscription
    - Commercial Real Estate (Co-working Desk)
    - B2B Freight Logistics

  Part 2: Abnormal / Stress / Edge-Case Scenarios
    - Case A: Predatory / Below-Cost Pricing Trap
    - Case B: Extreme Price Gouging / Supply Shock (5x-7x base)
    - Case C: Inverted Market (Competitor Dumping at 75% discount)
    - Case D: Hyper-Sensitive Bargain Hunter (Sensitivity=1.0, cliff test)
    - Case E: The Monopoly Whale (Sensitivity=0.0, Loyalty=1.0)
    - Case F: Total Guardrail Disqualification (Zero viable candidates)
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from predict_best_price import evaluate_candidates, print_report

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


# ===========================================================================
# 1. NEW DIVERSE DOMAIN SCENARIOS
# ===========================================================================

NEW_DOMAIN_SCENARIOS = {
    "cloud_gpu_compute": {
        "scenario_name": "Cloud & AI Infrastructure: High-End GPU Cluster (per hr)",
        "base_price": 3.50,
        "unit_cost": 1.20,
        "competitor_price": 3.20,
        "candidate_prices": [2.50, 3.00, 3.50, 4.00, 4.50],
        "customer_sensitivity": 0.35,
        "customer_loyalty_score": 0.55,
        "urgency_score": 0.85,
        "scarcity_score": 0.80,
        "demand_pressure": 1.5,
        "competitor_density": 0.60,
    },
    "airline_business_flight": {
        "scenario_name": "Airlines: Last-Minute Business Class Seat",
        "base_price": 450.00,
        "unit_cost": 180.00,
        "competitor_price": 430.00,
        "candidate_prices": [350.00, 420.00, 480.00, 550.00, 620.00],
        "customer_sensitivity": 0.20,
        "customer_loyalty_score": 0.70,
        "urgency_score": 0.95,
        "scarcity_score": 0.90,
        "demand_pressure": 1.6,
        "competitor_density": 0.45,
        "time_decay_factor": 0.90,
    },
    "edtech_subscription": {
        "scenario_name": "EdTech: Professional Upskilling Monthly Subscription",
        "base_price": 39.99,
        "unit_cost": 6.50,
        "competitor_price": 29.99,
        "candidate_prices": [19.99, 29.99, 39.99, 49.99, 59.99],
        "customer_sensitivity": 0.78,
        "customer_loyalty_score": 0.25,
        "urgency_score": 0.20,
        "scarcity_score": 0.15,
        "demand_pressure": 0.9,
        "competitor_density": 0.85,
    },
    "commercial_coworking": {
        "scenario_name": "Commercial Real Estate: Dedicated Coworking Desk (per mo)",
        "base_price": 450.00,
        "unit_cost": 190.00,
        "competitor_price": 475.00,
        "candidate_prices": [380.00, 420.00, 450.00, 490.00, 540.00],
        "customer_sensitivity": 0.45,
        "customer_loyalty_score": 0.50,
        "urgency_score": 0.40,
        "scarcity_score": 0.60,
        "demand_pressure": 1.1,
        "competitor_density": 0.50,
    },
    "b2b_freight_logistics": {
        "scenario_name": "B2B Logistics: Cross-Border Freight Forwarding (per pallet)",
        "base_price": 850.00,
        "unit_cost": 520.00,
        "competitor_price": 820.00,
        "candidate_prices": [700.00, 780.00, 850.00, 920.00, 1000.00],
        "customer_sensitivity": 0.60,
        "customer_loyalty_score": 0.40,
        "urgency_score": 0.65,
        "scarcity_score": 0.50,
        "demand_pressure": 1.2,
        "competitor_density": 0.70,
    },
}


# ===========================================================================
# 2. ABNORMAL & STRESS-TEST SCENARIOS
# ===========================================================================

ABNORMAL_SCENARIOS = {
    "case_a_below_cost_trap": {
        "scenario_name": "ABNORMAL A: Below-Cost Pricing Trap (Suicide Discounting)",
        "description": "Prices are set below unit cost ($65.00). Customers love it, but each purchase loses money.",
        "base_price": 100.00,
        "unit_cost": 65.00,
        "competitor_price": 95.00,
        "candidate_prices": [35.00, 48.00, 58.00, 68.00, 82.00],
        "customer_sensitivity": 0.70,
        "customer_loyalty_score": 0.30,
        "urgency_score": 0.30,
        "scarcity_score": 0.20,
        "demand_pressure": 1.0,
        "competitor_density": 0.60,
    },
    "case_b_extreme_price_gouging": {
        "scenario_name": "ABNORMAL B: Extreme Gouging / Crisis Shortage (Up to 7x Base)",
        "description": "Extreme supply drought with massive candidate markups. Tests extrapolation and ceiling limits.",
        "base_price": 50.00,
        "unit_cost": 15.00,
        "competitor_price": 60.00,
        "candidate_prices": [60.00, 120.00, 180.00, 260.00, 350.00],
        "customer_sensitivity": 0.50,
        "customer_loyalty_score": 0.40,
        "urgency_score": 0.98,
        "scarcity_score": 0.95,
        "demand_pressure": 1.9,
        "competitor_density": 0.20,
    },
    "case_c_competitor_price_dumping": {
        "scenario_name": "ABNORMAL C: Inverted Market (Competitor Dumping at 75% Discount)",
        "description": "Competitor enters with predatory $25 pricing while our base is $100. Tests competitor anchoring drop.",
        "base_price": 100.00,
        "unit_cost": 40.00,
        "competitor_price": 25.00,
        "candidate_prices": [60.00, 75.00, 90.00, 100.00, 115.00],
        "customer_sensitivity": 0.75,
        "customer_loyalty_score": 0.20,
        "urgency_score": 0.25,
        "scarcity_score": 0.15,
        "demand_pressure": 0.8,
        "competitor_density": 0.90,
    },
    "case_d_hyper_sensitive_cliff": {
        "scenario_name": "ABNORMAL D: Hyper-Sensitive Bargain Hunter (Sensitivity = 1.0)",
        "description": "Customer will instantly abandon the cart if price exceeds competitor ($48.00).",
        "base_price": 50.00,
        "unit_cost": 20.00,
        "competitor_price": 48.00,
        "candidate_prices": [38.00, 44.00, 48.00, 52.00, 58.00],
        "customer_sensitivity": 1.00,
        "customer_loyalty_score": 0.05,
        "urgency_score": 0.10,
        "scarcity_score": 0.10,
        "demand_pressure": 0.7,
        "competitor_density": 0.95,
    },
    "case_e_monopoly_whale": {
        "scenario_name": "ABNORMAL E: Monopoly Whale (Sensitivity = 0.0, Loyalty = 1.0)",
        "description": "Zero price sensitivity, 100% brand loyalty, no competitors. Maximum pricing power.",
        "base_price": 1000.00,
        "unit_cost": 250.00,
        "competitor_price": 1200.00,
        "candidate_prices": [900.00, 1100.00, 1300.00, 1450.00, 1600.00],
        "customer_sensitivity": 0.00,
        "customer_loyalty_score": 1.00,
        "urgency_score": 0.80,
        "scarcity_score": 0.70,
        "demand_pressure": 1.4,
        "competitor_density": 0.05,
    },
    "case_f_total_guardrail_failure": {
        "scenario_name": "ABNORMAL F: Total Guardrail Disqualification (No Safe Price)",
        "description": "All candidate prices fail loss-prevention (all prices < $80.00 cost). Verifies refusal to sell.",
        "base_price": 100.00,
        "unit_cost": 80.00,
        "competitor_price": 85.00,
        "candidate_prices": [40.00, 50.00, 62.00, 72.00, 78.00],
        "customer_sensitivity": 0.50,
        "customer_loyalty_score": 0.50,
        "urgency_score": 0.50,
        "scarcity_score": 0.50,
        "demand_pressure": 1.0,
        "competitor_density": 0.50,
    },
}


def run_test_suite():
    print("\n" + "=" * 90)
    print("  MONETIZE360 ML TEST SUITE: NEW DOMAINS & ABNORMAL EDGE CASES")
    print("=" * 90)

    # -----------------------------------------------------------------------
    # Run Part 1: New Diverse Domains
    # -----------------------------------------------------------------------
    print("\n" + "#" * 90)
    print("  PART 1: NEW DOMAINS (Cloud GPU, Airlines, EdTech, Real Estate, B2B Freight)")
    print("#" * 90)

    for key, spec in NEW_DOMAIN_SCENARIOS.items():
        params = dict(spec)
        s_name = params.pop("scenario_name")
        c_prices = params.pop("candidate_prices")

        df = evaluate_candidates(candidate_prices=c_prices, artifacts_dir=ARTIFACTS_DIR, **params)
        print_report(
            df=df,
            scenario_name=s_name,
            base_price=params["base_price"],
            unit_cost=params["unit_cost"],
            competitor_price=params["competitor_price"],
            customer_sensitivity=params.get("customer_sensitivity", 0.5),
            customer_loyalty_score=params.get("customer_loyalty_score", 0.5),
        )

    # -----------------------------------------------------------------------
    # Run Part 2: Abnormal Test Cases
    # -----------------------------------------------------------------------
    print("\n" + "#" * 90)
    print("  PART 2: ABNORMAL & STRESS-TEST SCENARIOS")
    print("#" * 90)

    for key, spec in ABNORMAL_SCENARIOS.items():
        params = dict(spec)
        s_name = params.pop("scenario_name")
        desc = params.pop("description")
        c_prices = params.pop("candidate_prices")

        print("\n" + "~" * 90)
        print(f"  [TEST CASE CONTEXT]: {desc}")
        print("~" * 90)

        df = evaluate_candidates(candidate_prices=c_prices, artifacts_dir=ARTIFACTS_DIR, **params)
        print_report(
            df=df,
            scenario_name=s_name,
            base_price=params["base_price"],
            unit_cost=params["unit_cost"],
            competitor_price=params["competitor_price"],
            customer_sensitivity=params.get("customer_sensitivity", 0.5),
            customer_loyalty_score=params.get("customer_loyalty_score", 0.5),
        )


def print_summary_table():
    """Print a clean executive table summarizing all scenarios."""
    print("\n" + "=" * 105)
    print("  MONETIZE360 ML TEST SUITE -- EXECUTIVE SUMMARY TABLE")
    print("=" * 105)
    header = f"{'Scenario':<42s} | {'Base':>7s} | {'Cost':>7s} | {'Winner':>7s} | {'P(Buy)':>7s} | {'E[Profit]':>9s} | {'Verdict / Status':<16s}"
    print(header)
    print("-" * 105)

    all_groups = [
        ("NEW DOMAINS", NEW_DOMAIN_SCENARIOS),
        ("ABNORMAL & STRESS TESTS", ABNORMAL_SCENARIOS),
    ]

    for group_name, group_dict in all_groups:
        print(f"\n--- {group_name} ---")
        for k, spec in group_dict.items():
            params = dict(spec)
            s_name = params.pop("scenario_name")
            params.pop("description", None)
            c_prices = params.pop("candidate_prices")

            df = evaluate_candidates(candidate_prices=c_prices, artifacts_dir=ARTIFACTS_DIR, **params)
            winners = df[df["verdict"].str.contains("WINNER")]
            if not winners.empty:
                w = winners.iloc[0]
                row_str = (
                    f"{s_name[:42]:<42s} | "
                    f"${params['base_price']:>6.2f} | "
                    f"${params['unit_cost']:>6.2f} | "
                    f"${w['candidate_price']:>6.2f} | "
                    f"{w['p_buy']:>6.1%} | "
                    f"${w['expected_profit']:>8.2f} | "
                    f"{'OPTIMAL':<16s}"
                )
            else:
                row_str = (
                    f"{s_name[:42]:<42s} | "
                    f"${params['base_price']:>6.2f} | "
                    f"${params['unit_cost']:>6.2f} | "
                    f"{'NONE':>7s} | "
                    f"{'N/A':>7s} | "
                    f"{'$0.00':>9s} | "
                    f"{'PROTECTED (BLOCKED)':<16s}"
                )
            print(row_str)

    print("=" * 105 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        print_summary_table()
    else:
        run_test_suite()
        print_summary_table()

