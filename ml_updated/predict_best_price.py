"""
predict_best_price.py — Domain-Agnostic Optimal Price Selector

FLOW:
=====
  INPUT  → base_price, unit_cost, customer persona, market context, 4-5 candidate prices
  ML     → Predict P(buy) for each candidate, holding customer & market features constant
  PROFIT → Calculate expected_profit = P(buy) × (price - unit_cost) for each candidate
  GUARD  → Apply business constraints (loss prevention, margin floor, price ceiling)
  OUTPUT → Recommended price with highest expected profit among eligible candidates

This works for ANY domain because:
  - All dollar amounts are converted to ratios before entering the model
  - Customer is described by behavioral scores, not demographic categories
  - Market context is described by universal signals, not industry-specific metrics
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from features import ALL_FEATURES, build_candidate_features


ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def load_model(artifacts_dir: Path = ARTIFACTS_DIR):
    """Load trained GBT model."""
    model_path = artifacts_dir / "model_gbt.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run: python generate_dataset.py && python train.py"
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def evaluate_candidates(
    candidate_prices: list[float],
    base_price: float,
    unit_cost: float,
    competitor_price: float,
    customer_sensitivity: float = 0.5,
    customer_loyalty_score: float = 0.3,
    urgency_score: float = 0.3,
    scarcity_score: float = 0.3,
    demand_pressure: float = 1.0,
    competitor_density: float = 0.5,
    is_promotion_active: int = 0,
    time_decay_factor: float = 0.5,
    perceived_value_gap: float | None = None,
    # Business constraints
    min_margin_pct: float = 0.05,
    max_price_change_pct: float = 0.50,
    min_conversion_rate: float = 0.10,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> pd.DataFrame:
    """
    Evaluate a list of candidate prices through the ML model + profit optimizer.

    For each candidate price:
      1. Build domain-agnostic feature row
      2. Predict P(buy) using the trained GBT
      3. Calculate expected_profit = P(buy) × (price - unit_cost)
      4. Apply business guardrails
      5. Select the winner with highest expected profit

    Returns a DataFrame with one row per candidate.
    """
    model = load_model(artifacts_dir)

    # Sort candidates for consistent ranking
    sorted_prices = sorted(candidate_prices)
    n = len(sorted_prices)

    # Compute business constraint thresholds
    min_safe_price = round(unit_cost * (1.0 + min_margin_pct), 2)
    max_ceiling = round(base_price * (1.0 + max_price_change_pct), 2)

    # Build feature rows for all candidates at once (batch prediction)
    feature_rows = []
    for i, price in enumerate(sorted_prices):
        row = build_candidate_features(
            candidate_price=price,
            base_price=base_price,
            unit_cost=unit_cost,
            competitor_price=competitor_price,
            candidate_rank_normalized=i / max(n - 1, 1),
            customer_sensitivity=customer_sensitivity,
            customer_loyalty_score=customer_loyalty_score,
            urgency_score=urgency_score,
            scarcity_score=scarcity_score,
            demand_pressure=demand_pressure,
            competitor_density=competitor_density,
            is_promotion_active=is_promotion_active,
            time_decay_factor=time_decay_factor,
            perceived_value_gap=perceived_value_gap,
        )
        feature_rows.append(row)

    features_df = pd.DataFrame(feature_rows)

    # Ensure column order matches training
    features_df = features_df[ALL_FEATURES]

    # Predict P(buy) for all candidates
    probs = model.predict_proba(features_df.values)[:, 1]

    # Build results
    results = []
    for i, (price, prob) in enumerate(zip(sorted_prices, probs)):
        unit_margin = price - unit_cost
        margin_pct = round((unit_margin / price) * 100 if price > 0 else 0, 1)
        markup_pct = round((unit_margin / unit_cost) * 100 if unit_cost > 0 else 0, 1)
        expected_profit = unit_margin * prob
        expected_revenue = price * prob

        # Business constraint checks
        is_loss = price < unit_cost
        is_below_margin = price < min_safe_price
        is_above_ceiling = price > max_ceiling
        is_low_conversion = prob < min_conversion_rate

        status = "VIABLE"
        notes = []

        if is_loss:
            status = "DISQUALIFIED_LOSS"
            notes.append(f"Price below unit cost (${unit_cost:.2f}) -- business takes loss!")
        elif is_below_margin:
            status = "BELOW_MARGIN"
            notes.append(f"Below min margin floor (${min_safe_price:.2f})")
        elif is_above_ceiling:
            status = "EXCEEDS_CEILING"
            notes.append(f"Exceeds price ceiling (${max_ceiling:.2f})")
        elif is_low_conversion:
            status = "LOW_CONVERSION"
            notes.append(f"Conversion {prob:.1%} below min threshold ({min_conversion_rate:.0%})")
        else:
            notes.append("Within all business guardrails")

        results.append({
            "candidate_price": price,
            "unit_cost": unit_cost,
            "unit_margin": round(unit_margin, 2),
            "margin_pct": margin_pct,
            "markup_pct": markup_pct,
            "p_buy": round(float(prob), 4),
            "expected_profit": round(expected_profit, 2),
            "expected_revenue": round(expected_revenue, 2),
            "status": status,
            "notes": "; ".join(notes),
        })

    df = pd.DataFrame(results)

    # Select winner: highest expected_profit among VIABLE candidates
    viable = df[df["status"] == "VIABLE"]
    df["verdict"] = ""

    if not viable.empty:
        best_idx = viable["expected_profit"].idxmax()
        for i, row in df.iterrows():
            if row["status"] == "DISQUALIFIED_LOSS":
                df.at[i, "verdict"] = "REJECTED -- Sells at Loss"
            elif row["status"] == "BELOW_MARGIN":
                df.at[i, "verdict"] = "REJECTED -- Unsafe Margin"
            elif row["status"] == "EXCEEDS_CEILING":
                df.at[i, "verdict"] = "REJECTED -- Ceiling Exceeded"
            elif row["status"] == "LOW_CONVERSION":
                df.at[i, "verdict"] = "REJECTED -- Low Conversion"
            elif i == best_idx:
                df.at[i, "verdict"] = "* WINNER -- Optimal Expected Profit"
            else:
                df.at[i, "verdict"] = "VIABLE -- Lower Expected Profit"
    else:
        df["verdict"] = "NO VIABLE PRICE"

    return df


def print_report(
    df: pd.DataFrame,
    scenario_name: str,
    base_price: float,
    unit_cost: float,
    competitor_price: float,
    customer_sensitivity: float,
    customer_loyalty_score: float,
) -> None:
    """Pretty-print the optimization report."""
    print()
    print("=" * 90)
    print(f"  MONETIZE360 -- DOMAIN-AGNOSTIC PRICE OPTIMIZATION REPORT")
    print(f"  Scenario: {scenario_name}")
    print("=" * 90)
    print(f"  Base Price: ${base_price:.2f} | Unit Cost: ${unit_cost:.2f} | "
          f"Competitor: ${competitor_price:.2f}")
    print(f"  Customer Sensitivity: {customer_sensitivity:.2f} | "
          f"Loyalty: {customer_loyalty_score:.2f}")
    print("-" * 90)

    # Table header
    headers = ["Price", "Margin", "Margin%", "Markup%", "P(Buy)", "E[Profit]", "E[Rev]", "Verdict"]
    widths = [10, 10, 8, 8, 8, 11, 11, 33]
    header_line = " | ".join(h.center(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * 105)

    for _, r in df.iterrows():
        is_winner = "WINNER" in r["verdict"]
        marker = "-> " if is_winner else "   "
        row = [
            f"${r['candidate_price']:.2f}",
            f"${r['unit_margin']:+.2f}",
            f"{r['margin_pct']:.1f}%",
            f"{r['markup_pct']:.1f}%",
            f"{r['p_buy']:.1%}",
            f"${r['expected_profit']:.2f}",
            f"${r['expected_revenue']:.2f}",
            r["verdict"],
        ]
        line = " | ".join(val.ljust(w) for val, w in zip(row, widths))
        print(f"{marker}{line}")

    print("-" * 105)

    winner = df[df["verdict"].str.contains("WINNER")]
    if not winner.empty:
        w = winner.iloc[0]
        print(f"\n  * RECOMMENDED PRICE: ${w['candidate_price']:.2f}")
        print(f"    Expected Profit:  ${w['expected_profit']:.2f} per quote")
        print(f"    Conversion Rate:  {w['p_buy']:.1%}")
        print(f"    Unit Margin:      ${w['unit_margin']:.2f} (Margin: {w['margin_pct']:.1f}% of price | Markup: {w['markup_pct']:.1f}% over cost)")
        print(f"\n  RATIONALE:")
        print(f"    The model predicts that among all viable candidates,")
        print(f"    ${w['candidate_price']:.2f} maximizes E[profit] = P(buy) x (price - cost).")
        print(f"    Lower prices yield higher conversion but leave profit on the table.")
        print(f"    Higher prices boost margin but suppress demand beyond the optimum.")
    else:
        print("\n  [!] WARNING: No viable candidate price found!")
        print("    All candidates either sell at a loss or fail business constraints.")

    print("\n" + "=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Predefined scenarios for demonstration across multiple domains
# ---------------------------------------------------------------------------

DEMO_SCENARIOS = {
    "ecommerce_headphones": {
        "scenario_name": "E-Commerce: Wireless Headphones",
        "base_price": 79.99,
        "unit_cost": 35.00,
        "competitor_price": 74.99,
        "candidate_prices": [59.99, 69.99, 79.99, 89.99, 99.99],
        "customer_sensitivity": 0.65,
        "customer_loyalty_score": 0.30,
        "urgency_score": 0.20,
        "scarcity_score": 0.25,
        "demand_pressure": 1.1,
        "competitor_density": 0.70,
    },
    "hotel_room": {
        "scenario_name": "Hospitality: Deluxe Hotel Room (per night)",
        "base_price": 220.00,
        "unit_cost": 85.00,
        "competitor_price": 199.00,
        "candidate_prices": [175.00, 199.00, 220.00, 249.00, 279.00],
        "customer_sensitivity": 0.45,
        "customer_loyalty_score": 0.55,
        "urgency_score": 0.60,
        "scarcity_score": 0.70,
        "demand_pressure": 1.4,
        "competitor_density": 0.50,
    },
    "car_rental": {
        "scenario_name": "Car Rental: Midsize SUV (per day)",
        "base_price": 65.00,
        "unit_cost": 28.00,
        "competitor_price": 59.99,
        "candidate_prices": [49.00, 55.00, 65.00, 75.00, 85.00],
        "customer_sensitivity": 0.50,
        "customer_loyalty_score": 0.40,
        "urgency_score": 0.50,
        "scarcity_score": 0.45,
        "demand_pressure": 1.2,
        "competitor_density": 0.55,
    },
    "saas_subscription": {
        "scenario_name": "SaaS: Monthly Subscription Plan",
        "base_price": 49.00,
        "unit_cost": 8.00,
        "competitor_price": 39.00,
        "candidate_prices": [29.00, 39.00, 49.00, 59.00, 69.00],
        "customer_sensitivity": 0.55,
        "customer_loyalty_score": 0.60,
        "urgency_score": 0.15,
        "scarcity_score": 0.10,
        "demand_pressure": 0.9,
        "competitor_density": 0.80,
    },
    "healthcare_device": {
        "scenario_name": "Healthcare: Blood Pressure Monitor",
        "base_price": 145.00,
        "unit_cost": 62.00,
        "competitor_price": 139.00,
        "candidate_prices": [115.00, 130.00, 145.00, 160.00, 175.00],
        "customer_sensitivity": 0.30,
        "customer_loyalty_score": 0.50,
        "urgency_score": 0.70,
        "scarcity_score": 0.35,
        "demand_pressure": 1.0,
        "competitor_density": 0.40,
    },
    "luxury_watch": {
        "scenario_name": "Luxury: Premium Wristwatch",
        "base_price": 2500.00,
        "unit_cost": 800.00,
        "competitor_price": 2200.00,
        "candidate_prices": [2000.00, 2250.00, 2500.00, 2750.00, 3000.00],
        "customer_sensitivity": 0.15,
        "customer_loyalty_score": 0.70,
        "urgency_score": 0.10,
        "scarcity_score": 0.55,
        "demand_pressure": 0.8,
        "competitor_density": 0.30,
    },
}


def run_demo(scenarios: dict | None = None) -> None:
    """Run all demo scenarios and print reports."""
    scenarios = scenarios or DEMO_SCENARIOS

    for key, params in scenarios.items():
        scenario_name = params.pop("scenario_name", key)
        candidate_prices = params.pop("candidate_prices")

        df = evaluate_candidates(
            candidate_prices=candidate_prices,
            **params,
        )

        print_report(
            df=df,
            scenario_name=scenario_name,
            base_price=params["base_price"],
            unit_cost=params["unit_cost"],
            competitor_price=params["competitor_price"],
            customer_sensitivity=params["customer_sensitivity"],
            customer_loyalty_score=params["customer_loyalty_score"],
        )

        # Restore popped keys for re-run
        params["scenario_name"] = scenario_name
        params["candidate_prices"] = candidate_prices


def main():
    parser = argparse.ArgumentParser(
        description="Domain-Agnostic Price Optimizer -- Predict optimal price using ML"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run all demo scenarios (e-commerce, hotel, car rental, SaaS, healthcare, luxury)"
    )
    parser.add_argument("--prices", type=float, nargs="+", help="Candidate prices")
    parser.add_argument("--base-price", type=float, default=100.0, help="Base/reference price")
    parser.add_argument("--unit-cost", type=float, default=40.0, help="Unit cost")
    parser.add_argument("--competitor-price", type=float, default=95.0, help="Competitor price")
    parser.add_argument("--sensitivity", type=float, default=0.5, help="Customer sensitivity [0-1]")
    parser.add_argument("--loyalty", type=float, default=0.3, help="Customer loyalty score [0-1]")
    parser.add_argument("--urgency", type=float, default=0.3, help="Urgency score [0-1]")
    parser.add_argument("--scarcity", type=float, default=0.3, help="Scarcity score [0-1]")
    parser.add_argument("--demand", type=float, default=1.0, help="Demand pressure [0-2]")

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.prices:
        df = evaluate_candidates(
            candidate_prices=args.prices,
            base_price=args.base_price,
            unit_cost=args.unit_cost,
            competitor_price=args.competitor_price,
            customer_sensitivity=args.sensitivity,
            customer_loyalty_score=args.loyalty,
            urgency_score=args.urgency,
            scarcity_score=args.scarcity,
            demand_pressure=args.demand,
        )
        print_report(
            df=df,
            scenario_name="Custom Scenario",
            base_price=args.base_price,
            unit_cost=args.unit_cost,
            competitor_price=args.competitor_price,
            customer_sensitivity=args.sensitivity,
            customer_loyalty_score=args.loyalty,
        )
    else:
        print("Usage:")
        print("  python predict_best_price.py --demo")
        print("  python predict_best_price.py --prices 80 90 100 110 120 --base-price 100 --unit-cost 40")
        parser.print_help()


if __name__ == "__main__":
    main()
