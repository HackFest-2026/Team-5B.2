"""
benchmark_baselines.py - Benchmark the ML Profit Optimizer against 3 Traditional Baselines

Baselines:
  1. Base Price Policy (Status Quo):
     Always pick candidate closest to base_price among eligible candidates.
  2. Fixed Cost-Plus Markup Policy:
     Standard rule-of-thumb cost-plus pricing (e.g., Cost + 50% markup).
  3. Max Conversion Policy:
     Market-share maximizer: Pick the eligible candidate with highest P(buy).
  4. ML Profit Optimizer (Dynamic):
     Picks eligible candidate maximizing E[Profit] = P(buy) * (Price - Cost).

Outputs:
  - Table of expected profit across all scenarios for each strategy
  - Percentage Profit Lift (+Delta%) achieved by the ML Profit Optimizer
  - Machine-readable artifact: artifacts/benchmark_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from predict_best_price import evaluate_candidates
from test_scenarios import NEW_DOMAIN_SCENARIOS, ABNORMAL_SCENARIOS

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def select_base_price_baseline(df: pd.DataFrame, base_price: float) -> pd.Series | None:
    """Baseline 1: Pick eligible candidate closest to base_price."""
    viable = df[df["status"] == "VIABLE"]
    if viable.empty:
        return None
    idx = (viable["candidate_price"] - base_price).abs().idxmin()
    return viable.loc[idx]


def select_fixed_markup_baseline(df: pd.DataFrame, unit_cost: float, target_markup: float = 0.50) -> pd.Series | None:
    """Baseline 2: Pick eligible candidate closest to Cost + target_markup (e.g., 50%)."""
    viable = df[df["status"] == "VIABLE"]
    if viable.empty:
        return None
    target_price = unit_cost * (1.0 + target_markup)
    idx = (viable["candidate_price"] - target_price).abs().idxmin()
    return viable.loc[idx]


def select_max_conversion_baseline(df: pd.DataFrame) -> pd.Series | None:
    """Baseline 3: Pick eligible candidate with highest P(buy) (volume maximizer)."""
    viable = df[df["status"] == "VIABLE"]
    if viable.empty:
        return None
    idx = viable["p_buy"].idxmax()
    return viable.loc[idx]


def run_benchmarks():
    print("\n" + "=" * 115)
    print("  MONETIZE360 ML PROFIT OPTIMIZER vs 3 PRICING BASELINES")
    print("=" * 115)

    scenarios = {}
    scenarios.update(NEW_DOMAIN_SCENARIOS)
    scenarios.update(ABNORMAL_SCENARIOS)

    records = []
    json_output = {"benchmark_summary": [], "scenarios": {}}

    for key, spec in scenarios.items():
        params = dict(spec)
        s_name = params.pop("scenario_name")
        params.pop("description", None)
        c_prices = params.pop("candidate_prices")
        base_price = params["base_price"]
        unit_cost = params["unit_cost"]

        df = evaluate_candidates(candidate_prices=c_prices, artifacts_dir=ARTIFACTS_DIR, **params)

        # 1. Optimizer Winner
        opt_winner = df[df["verdict"].str.contains("WINNER")]
        opt_row = opt_winner.iloc[0] if not opt_winner.empty else None

        # 2. Baseline 1: Base Price
        b1_row = select_base_price_baseline(df, base_price)

        # 3. Baseline 2: Fixed Cost-Plus 50%
        b2_row = select_fixed_markup_baseline(df, unit_cost, target_markup=0.50)

        # 4. Baseline 3: Max Conversion
        b3_row = select_max_conversion_baseline(df)

        opt_profit = opt_row["expected_profit"] if opt_row is not None else 0.0
        b1_profit = b1_row["expected_profit"] if b1_row is not None else 0.0
        b2_profit = b2_row["expected_profit"] if b2_row is not None else 0.0
        b3_profit = b3_row["expected_profit"] if b3_row is not None else 0.0

        # Best alternative baseline
        best_base_profit = max(b1_profit, b2_profit, b3_profit)
        lift_vs_best = ((opt_profit - best_base_profit) / best_base_profit * 100) if best_base_profit > 0 else 0.0
        lift_vs_b1 = ((opt_profit - b1_profit) / b1_profit * 100) if b1_profit > 0 else 0.0

        records.append({
            "scenario": s_name,
            "b1_base": b1_profit,
            "b2_costplus": b2_profit,
            "b3_maxconv": b3_profit,
            "opt_profit": opt_profit,
            "opt_price": opt_row["candidate_price"] if opt_row is not None else None,
            "opt_pbuy": opt_row["p_buy"] if opt_row is not None else None,
            "lift_vs_base_price": lift_vs_b1,
            "lift_vs_best_baseline": lift_vs_best,
        })

        json_output["scenarios"][key] = {
            "name": s_name,
            "base_price": base_price,
            "unit_cost": unit_cost,
            "candidates": df.to_dict(orient="records"),
            "optimizer": {
                "price": float(opt_row["candidate_price"]) if opt_row is not None else None,
                "profit": float(opt_profit),
                "p_buy": float(opt_row["p_buy"]) if opt_row is not None else None,
            },
            "baseline_base_price": {
                "price": float(b1_row["candidate_price"]) if b1_row is not None else None,
                "profit": float(b1_profit),
            },
            "baseline_cost_plus_50": {
                "price": float(b2_row["candidate_price"]) if b2_row is not None else None,
                "profit": float(b2_profit),
            },
            "baseline_max_conversion": {
                "price": float(b3_row["candidate_price"]) if b3_row is not None else None,
                "profit": float(b3_profit),
            },
            "lift_vs_base_price_pct": round(lift_vs_b1, 2),
            "lift_vs_best_baseline_pct": round(lift_vs_best, 2),
        }

    # Print Comparison Table
    headers = [
        f"{'Scenario':<36s}",
        f"{'B1: Base Price':>14s}",
        f"{'B2: Cost+50%':>14s}",
        f"{'B3: Max Conv':>14s}",
        f"{'ML Optimizer':>14s}",
        f"{'Lift vs Base':>14s}",
    ]
    print(" | ".join(headers))
    print("-" * 115)

    total_opt = 0.0
    total_b1 = 0.0
    total_b2 = 0.0
    total_b3 = 0.0

    for r in records:
        opt_p = f"${r['opt_profit']:.2f}" if r['opt_profit'] > 0 else "$0.00"
        b1_p = f"${r['b1_base']:.2f}" if r['b1_base'] > 0 else "$0.00"
        b2_p = f"${r['b2_costplus']:.2f}" if r['b2_costplus'] > 0 else "$0.00"
        b3_p = f"${r['b3_maxconv']:.2f}" if r['b3_maxconv'] > 0 else "$0.00"
        lift_str = f"+{r['lift_vs_base_price']:.1f}%" if r['lift_vs_base_price'] > 0 else ("0.0%" if r['opt_profit'] > 0 else "N/A")

        total_opt += r['opt_profit']
        total_b1 += r['b1_base']
        total_b2 += r['b2_costplus']
        total_b3 += r['b3_maxconv']

        print(f"{r['scenario'][:36]:<36s} | {b1_p:>14s} | {b2_p:>14s} | {b3_p:>14s} | {opt_p:>14s} | {lift_str:>14s}")

    print("-" * 115)
    overall_lift = ((total_opt - total_b1) / total_b1 * 100) if total_b1 > 0 else 0.0
    print(f"{'TOTAL EXPECTED PROFIT (ALL SCENARIOS)':<36s} | ${total_b1:>13.2f} | ${total_b2:>13.2f} | ${total_b3:>13.2f} | ${total_opt:>13.2f} | +{overall_lift:>12.1f}%")
    print("=" * 115 + "\n")

    json_output["benchmark_summary"] = {
        "total_optimizer_profit": round(total_opt, 2),
        "total_base_price_profit": round(total_b1, 2),
        "total_cost_plus_profit": round(total_b2, 2),
        "total_max_conversion_profit": round(total_b3, 2),
        "overall_lift_vs_base_price_pct": round(overall_lift, 2),
    }

    # Save to disk
    out_file = ARTIFACTS_DIR / "benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"[benchmark] Saved machine-readable results to {out_file}\n")


if __name__ == "__main__":
    run_benchmarks()
