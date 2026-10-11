"""
verify_behavior.py - Rigorous checks for boundary conditions, elasticity grid, and competitor anchoring
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from features import ALL_FEATURES
from predict_best_price import evaluate_candidates, load_model

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def test_ceiling_boundary():
    """
    Test exact boundary condition for price ceiling:
    Base = $50.00, Max Change = 50% -> Ceiling is $75.00.
    Compare $74.99, $75.00, and $75.01.
    """
    print("\n" + "=" * 80)
    print("  CHECK 1: EXACT CEILING BOUNDARY CONDITION TEST")
    print("=" * 80)
    print("  Base Price: $50.00 | Max price change: +50% -> Ceiling = $75.00")
    
    candidates = [74.99, 75.00, 75.01]
    df = evaluate_candidates(
        candidate_prices=candidates,
        base_price=50.0,
        unit_cost=15.0,
        competitor_price=60.0,
        max_price_change_pct=0.50,
        artifacts_dir=ARTIFACTS_DIR,
    )
    
    for _, r in df.iterrows():
        p = r["candidate_price"]
        status = r["status"]
        verdict = r["verdict"]
        notes = r["notes"]
        print(f"  Price: ${p:.2f} | Status: {status:16s} | Verdict: {verdict}")
        print(f"    Notes: {notes}")
    print("-" * 80)


def test_competitor_anchoring_depth():
    """
    Investigate Issue 3: Why did conversion stay 67.9% at $75 when competitor was $25?
    Let's sweep competitor price from $25 up to $100 while holding candidate price fixed at $75.
    """
    print("\n" + "=" * 80)
    print("  CHECK 2: COMPETITOR ANCHORING ELASTICITY CHECK")
    print("=" * 80)
    print("  Holding candidate price = $75.00, Base price = $100.00, Unit Cost = $40.00")
    print("  Customer Sensitivity = 0.75, Competitor Density = 0.90")
    print("  Sweeping Competitor Price from $25 (dumping) to $100 (parity):")
    
    comp_prices = [25.0, 35.0, 50.0, 65.0, 75.0, 90.0, 100.0]
    for c_price in comp_prices:
        df = evaluate_candidates(
            candidate_prices=[75.0],
            base_price=100.0,
            unit_cost=40.0,
            competitor_price=c_price,
            customer_sensitivity=0.75,
            customer_loyalty_score=0.20,
            competitor_density=0.90,
            artifacts_dir=ARTIFACTS_DIR,
        )
        row = df.iloc[0]
        print(f"  Competitor: ${c_price:6.2f} (Our Price vs Comp: {75.0/c_price:4.2f}x) | P(buy): {row['p_buy']:.1%}")
    print("-" * 80)


def test_sensitivity_loyalty_grid():
    """
    Investigate Issue 4: Verify how sensitivity and loyalty interact across a 2D grid.
    Sweep Price from $900 to $1500 across 3 personas:
      1. Zero Sensitivity (sens=0.0, loyalty=1.0)
      2. Average Persona (sens=0.5, loyalty=0.5)
      3. Extreme Sensitive (sens=1.0, loyalty=0.0)
    """
    print("\n" + "=" * 80)
    print("  CHECK 3: SENSITIVITY x LOYALTY ELASTICITY GRID")
    print("=" * 80)
    print("  Base Price = $1000.00, Unit Cost = $250.00, Competitor = $1200.00")
    
    prices = [900.0, 1100.0, 1300.0, 1500.0]
    personas = [
        ("Monopoly Whale (sens=0.0, loy=1.0)", 0.0, 1.0),
        ("Average Shopper (sens=0.5, loy=0.5)", 0.5, 0.5),
        ("Extreme Sensitive (sens=1.0, loy=0.0)", 1.0, 0.0),
    ]
    
    for p_name, sens, loy in personas:
        print(f"\n  Persona: {p_name}")
        df = evaluate_candidates(
            candidate_prices=prices,
            base_price=1000.0,
            unit_cost=250.0,
            competitor_price=1200.0,
            customer_sensitivity=sens,
            customer_loyalty_score=loy,
            artifacts_dir=ARTIFACTS_DIR,
        )
        for _, r in df.iterrows():
            print(f"    Price: ${r['candidate_price']:7.2f} | P(buy): {r['p_buy']:5.1%} | E[Profit]: ${r['expected_profit']:7.2f}")


if __name__ == "__main__":
    test_ceiling_boundary()
    test_competitor_anchoring_depth()
    test_sensitivity_loyalty_grid()
