"""
evaluate.py — Model Evaluation & Diagnostics

Evaluations:
  1. Classification metrics (AUC, accuracy, precision, recall, F1)
  2. Calibration analysis (predicted P vs actual P)
  3. Price sensitivity monotonicity test
  4. Customer persona differentiation test
  5. Stress tests (edge cases)
  6. Comparison with LR baseline
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
)

from features import ALL_FEATURES, TARGET_COL, get_X_y, load_training_data, random_split

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DATA_DIR = Path(__file__).parent / "data"


def load_models():
    """Load both GBT and LR models."""
    with open(ARTIFACTS_DIR / "model_gbt.pkl", "rb") as f:
        gbt = pickle.load(f)
    with open(ARTIFACTS_DIR / "model_lr.pkl", "rb") as f:
        lr = pickle.load(f)
    with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return gbt, lr, scaler


def evaluate():
    """Run full evaluation suite."""
    print("\n" + "=" * 70)
    print("  DOMAIN-AGNOSTIC MODEL — FULL EVALUATION SUITE")
    print("=" * 70)

    # Load data and models
    df = load_training_data(DATA_DIR)
    _, _, test_df = random_split(df)
    gbt, lr, scaler = load_models()

    X_test, y_test = get_X_y(test_df)
    X_test_scaled = scaler.transform(X_test)

    # ── 1. Classification Metrics ----------------------------------------
    print("\n" + "-" * 70)
    print("  1. CLASSIFICATION METRICS (Test Set)")
    print("-" * 70)

    for name, model, X in [("GBT", gbt, X_test.values), ("LR", lr, X_test_scaled)]:
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs)

        print(f"\n  {name}:")
        print(f"    AUC:      {auc:.4f}")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    LogLoss:  {ll:.4f}")
        print(f"\n    Classification Report:")
        report = classification_report(y_test, preds, target_names=["No Buy", "Buy"])
        print(report)

    # ── 2. Calibration Analysis ------------------------------------------
    print("-" * 70)
    print("  2. CALIBRATION ANALYSIS (GBT)")
    print("-" * 70)

    gbt_probs = gbt.predict_proba(X_test.values)[:, 1]
    test_df = test_df.copy()
    test_df["predicted_prob"] = gbt_probs

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    test_df["prob_bin"] = pd.cut(test_df["predicted_prob"], bins=bins)
    cal = test_df.groupby("prob_bin", observed=True).agg(
        n=("purchased", "count"),
        actual_rate=("purchased", "mean"),
        mean_predicted=("predicted_prob", "mean"),
    )

    print(f"\n  {'Bin':20s} {'Count':>8s} {'Actual':>8s} {'Predicted':>10s} {'Gap':>8s}")
    print("  " + "-" * 56)
    for idx, row in cal.iterrows():
        gap = row["actual_rate"] - row["mean_predicted"]
        print(f"  {str(idx):20s} {int(row['n']):8d} {row['actual_rate']:8.3f} "
              f"{row['mean_predicted']:10.3f} {gap:+8.3f}")

    # ── 3. Price Sensitivity Monotonicity --------------------------------
    print("\n" + "-" * 70)
    print("  3. PRICE SENSITIVITY MONOTONICITY")
    print("-" * 70)

    _test_monotonicity(gbt, X_test)

    # ── 4. Customer Persona Differentiation ------------------------------
    print("\n" + "-" * 70)
    print("  4. CUSTOMER PERSONA DIFFERENTIATION")
    print("-" * 70)

    _test_persona_differentiation(gbt, X_test)

    # ── 5. Edge Case Stress Tests ----------------------------------------
    print("\n" + "-" * 70)
    print("  5. EDGE CASE STRESS TESTS")
    print("-" * 70)

    _test_edge_cases(gbt, X_test)

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70 + "\n")


def _test_monotonicity(model, X_train: pd.DataFrame):
    """Sweep price_ratio while holding everything else at median."""
    median_row = X_train.median()
    test_ratios = [0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50]

    personas = [
        ("Budget Customer (sensitivity=0.8)", 0.8, 0.2),
        ("Moderate Customer (sensitivity=0.5)", 0.5, 0.5),
        ("Loyal Premium (sensitivity=0.2)", 0.2, 0.8),
    ]

    for persona_name, sens, loyalty in personas:
        probs = []
        for ratio in test_ratios:
            row = median_row.copy()
            row["price_ratio"] = ratio
            row["log_price_ratio"] = np.log(ratio)
            row["customer_sensitivity"] = sens
            row["customer_loyalty_score"] = loyalty
            row_df = pd.DataFrame([row])[ALL_FEATURES]
            p = model.predict_proba(row_df.values)[0, 1]
            probs.append(p)

        is_mono = all(probs[i] >= probs[i+1] - 0.01 for i in range(len(probs)-1))
        status = "[PASS]" if is_mono else "[FAIL]"

        print(f"\n  {persona_name}: {status}")
        for ratio, p in zip(test_ratios, probs):
            bar = "#" * int(p * 40)
            print(f"    ratio={ratio:.2f}  P(buy)={p:.3f}  {bar}")


def _test_persona_differentiation(model, X_train: pd.DataFrame):
    """Same price, different customers -> should give different P(buy)."""
    median_row = X_train.median()
    median_row["price_ratio"] = 1.0
    median_row["log_price_ratio"] = 0.0

    personas = [
        ("Price-insensitive loyal",   0.15, 0.85),
        ("Moderate shopper",           0.50, 0.50),
        ("Price-sensitive new",        0.85, 0.15),
    ]

    print(f"\n  Same price (ratio=1.0), different customers:")
    prev_p = None
    for name, sens, loyalty in personas:
        row = median_row.copy()
        row["customer_sensitivity"] = sens
        row["customer_loyalty_score"] = loyalty
        row_df = pd.DataFrame([row])[ALL_FEATURES]
        p = model.predict_proba(row_df.values)[0, 1]
        print(f"    {name:30s}  sens={sens:.2f}  loyalty={loyalty:.2f}  P(buy)={p:.3f}")
        if prev_p is not None and p >= prev_p:
            print(f"      [WARN] Expected lower P for more sensitive customer")
        prev_p = p


def _test_edge_cases(model, X_train: pd.DataFrame):
    """Test extreme scenarios."""
    median_row = X_train.median()

    cases = [
        ("Massive discount (ratio=0.5)", {"price_ratio": 0.5, "log_price_ratio": np.log(0.5)}),
        ("Huge premium (ratio=2.0)", {"price_ratio": 2.0, "log_price_ratio": np.log(2.0)}),
        ("Flash sale + urgency", {"is_promotion_active": 1, "urgency_score": 0.95, "scarcity_score": 0.90}),
        ("No urgency, abundant supply", {"urgency_score": 0.0, "scarcity_score": 0.0, "demand_pressure": 0.3}),
        ("Monopoly (no competitors)", {"competitor_density": 0.0, "price_vs_competitor": 0.8}),
        ("Saturated market", {"competitor_density": 1.0, "price_vs_competitor": 1.3}),
    ]

    for name, overrides in cases:
        row = median_row.copy()
        for k, v in overrides.items():
            row[k] = v
        row_df = pd.DataFrame([row])[ALL_FEATURES]
        p = model.predict_proba(row_df.values)[0, 1]
        print(f"  {name:40s}  P(buy) = {p:.3f}")


if __name__ == "__main__":
    evaluate()
