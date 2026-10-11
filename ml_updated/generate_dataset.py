"""
generate_dataset.py — Synthetic Domain-Agnostic Dataset Generator

Generates training data for a purchase-probability model that works
across ANY domain (e-commerce, hospitality, car rental, banking, SaaS, etc.)

KEY DESIGN PRINCIPLES:
======================
1. ZERO domain-specific attributes — no "category", "room_type", "car_class", etc.
2. All features are expressed as NORMALIZED RATIOS and BEHAVIORAL SIGNALS
   that are universally meaningful regardless of what is being sold.
3. Purchase decisions are modeled using behavioral economics:
   - Prospect Theory (loss aversion around reference price)
   - Price sensitivity varies by customer persona
   - Market pressure (urgency, scarcity) shifts willingness to pay
   - Competitor anchoring affects perceived value

GENERATED FEATURES (all observable at quote time):
==================================================
  price_ratio            float  candidate_price / base_price      (1.0 = at reference)
  log_price_ratio        float  log(price_ratio)                  (symmetrizes discounts/premiums)
  price_vs_competitor    float  candidate_price / competitor_price (>1 = pricier than market)
  margin_ratio           float  (candidate_price - unit_cost) / base_price  (business health signal)
  price_rank_normalized  float  rank of this candidate among all candidates offered (0=cheapest, 1=most expensive)
  customer_sensitivity   float  0.0 (price-insensitive) to 1.0 (very price-sensitive)
  customer_loyalty_score float  0.0 (new customer) to 1.0 (deeply loyal)
  urgency_score          float  0.0 (no rush) to 1.0 (must buy now)
  scarcity_score         float  0.0 (abundant supply) to 1.0 (almost sold out)
  demand_pressure        float  0.0 (low demand) to 2.0 (extreme demand)
  competitor_density     float  0.0 (monopoly) to 1.0 (saturated market)
  is_promotion_active    int    0 or 1 — any active promotion/discount event
  time_decay_factor      float  0.0 to 1.0 — temporal urgency (booking window, offer expiry)
  perceived_value_gap    float  (perceived_value - candidate_price) / base_price

TARGET:
=======
  purchased              int    0 = did not buy, 1 = bought

PURCHASE PROBABILITY MODEL (behavioral economics):
===================================================
  logit(P) = β₀
           + β₁ · log_price_ratio           (higher price → lower P)
           + β₂ · price_vs_competitor        (pricier than competitors → lower P)
           + β₃ · customer_sensitivity       (sensitive customers react more to price)
           + β₄ · interaction(log_price × sensitivity)  (key interaction)
           + β₅ · customer_loyalty_score     (loyal customers tolerate higher prices)
           + β₆ · urgency_score              (urgent need → higher P)
           + β₇ · scarcity_score             (scarcity → higher P)
           + β₈ · demand_pressure            (high demand → social proof)
           + β₉ · competitor_density         (more competitors → lower P for overpriced)
           + β₁₀ · is_promotion_active       (promotions boost conversion)
           + β₁₁ · time_decay_factor         (urgency from time pressure)
           + β₁₂ · perceived_value_gap       (positive gap → higher P)
           + noise

  The noise is heteroscedastic — it increases with price_ratio to simulate
  real-world uncertainty at extreme prices.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_N_QUOTES = 80_000        # Total quote records
DEFAULT_N_PRODUCTS = 500         # Unique product archetypes
DEFAULT_N_CUSTOMERS = 2000       # Unique customer profiles
DEFAULT_CANDIDATES_PER_QUOTE = 5 # Candidate prices per quote event
SEED = 42

# Behavioral economics coefficients (ground truth for simulation)
GROUND_TRUTH_COEFFICIENTS = {
    "intercept":           -0.40,     # base willingness (logit scale) — tuned for ~55% purchase rate
    "log_price_ratio":     -3.80,     # STRONG negative: price ↑ → buy ↓
    "price_vs_competitor":  -1.40,    # competitor anchoring
    "customer_sensitivity": -1.00,    # sensitive customers are harder to convert
    "price_x_sensitivity":  -2.80,    # KEY INTERACTION: sensitive + expensive = very low P
    "loyalty_score":         1.20,    # loyal customers tolerate higher prices
    "urgency_score":         1.50,    # urgency boosts conversion
    "scarcity_score":        0.90,    # scarcity FOMO
    "demand_pressure":       0.40,    # social proof / herd behavior
    "competitor_density":   -0.60,    # crowded market penalizes overpricing
    "is_promotion_active":   0.70,    # promotions help conversion
    "time_decay_factor":     0.80,    # time pressure to act
    "perceived_value_gap":   1.80,    # positive value surplus → buy
    "noise_scale":           0.45,    # base noise (heteroscedastic) — increased for realism
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def generate_products(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate product archetypes with base prices and unit costs."""
    # Base prices span a very wide range to be domain-agnostic
    # (could be $2 grocery item or $50,000 luxury goods)
    log_base_prices = rng.uniform(np.log(5), np.log(5000), size=n)
    base_prices = np.exp(log_base_prices).round(2)

    # Unit cost is 30-80% of base price (realistic margin structures)
    cost_ratios = rng.uniform(0.30, 0.80, size=n)
    unit_costs = (base_prices * cost_ratios).round(2)

    return pd.DataFrame({
        "product_id": [f"P{i:05d}" for i in range(n)],
        "base_price": base_prices,
        "unit_cost": unit_costs,
        "cost_ratio": cost_ratios.round(4),
    })


def generate_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate customer profiles with behavioral attributes."""
    # Customer sensitivity: Beta distribution (most people are moderate)
    sensitivity = rng.beta(2.5, 2.5, size=n).round(4)

    # Loyalty score: mixture of new (low) and established (high) customers
    loyalty_base = rng.beta(1.5, 3.0, size=n)
    # 20% of customers are highly loyal
    loyal_mask = rng.random(n) < 0.20
    loyalty_base[loyal_mask] = rng.beta(5.0, 1.5, size=loyal_mask.sum())
    loyalty_score = np.clip(loyalty_base, 0, 1).round(4)

    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "customer_sensitivity": sensitivity,
        "customer_loyalty_score": loyalty_score,
    })


def generate_market_contexts(n_quotes: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate market context signals for each quote event."""
    n = n_quotes

    # Urgency: mostly low, some high (exponential-ish)
    urgency = np.clip(rng.beta(1.5, 4.0, size=n), 0, 1).round(4)

    # Scarcity: mostly moderate, with tail of high scarcity
    scarcity = np.clip(rng.beta(2.0, 3.0, size=n), 0, 1).round(4)

    # Demand pressure: centered around 1.0, range [0.2, 2.0]
    demand = np.clip(rng.normal(1.0, 0.3, size=n), 0.2, 2.0).round(4)

    # Competitor density: uniform — some markets are monopolistic, some saturated
    competitor_density = rng.uniform(0.0, 1.0, size=n).round(4)

    # Promotion: 15% of quotes happen during a promotion
    is_promotion = (rng.random(n) < 0.15).astype(int)

    # Time decay: simulates booking window / offer expiry urgency
    time_decay = np.clip(rng.beta(2.0, 2.0, size=n), 0, 1).round(4)

    return pd.DataFrame({
        "urgency_score": urgency,
        "scarcity_score": scarcity,
        "demand_pressure": demand,
        "competitor_density": competitor_density,
        "is_promotion_active": is_promotion,
        "time_decay_factor": time_decay,
    })


def generate_candidate_prices(
    base_prices: np.ndarray,
    n_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate candidate prices as ratios of base price.
    Returns shape (n_quotes, n_candidates).

    Candidates span from 15% discount to 40% premium, with realistic clustering.
    """
    n_quotes = len(base_prices)

    # Generate candidate ratios: biased toward competitive range [0.85, 1.25]
    # but with some exploration in [0.60, 1.50]
    ratios = np.zeros((n_quotes, n_candidates))

    for i in range(n_candidates):
        # Mix of tight and wide exploration
        tight = rng.normal(1.0, 0.10, size=n_quotes)
        wide = rng.normal(1.0, 0.20, size=n_quotes)
        mix_mask = rng.random(n_quotes) < 0.30  # 30% wide exploration
        raw = np.where(mix_mask, wide, tight)
        # Offset each candidate slightly to ensure diversity
        offset = (i - n_candidates / 2) * 0.06
        raw += offset
        ratios[:, i] = np.clip(raw, 0.60, 1.50)

    # Sort candidates so they go from cheapest to most expensive
    ratios.sort(axis=1)

    # Convert to actual prices
    candidate_prices = (base_prices[:, np.newaxis] * ratios).round(2)

    return candidate_prices


def compute_competitor_prices(
    base_prices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Competitor price = base_price * random factor around 1.0"""
    factor = np.clip(rng.normal(1.0, 0.12, size=len(base_prices)), 0.70, 1.40)
    return (base_prices * factor).round(2)


def compute_perceived_value(
    base_prices: np.ndarray,
    loyalty_scores: np.ndarray,
    demand_pressure: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Perceived value is what the customer thinks the item is worth.
    It's influenced by:
      - Base price (anchor)
      - Loyalty (loyal customers perceive higher value)
      - Demand (high demand = social proof of value)
      - Individual noise (subjective)
    """
    value_multiplier = (
        1.0
        + 0.15 * loyalty_scores      # loyal → higher perceived value
        + 0.10 * (demand_pressure - 1.0)  # high demand → FOMO
        + rng.normal(0, 0.08, size=len(base_prices))  # personal taste
    )
    return (base_prices * np.clip(value_multiplier, 0.70, 1.50)).round(2)


def compute_purchase_probability(
    log_price_ratio: np.ndarray,
    price_vs_competitor: np.ndarray,
    customer_sensitivity: np.ndarray,
    loyalty_score: np.ndarray,
    urgency: np.ndarray,
    scarcity: np.ndarray,
    demand_pressure: np.ndarray,
    competitor_density: np.ndarray,
    is_promotion: np.ndarray,
    time_decay: np.ndarray,
    perceived_value_gap: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute purchase probability using the behavioral economics model.
    Returns P(buy) ∈ [0, 1] for each quote.
    """
    C = GROUND_TRUTH_COEFFICIENTS

    logit = (
        C["intercept"]
        + C["log_price_ratio"] * log_price_ratio
        + C["price_vs_competitor"] * (price_vs_competitor - 1.0)  # centered at 1
        + C["customer_sensitivity"] * customer_sensitivity
        + C["price_x_sensitivity"] * (log_price_ratio * customer_sensitivity)
        + C["loyalty_score"] * loyalty_score
        + C["urgency_score"] * urgency
        + C["scarcity_score"] * scarcity
        + C["demand_pressure"] * (demand_pressure - 1.0)
        + C["competitor_density"] * competitor_density
        + C["is_promotion_active"] * is_promotion
        + C["time_decay_factor"] * time_decay
        + C["perceived_value_gap"] * perceived_value_gap
    )

    # Heteroscedastic noise: more uncertainty at extreme prices
    noise_scale = C["noise_scale"] * (1.0 + 0.5 * np.abs(log_price_ratio))
    noise = rng.normal(0, noise_scale)

    prob = _sigmoid(logit + noise)
    return prob


def generate_dataset(
    n_quotes: int = DEFAULT_N_QUOTES,
    n_products: int = DEFAULT_N_PRODUCTS,
    n_customers: int = DEFAULT_N_CUSTOMERS,
    n_candidates: int = DEFAULT_CANDIDATES_PER_QUOTE,
    seed: int = SEED,
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Generate the complete domain-agnostic training dataset.

    Each row represents one (quote_event, candidate_price) pair.
    Total rows ≈ n_quotes × n_candidates.

    Returns the full DataFrame and saves to CSV if output_dir is specified.
    """
    rng = np.random.default_rng(seed)

    print(f"\n{'='*70}")
    print(f"  DOMAIN-AGNOSTIC DATASET GENERATOR")
    print(f"  Quotes: {n_quotes:,} | Products: {n_products:,} | Customers: {n_customers:,}")
    print(f"  Candidates per quote: {n_candidates}")
    print(f"{'='*70}\n")

    # --- Step 1: Generate archetypes ---
    print("[1/6] Generating product archetypes...")
    products = generate_products(n_products, rng)

    print("[2/6] Generating customer profiles...")
    customers = generate_customers(n_customers, rng)

    # --- Step 2: Sample quote events ---
    print("[3/6] Sampling quote events...")
    quote_product_idx = rng.integers(0, n_products, size=n_quotes)
    quote_customer_idx = rng.integers(0, n_customers, size=n_quotes)

    base_prices = products["base_price"].values[quote_product_idx]
    unit_costs = products["unit_cost"].values[quote_product_idx]
    cust_sensitivity = customers["customer_sensitivity"].values[quote_customer_idx]
    cust_loyalty = customers["customer_loyalty_score"].values[quote_customer_idx]

    # --- Step 3: Generate market contexts ---
    print("[4/6] Generating market contexts...")
    contexts = generate_market_contexts(n_quotes, rng)

    # --- Step 4: Generate candidate prices ---
    print("[5/6] Generating candidate prices and computing features...")
    candidate_prices = generate_candidate_prices(base_prices, n_candidates, rng)
    competitor_prices = compute_competitor_prices(base_prices, rng)
    perceived_values = compute_perceived_value(
        base_prices, cust_loyalty, contexts["demand_pressure"].values, rng
    )

    # --- Step 5: Expand to (quote × candidate) rows ---
    rows = []
    all_probs = []

    for c_idx in range(n_candidates):
        n = n_quotes
        cp = candidate_prices[:, c_idx]

        price_ratio = np.clip(cp / np.maximum(base_prices, 0.01), 0.1, 10.0)
        log_price_ratio = np.log(price_ratio)
        pvc = np.clip(cp / np.maximum(competitor_prices, 0.01), 0.1, 10.0)
        margin_ratio = ((cp - unit_costs) / np.maximum(base_prices, 0.01)).round(4)
        rank_norm = np.full(n, c_idx / max(n_candidates - 1, 1))
        pvg = ((perceived_values - cp) / np.maximum(base_prices, 0.01)).round(4)

        prob = compute_purchase_probability(
            log_price_ratio=log_price_ratio,
            price_vs_competitor=pvc,
            customer_sensitivity=cust_sensitivity,
            loyalty_score=cust_loyalty,
            urgency=contexts["urgency_score"].values,
            scarcity=contexts["scarcity_score"].values,
            demand_pressure=contexts["demand_pressure"].values,
            competitor_density=contexts["competitor_density"].values,
            is_promotion=contexts["is_promotion_active"].values,
            time_decay=contexts["time_decay_factor"].values,
            perceived_value_gap=pvg,
            rng=rng,
        )

        purchased = (rng.random(n) < prob).astype(int)

        chunk = pd.DataFrame({
            "quote_id": [f"Q{q:06d}" for q in range(n)],
            "product_id": products["product_id"].values[quote_product_idx],
            "customer_id": customers["customer_id"].values[quote_customer_idx],
            "candidate_idx": c_idx,
            "candidate_price": cp,
            "base_price": base_prices,
            "unit_cost": unit_costs,
            "competitor_price": competitor_prices,
            # --- Features ---
            "price_ratio": price_ratio.round(4),
            "log_price_ratio": log_price_ratio.round(4),
            "price_vs_competitor": pvc.round(4),
            "margin_ratio": margin_ratio,
            "price_rank_normalized": rank_norm.round(4),
            "customer_sensitivity": cust_sensitivity.round(4),
            "customer_loyalty_score": cust_loyalty.round(4),
            "urgency_score": contexts["urgency_score"].values,
            "scarcity_score": contexts["scarcity_score"].values,
            "demand_pressure": contexts["demand_pressure"].values,
            "competitor_density": contexts["competitor_density"].values,
            "is_promotion_active": contexts["is_promotion_active"].values,
            "time_decay_factor": contexts["time_decay_factor"].values,
            "perceived_value_gap": pvg,
            # --- Target ---
            "purchased": purchased,
            # --- Eval-only (NOT used in training features) ---
            "true_purchase_probability": prob.round(6),
        })
        rows.append(chunk)

    print("[6/6] Assembling final dataset...")
    df = pd.concat(rows, ignore_index=True)

    # Shuffle so candidates from the same quote aren't adjacent
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # --- Stats ---
    print(f"\n  Total rows        : {len(df):,}")
    print(f"  Unique quotes     : {df['quote_id'].nunique():,}")
    print(f"  Unique products   : {df['product_id'].nunique():,}")
    print(f"  Unique customers  : {df['customer_id'].nunique():,}")
    print(f"  Purchase rate     : {df['purchased'].mean():.1%}")
    print(f"  Price ratio range : [{df['price_ratio'].min():.2f}, {df['price_ratio'].max():.2f}]")
    print(f"  Base price range  : [${df['base_price'].min():.2f}, ${df['base_price'].max():.2f}]")

    # --- Save ---
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "training_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")

        # Save product and customer lookup tables
        products.to_csv(output_dir / "products.csv", index=False)
        customers.to_csv(output_dir / "customers.csv", index=False)
        print(f"  Saved: {output_dir / 'products.csv'}")
        print(f"  Saved: {output_dir / 'customers.csv'}")

        # Save ground truth coefficients for audit
        gt_path = output_dir / "ground_truth_coefficients.json"
        with open(gt_path, "w") as f:
            json.dump(GROUND_TRUTH_COEFFICIENTS, f, indent=2)
        print(f"  Saved: {gt_path}")

        # Save manifest
        manifest = {
            "dataset": "domain_agnostic_purchase_behavior",
            "version": "2.0.0",
            "synthetic": True,
            "n_rows": len(df),
            "n_quotes": int(df["quote_id"].nunique()),
            "n_products": n_products,
            "n_customers": n_customers,
            "n_candidates_per_quote": n_candidates,
            "purchase_rate": float(df["purchased"].mean()),
            "features": {
                "price_features": [
                    "price_ratio", "log_price_ratio", "price_vs_competitor",
                    "margin_ratio", "price_rank_normalized"
                ],
                "customer_features": [
                    "customer_sensitivity", "customer_loyalty_score"
                ],
                "market_features": [
                    "urgency_score", "scarcity_score", "demand_pressure",
                    "competitor_density", "is_promotion_active",
                    "time_decay_factor"
                ],
                "value_features": [
                    "perceived_value_gap"
                ],
            },
            "target": "purchased",
            "leakage_excluded": ["true_purchase_probability"],
            "domain_specific_attributes": "NONE — fully domain agnostic",
        }
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Saved: {output_dir / 'manifest.json'}")

    print(f"\n{'='*70}")
    print(f"  Dataset generation complete!")
    print(f"{'='*70}\n")

    return df


if __name__ == "__main__":
    output = Path(__file__).parent / "data"
    df = generate_dataset(output_dir=output)
