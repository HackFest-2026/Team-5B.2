# Monetize360 ML_Updated — Domain-Agnostic Purchase Behavior Model

## What Is This?

A **completely domain-agnostic** ML pipeline that predicts whether a customer will buy at a given price, and then selects the price that **maximizes business profit** from a set of candidates.

**It works for any industry** — e-commerce, hospitality, car rental, SaaS, healthcare, luxury goods — without any domain-specific features.

---

## How It Works (End-to-End Flow)


┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  INPUT                                                                       │
│  ├── Base price ($100.00)                                                    │
│  ├── Unit cost ($40.00)                                                      │
│  ├── Competitor price ($95.00)                                               │
│  ├── Customer persona (sensitivity=0.65, loyalty=0.30)                       │
│  ├── Market context (urgency, scarcity, demand, competitors)                 │
│  └── 4-5 candidate prices ($80, $90, $100, $110, $120)                       │
│                                                                              │
│  ┌─────────────────────────────────────────┐                                 │
│  │ STEP 1: Feature Engineering             │                                 │
│  │ Convert all $ amounts to RATIOS         │                                 │
│  │ $90/$100 → ratio=0.90, log=-0.105       │                                 │
│  │ $90/$95  → vs_competitor=0.947          │                                 │
│  │ No domain info enters the model!        │                                 │
│  └─────────────────┬───────────────────────┘                                 │
│                    ▼                                                         │
│  ┌─────────────────────────────────────────┐                                 │
│  │ STEP 2: ML Model Prediction             │                                 │
│  │ GBT predicts P(buy) for EACH candidate  │                                 │
│  │ $80 → P=82%, $90 → P=71%, $100 → P=58% │                                 │
│  │ $110 → P=42%, $120 → P=28%             │                                 │
│  └─────────────────┬───────────────────────┘                                 │
│                    ▼                                                         │
│  ┌─────────────────────────────────────────┐                                 │
│  │ STEP 3: Profit Optimization             │                                 │
│  │ E[Profit] = P(buy) × (price - cost)     │                                 │
│  │ $80 → 82% × $40 = $32.80               │                                 │
│  │ $90 → 71% × $50 = $35.50  ← WINNER     │                                 │
│  │ $100 → 58% × $60 = $34.80              │                                 │
│  └─────────────────┬───────────────────────┘                                 │
│                    ▼                                                         │
│  ┌─────────────────────────────────────────┐                                 │
│  │ STEP 4: Business Guardrails             │                                 │
│  │ ✗ Price < unit_cost → REJECTED (loss)   │                                 │
│  │ ✗ Price > ceiling → REJECTED            │                                 │
│  │ ✗ P(buy) < 10% → REJECTED (too low)    │                                 │
│  │ ✓ Highest E[profit] among safe prices   │                                 │
│  └─────────────────┬───────────────────────┘                                 │
│                    ▼                                                         │
│  OUTPUT                                                                      │
│  └── Recommended price: $90.00                                               │
│      Expected profit: $35.50, Conversion: 71%                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Dataset Schema

### Training Features (14 — All Domain-Agnostic)

| # | Feature | Type | Range | What It Captures |
|---|---------|------|-------|------------------|
| 1 | `price_ratio` | float | 0.6–1.5 | candidate_price / base_price (1.0 = at reference) |
| 2 | `log_price_ratio` | float | −0.5 to +0.4 | Log-transformed ratio (symmetrizes discounts/premiums) |
| 3 | `price_vs_competitor` | float | 0.6–1.5 | candidate_price / competitor_price (>1 = overpriced) |
| 4 | `margin_ratio` | float | varies | (price − cost) / base_price (business margin signal) |
| 5 | `price_rank_normalized` | float | 0.0–1.0 | Rank among candidates (0=cheapest, 1=most expensive) |
| 6 | `customer_sensitivity` | float | 0.0–1.0 | How price-sensitive the customer is |
| 7 | `customer_loyalty_score` | float | 0.0–1.0 | Customer loyalty/stickiness level |
| 8 | `urgency_score` | float | 0.0–1.0 | How urgently the customer needs the product |
| 9 | `scarcity_score` | float | 0.0–1.0 | Supply scarcity (FOMO driver) |
| 10 | `demand_pressure` | float | 0.2–2.0 | Market demand level (1.0 = normal) |
| 11 | `competitor_density` | float | 0.0–1.0 | Market competition intensity |
| 12 | `is_promotion_active` | int | 0 or 1 | Whether a promotion is currently running |
| 13 | `time_decay_factor` | float | 0.0–1.0 | Time pressure (offer expiry, booking window) |
| 14 | `perceived_value_gap` | float | varies | (perceived_value − price) / base_price |

### Target Label

| Column | Type | Values | Meaning |
|--------|------|--------|---------|
| `purchased` | int | 0 or 1 | 0 = customer walked away, 1 = customer bought |

### Why No Domain Features?

- **No `category`** (electronics, fashion) — replaced by `price_ratio` which is universally meaningful
- **No `room_type`** or `car_class` — replaced by behavioral signals
- **No raw dollar amounts** in features — everything is a **ratio** or **score**
- A `price_ratio = 1.15` means "15% above reference" whether it's a $10 grocery item or a $5,000 luxury watch

---

## Files

```
ml_updated/
├── __init__.py                 # Package init
├── generate_dataset.py         # Synthetic data generator (behavioral economics)
├── features.py                 # Feature schema & engineering (14 features)
├── train.py                    # Model training (GBT + LR baseline)
├── predict_best_price.py       # Price optimizer with demo scenarios
├── evaluate.py                 # Full evaluation suite
├── data/                       # Generated training data (CSV)
│   ├── training_data.csv       # 400K rows of domain-agnostic training data
│   ├── products.csv            # Product archetypes
│   ├── customers.csv           # Customer profiles
│   ├── manifest.json           # Dataset metadata
│   └── ground_truth_coefficients.json  # Simulation ground truth
├── artifacts/                  # Trained model artifacts
│   ├── model_gbt.pkl           # Primary model (Gradient Boosted Trees)
│   ├── model_lr.pkl            # Baseline model (Logistic Regression)
│   ├── scaler.pkl              # StandardScaler for LR
│   ├── training_metadata.json  # Full training report
│   └── feature_importance.csv  # GBT feature importances
└── README.md                   # This file
```

---

## Usage

### 1. Generate Dataset
```bash
python generate_dataset.py
```
Generates 400,000 training rows (80K quotes × 5 candidates each).

### 2. Train Model
```bash
python train.py
```
Trains GBT (primary) and Logistic Regression (baseline). Saves to `artifacts/`.

### 3. Predict Best Price
```bash
# Run demo scenarios across 6 domains
python predict_best_price.py --demo

# Custom scenario
python predict_best_price.py \
  --prices 80 90 100 110 120 \
  --base-price 100 \
  --unit-cost 40 \
  --competitor-price 95 \
  --sensitivity 0.6 \
  --loyalty 0.3
```

### 4. Full Evaluation
```bash
python evaluate.py
```

---

## How Does It Decide the Business Verdict?

The system makes **two separate decisions** that work together:

### Decision 1: ML Model → "Will this customer buy?"
The model outputs `P(buy)` — the probability that this specific customer, in this market context, will purchase at this candidate price. This is purely a **behavioral prediction**.

### Decision 2: Profit Optimizer → "Is this price good for business?"
For each viable candidate price:
```
Expected Profit = P(buy) × (candidate_price − unit_cost)
```

The optimizer selects the price with the **highest expected profit** among candidates that pass all business guardrails:
- ✗ No selling below cost (loss prevention)
- ✗ No exceeding price ceiling (brand/regulatory compliance)
- ✗ No prices with < 10% predicted conversion (too risky)
- ✓ Maximize E[Profit] among remaining safe candidates

### Why This Works

A cheap price has high conversion but low margin → moderate E[profit].
An expensive price has high margin but low conversion → moderate E[profit].
The **optimal price** is the one where `P(buy) × margin` peaks.

---

## How Does It Capture Customer Behavior?

The model learns from **Behavioral Economics** principles:

1. **Price Sensitivity × Price Interaction** — The KEY feature. A price-sensitive customer reacts sharply to a 10% price increase; a loyal customer barely notices.

2. **Loss Aversion** — Customers perceive prices above the reference (ratio > 1.0) more negatively than they perceive equivalent discounts positively. This is encoded in the asymmetric response to `log_price_ratio`.

3. **Competitor Anchoring** — Customers compare your price to competitors. `price_vs_competitor > 1` means you're more expensive → conversion drops.

4. **Urgency & Scarcity** — FOMO effects. High urgency + high scarcity = customers tolerate higher prices.

5. **Social Proof** — High demand pressure signals "others are buying" → positive herd effect.

6. **Perceived Value Gap** — When perceived value exceeds the asking price, customers feel they're getting a deal → conversion increases.

---

## Model Training Details

| Property | Value |
|----------|-------|
| Algorithm | GradientBoostingClassifier (scikit-learn) |
| Number of Trees | 300 |
| Max Depth | 5 |
| Learning Rate | 0.05 |
| Training Rows | 279,999 (70% split of 400,000 total) |
| Validation Rows | 60,000 |
| Test Rows | 60,001 |
| Features | 14 (all numeric, domain-agnostic) |
| Target | `purchased` (binary: 0 or 1, 65.9% base conversion) |
| Baseline | Logistic Regression (for comparison) |
| Preprocessing | StandardScaler for LR; raw numeric values for GBT |
| Test Set ROC-AUC | 0.8072 (GBT) vs 0.8084 (LR) |
| Test Set Accuracy | 75.60% (GBT) vs 72.61% (LR) |
| Monotonicity Test | 100% PASS across all personas |

---

## Final Feature Importances (GBT)

| Feature | Importance | Interpretation |
|---------|------------|----------------|
| `price_ratio` | 31.8% | Primary driver: offered price relative to base price |
| `log_price_ratio` | 17.2% | Logarithmic diminishing returns and loss aversion curvature |
| `perceived_value_gap` | 16.4% | Perceived product value surplus relative to price |
| `price_vs_competitor` | 10.3% | Market anchoring against competitor quotes |
| `customer_loyalty_score` | 4.7% | Loyalty cushioning against price sensitivity |
| `urgency_score` | 3.5% | Urgency / time pressure to transact |
| `price_rank_normalized` | 2.8% | Relative rank within candidate set |
| `is_promotion_active` | 2.6% | Active promotional framing effect |
| `customer_sensitivity` | 2.5% | Base customer price sensitivity |
| `scarcity_score` | 1.9% | Scarcity / remaining inventory pressure |
| `time_decay_factor` | 1.9% | Offer expiration window |
| `competitor_density` | 1.8% | Number of alternative market choices |
| `margin_ratio` | 1.8% | Profit margin room |
| `demand_pressure` | 0.9% | Market-wide aggregate demand / social proof |

---

## Cross-Domain & Abnormal Edge-Case Test Suite

Run the comprehensive test suite covering **11 diverse scenarios**:

```bash
# Full test suite with diagnostic reports
python test_scenarios.py

# Executive summary table only
python test_scenarios.py --summary
```

### Domains Tested:
- **E-Commerce** (Electronics, Retail)
- **Hospitality** (Hotels, Vacation Rentals)
- **Car Rental** (Transportation)
- **SaaS** (Monthly Subscriptions)
- **Healthcare** (Medical Devices)
- **Luxury Goods** (Timepieces, Fine Jewelry)
- **Cloud Infrastructure** (AI GPU Compute per hour)
- **Airlines** (Dynamic Flight Seats)
- **EdTech** (Upskilling Courses)
- **Commercial Real Estate** (Coworking Space Leases)
- **B2B Logistics** (Freight Forwarding per pallet)

### Abnormal Edge Cases Tested:
- **Case A: Below-Cost Pricing Trap** — Disqualifies suicide discounts despite high customer conversion.
- **Case B: Crisis Shortage Gouging** — Prevents runaway price hikes via guardrail ceilings.
- **Case C: Inverted Market (Competitor Dumping)** — Accurately penalizes extreme price anchoring gap.
- **Case D: Hyper-Sensitive Bargain Hunter** — Captures cliff drop when price exceeds competitor.
- **Case E: Monopoly Whale** — Extracts optimal profit from zero-elasticity loyal customers.
- **Case F: Total Guardrail Disqualification** — Correctly refuses to transact when all prices lose money.

