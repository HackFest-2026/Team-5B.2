# Monetize360 Customer-Agent Validation Module (`customer_agent_sim`)

A standalone, domain-agnostic customer-agent simulation and algorithmic validation module in Python.

This module provides programmatic behavioral validation positioned downstream of Monetize360's machine learning pricing optimizers. It simulates heterogeneous customer agent populations, models their individual choice utilities under varied market environments, evaluates the economic plausibility of candidate prices, and outputs structured diagnostics (`PASS`, `FAIL`, `REVIEW_REQUIRED`, or `ERROR`).

```
Pipeline Flow:
Pricing Engine → ML Probability Estimation → Price Optimization → Customer-Agent Validation → PASS / FAIL / REVIEW_REQUIRED
```

---

## 1. Purpose and Shadow-Mode Guarantee

### Purpose
In dynamic pricing architectures, machine learning models estimate purchase probabilities $P(\text{buy} \mid \text{price}, \mathbf{x})$, and constrained mathematical optimizers select prices that maximize expected profit or revenue. While optimizers enforce hard policy floors, ceilings, and evidence bounds, ML models can still suffer from:
- **Extrapolation artifacts and uncalibrated probabilities**: Overestimating purchase probability for extreme prices.
- **Demand curve non-monotonicity**: Predicting higher conversion for higher prices in standard consumer settings without justifiable quality signals.
- **Fragility under market shocks**: Failing to account for consumer budget constraints, competitor discounting, or drops in perceived value.
- **Sub-optimal ranking**: Selecting a candidate price that drops significantly in consumer preference when evaluated against alternative candidate prices.

The `customer_agent_sim` package introduces a programmatic, non-LLM Monte Carlo sanity gate that evaluates recommended prices against an independent population of utility-maximizing simulated consumers before quotes are deployed.

### Shadow-Mode Guarantee (Observational Only)
Within Monetize360, this module operates strictly in **shadow mode**:

1. **Post-Selection Execution**: Validation runs strictly *after* the optimizer has selected its optimal price (`select_optimal_price` or `select_optimal_price_from_objective`).
2. **Zero Invariant Modification**: The simulator **never** alters or overrides:
   - `optimal_price` (the price chosen by the optimizer)
   - `expected_profit` or `optimal_objective` (the objective value calculated by the optimizer)
   - `optimal_index` (the candidate grid index chosen by the optimizer)
   - `is_price_feasible` (feasibility remains governed solely by hard statutory, margin, and evidence constraints)
3. **Zero Order Eligibility Modification**: Fulfillment and underwriting eligibility (`order_eligible`, `can_checkout`, `eligibility_reason`) remain strictly determined by business rules (inventory on hand, available rooms, credit score / DTI). Validation status never blocks checkout.
4. **Fault Containment**: If the simulator encounters an internal error or invalid inputs, the exception is caught, logged, and returned as an `ERROR` status inside the diagnostic payload. Pricing and quote generation are **never interrupted or blocked**.
5. **Observational Diagnostics**: The validation gate attaches an advisory `shadow_validation` dictionary to the optimizer's returned dictionary, logging diagnostics for monitoring, calibration, and analyst review.

---

## 2. Architecture and Module Responsibilities

The package is self-contained within `customer_agent_sim/` and has zero runtime dependencies on LLMs, heavy neural networks, or external network services.

```
customer_agent_sim/
├── __init__.py               # Primary public entrypoints (validate_offer, exports)
├── models.py                 # Domain-neutral dataclasses (Offer, Traits, State, Report, Config)
├── agent.py                  # CustomerAgent entity encapsulating state, traits, and evaluation
├── population.py             # Heterogeneous population generator and subgroup partitioning
├── decision_policy.py        # Behavioral choice policies (StandardDecisionPolicy logit model)
├── runner.py                 # Monte Carlo execution orchestrator (Common Random Numbers)
├── scenarios.py              # Market shock and sensitivity transformations (STANDARD_SCENARIOS)
├── validation.py             # Core 5 validation capabilities and unified Decision Gate
├── metrics.py                # Wilson score confidence intervals, variance, rank metrics
├── feedback.py               # Structured feedback signals and bounded retry pipeline
├── config/
│   └── default_config.json   # Configurable thresholds and default scenario parameters
├── examples/
│   └── run_demo.py           # End-to-end runnable demonstration script
└── tests/                    # 32 unit and regression tests
    ├── test_agent_behaviour.py
    ├── test_population.py
    ├── test_scenarios.py
    ├── test_ranking_stability.py
    ├── test_validation_gate.py
    ├── test_reproducibility.py
    └── test_feedback_loop.py
```

### Module Responsibilities

| Module | Core Responsibility |
| :--- | :--- |
| `models.py` | Defines domain-neutral data schemas: `Offer`, `AgentTraits`, `AgentState`, `CandidatePriceResult`, `SimulationMetrics`, `ValidationConfig`, `ValidationReport`, and `FeedbackDiagnostic`. |
| `agent.py` | Represents an individual customer agent; computes choice probabilities and samples discrete decisions (`BUY`, `DEFER`, `REJECT`). |
| `population.py` | Generates heterogeneous cohorts via continuous probability distributions (Lognormal, Normal) with deterministic random seeds. |
| `decision_policy.py`| Implements the multinomial logit choice model combining surplus, price disutility, loss aversion, budget penalties, and loyalty. |
| `runner.py` | Orchestrates batch simulations and candidate price evaluation using Common Random Numbers (CRN) for variance reduction. |
| `scenarios.py` | Applies ceteris-paribus transformations to the exact same customer population to test resilience against market shifts. |
| `validation.py` | Implements the five independent validation tests and arbitrates the unified PASS / FAIL / REVIEW_REQUIRED decision gate. |
| `feedback.py` | Translates validation findings into structured `FeedbackDiagnostic` signals and bounds iterative optimization retries. |
| `metrics.py` | Computes statistical metrics: Wilson score intervals, sample variance, relative rank metrics. |

### Execution Flow

```
1. Input: Offer, ml_probability, candidate_prices, ml_probabilities, config
   ↓
2. Population Generation: Generate N agents (or reuse pre-generated population)
   ↓
3. Baseline Simulation: Compute conversion rate, Wilson 95% CI, and choice probabilities
   ↓
4. Five Independent Validation Checks:
   ├── Check A: Behavioural Consistency (Price ladder monotonicity & value sensitivity)
   ├── Check B: Population Heterogeneity (Variance threshold & trait subgroup divergence)
   ├── Check C: Scenario Robustness (Ceteris-paribus market stress tests)
   ├── Check D: Ranking Stability (Common Random Numbers ranking comparison)
   └── Check E: ML Discrepancy (Absolute probability gap & anomaly detection)
   ↓
5. Decision Gate Arbitration: Synthesize checks into PASS, FAIL, or REVIEW_REQUIRED
   ↓
6. Diagnostic Extraction: Map report into FeedbackDiagnostic with recommended actions
   ↓
7. Output: ValidationReport + FeedbackDiagnostic (attached to optimizer output)
```

---

## 3. Customer-Agent Behavioural Model

Agents are not monolithic static personas. Each agent possesses continuous, independent traits and dynamic situational states generated from probability distributions.

### A. Stable Traits (`AgentTraits`)
All traits are strictly positive floats (`> 0.0`):
- `price_sensitivity`: Elasticity and disutility of price premiums ($\sim \text{Lognormal}(\mu=0.0, \sigma=0.35)$, clipped to $[0.1, 4.0]$).
- `value_orientation`: Weight placed on perceived quality and surplus ($\sim \text{Normal}(\mu=1.0, \sigma=0.30)$, clipped to $[0.1, 3.0]$).
- `risk_aversion`: Reluctance to commit under uncertain or premium conditions ($\sim \text{Normal}(\mu=1.0, \sigma=0.35)$, clipped to $[0.1, 3.0]$).
- `patience`: Propensity to postpone purchases in anticipation of future deals ($\sim \text{Normal}(\mu=1.0, \sigma=0.35)$, clipped to $[0.1, 3.0]$).
- `loyalty`: Brand affinity mitigating competitor price discounts ($\sim \text{Normal}(\mu=1.0, \sigma=0.35)$, clipped to $[0.1, 2.5]$).
- `decision_noise`: Stochastic choice temperature $\tau$ ($\sim \text{Normal}(\mu=0.40, \sigma=0.15)$, clipped to $[0.05, 1.5]$).

### B. Dynamic State (`AgentState`)
- `budget_ratio`: Total customer budget relative to the reference base price ($B = \text{base\_price} \times \text{budget\_ratio}$, where $\text{budget\_ratio} \sim \text{Lognormal}(\mu=0.15, \sigma=0.30)$, clipped to $[0.3, 3.5]$).
- `need_intensity`: Current urgency and requirement for the good ($\sim \text{Normal}(\mu=1.0, \sigma=0.30)$, clipped to $[0.1, 2.5]$).
- `reference_price_bias`: Internal anchor deviation ($\sim \text{Normal}(\mu=0.0, \sigma=0.12)$, clipped to $[-0.4, 0.4]$, constrained to $> -1.0$).
- `memory`: Extensible dictionary slot for historical interactions or prior quotes.

### C. Decision Policy Formulation (`StandardDecisionPolicy`)

For a given `Offer`, each agent evaluates choice probabilities through an interpretable multinomial logit framework:

1. **Effective Anchor Price**:
   $$R_{\text{eff}} = \text{base\_price} \times (1.0 + \text{state.reference\_price\_bias})$$
2. **Normalized Price Ratio**:
   $$r = \frac{\text{price}}{\max(R_{\text{eff}}, 0.01)}$$
3. **Perceived Value Surplus**:
   $$\text{surplus} = \frac{\text{perceived\_value} - \text{price}}{\max(R_{\text{eff}}, 0.01)}$$
   $$U_{\text{surplus}} = \text{traits.value\_orientation} \times \text{surplus} \times 2.20$$
4. **Price Disutility (with Loss Aversion for $r > 1.0$)**:
   $$U_{\text{price}} = \begin{cases} - \text{traits.price\_sensitivity} \times (r - 1.0) \times 2.50 & \text{if } r \le 1.0 \\ - \text{traits.price\_sensitivity} \times (r - 1.0) \times 2.50 \times 1.60 & \text{if } r > 1.0 \end{cases}$$
5. **Budget Penalty**:
   $$U_{\text{budget}} = \begin{cases} - (1.0 + \text{traits.risk\_aversion}) \times \frac{\text{price} - B}{\max(B, 0.01)} \times 6.00 & \text{if } \text{price} > B \\ 0.0 & \text{if } \text{price} \le B \end{cases}$$
6. **Competitor Contrast & Loyalty Shielding**:
   If `competitor_price` is specified and $> 0$:
   $$\Delta_{\text{comp}} = \frac{\text{competitor\_price} - \text{price}}{\max(R_{\text{eff}}, 0.01)}$$
   $$U_{\text{comp}} = \begin{cases} (1.0 + 0.5 \times \text{traits.loyalty}) \times \Delta_{\text{comp}} \times 1.80 & \text{if } \Delta_{\text{comp}} \ge 0 \text{ (our price is cheaper)} \\ - \max(0.2, 2.0 - \text{traits.loyalty}) \times |\Delta_{\text{comp}}| \times 1.80 & \text{if } \Delta_{\text{comp}} < 0 \text{ (competitor is cheaper)} \end{cases}$$
   *(If `competitor_price` is `None`, $U_{\text{comp}} = 0.0$. Higher loyalty reduces the penalty from 2.0 down to 0.2).*
7. **Urgency, Scarcity & Need**:
   $$U_{\text{urgency}} = (\text{offer.urgency} \times \text{state.need\_intensity} \times 1.50 + \text{offer.scarcity} \times 1.0) \times 1.40$$
8. **Risk Disutility on Price Premium**:
   $$U_{\text{risk}} = \begin{cases} - 0.5 \times \text{traits.risk\_aversion} \times (r - 1.0) & \text{if } r > 1.0 \\ 0.0 & \text{if } r \le 1.0 \end{cases}$$
9. **Total Latent Buy Utility**:
   $$U_{\text{buy}} = 0.20 + U_{\text{surplus}} + U_{\text{price}} + U_{\text{budget}} + U_{\text{comp}} + U_{\text{urgency}} + U_{\text{risk}}$$
10. **Latent Defer Utility**:
    $$U_{\text{defer}} = -0.50 + \text{traits.patience} \times 1.20 - (\text{offer.urgency} \times 2.0 + \text{offer.scarcity} \times 1.5) - 0.5 \times \text{state.need\_intensity} - 0.5 \times |r - 1.0|$$
11. **Multinomial Logit Choice Probabilities**:
    With temperature $\tau = \max(0.05, \text{traits.decision\_noise})$ and baseline reject utility $U_{\text{reject}} = 0$:
    $$D = 1.0 + \exp\left(\frac{U_{\text{buy}}}{\tau}\right) + \exp\left(\frac{U_{\text{defer}}}{\tau}\right)$$
    $$P(\text{BUY}) = \frac{\exp(U_{\text{buy}} / \tau)}{D}, \quad P(\text{DEFER}) = \frac{\exp(U_{\text{defer}} / \tau)}{D}, \quad P(\text{REJECT}) = \frac{1.0}{D}$$
12. **Decision Sampling**:
    Discrete agent choices (`BUY`, `DEFER`, `REJECT`) are sampled from $[P(\text{BUY}), P(\text{DEFER}), P(\text{REJECT})]$ using `rng.choice(...)`.

---

## 4. Population Generation and Reproducibility

### Distribution Parameters

`PopulationGenerator` (`customer_agent_sim/population.py`) samples agents independently:

```python
from customer_agent_sim import PopulationGenerator

generator = PopulationGenerator()
population = generator.generate(size=1000, seed=42)
```

Each agent receives traits and states sampled via `np.random.default_rng(seed)`:
- `price_sensitivity`: `rng.lognormal(mean=0.0, sigma=0.35)` clipped to $[0.1, 4.0]$
- `value_orientation`: `rng.normal(1.0, 0.30)` clipped to $[0.1, 3.0]$
- `risk_aversion`: `rng.normal(1.0, 0.35)` clipped to $[0.1, 3.0]$
- `patience`: `rng.normal(1.0, 0.35)` clipped to $[0.1, 3.0]$
- `loyalty`: `rng.normal(1.0, 0.35)` clipped to $[0.1, 2.5]$
- `decision_noise`: `rng.normal(0.40, 0.15)` clipped to $[0.05, 1.5]$
- `budget_ratio`: `rng.lognormal(mean=0.15, sigma=0.30)` clipped to $[0.3, 3.5]$
- `need_intensity`: `rng.normal(1.0, 0.30)` clipped to $[0.1, 2.5]$
- `reference_price_bias`: `rng.normal(0.0, 0.12)` clipped to $[-0.4, 0.4]$

### Deterministic Reproducibility
Passing the same `seed` guarantees identical population generation and simulation metrics. Different seeds produce distinct populations with measurable behavioral divergence.

### Common Random Numbers (CRN)
When comparing multiple candidate prices or scenario variations, `runner.py` uses **Common Random Numbers (CRN)**:
- The exact same population of `CustomerAgent` objects is evaluated across all candidate prices.
- The same random seed is passed to each candidate's simulation.
- This isolates the effect of price changes from Monte Carlo sampling noise, drastically reducing variance in candidate rank ordering.

### Shared Population Reuse for Batch Performance
Generating 1,000 agents takes approximately 80–120ms. In high-throughput quoting, callers can generate a population once and pass it across multiple validations via `population=shared_population`, reducing per-quote evaluation time to under 15ms.

---

## 5. The Five Validation Checks

### Check A: Behavioural Consistency (`validate_behavioural_consistency`)
Verifies basic economic axioms on the offer:
1. **Price Ladder Monotonicity**: Evaluates the offer across relative multipliers $[0.8, 0.9, 1.0, 1.1, 1.2] \times \text{price}$. Verifies that simulated conversion does not systematically increase as price increases:
   $$\text{conv}_{i+1} \le \text{conv}_i + 0.02 \quad (\text{tolerance allowance for noise})$$
2. **Value Sensitivity**: Evaluates offer with $1.2 \times \text{perceived\_value}$. Verifies conversion increases or holds:
   $$\text{conv}_{\text{boosted}} \ge \text{conv}_{\text{base}} - 0.02$$
3. **Competitor Sensitivity**: If `competitor_price` is present, evaluates offer with competitor discount ($0.8 \times \text{competitor\_price}$) and asserts conversion responds plausibly.
4. **Urgency & Scarcity Sensitivity**: Evaluates offer with $\text{urgency}=0.85$ and $\text{scarcity}=0.85$, checking for non-negative conversion response.

*Failure Condition*: Failure of price ladder monotonicity or value sensitivity triggers a `FAIL` status.

### Check B: Population Heterogeneity (`validate_population_heterogeneity`)
Ensures customer traits produce meaningful variance rather than a cloned cohort:
1. Computes population choice probability variance:
   $$\text{Var}(P_{\text{buy}}) = \frac{1}{N - 1} \sum_{i=1}^N (P_{\text{buy}, i} - \bar{P}_{\text{buy}})^2 \ge \text{min\_heterogeneity\_variance} \quad (\text{default: } 0.002)$$
2. Partitions population into subgroups via median trait split on `price_sensitivity` (at 0.50 quantile); measures conversion difference $\Delta = \text{conv}_{\text{low\_ps}} - \text{conv}_{\text{high\_ps}}$.

*Failure Condition*: $\text{Var}(P_{\text{buy}}) < 0.0005$ triggers `FAIL` (cloned population). $0.0005 \le \text{Var} < 0.002$ triggers a warning.

### Check C: Scenario Robustness (`validate_scenario_robustness`)
Re-evaluates the recommended price under market shifts using the **exact same customer population**:
- `budget_squeeze`: Customer budgets reduced by 25% (`budget_factor = 0.75`).
- `competitor_undercut`: Competitor lowers price by 20% (`competitor_factor = 0.80`).
- `high_urgency`: Urgency raised to 0.90, need intensity $+25\%$.
- `perceived_value_drop`: Perceived value reduced by 20% (`perceived_value_factor = 0.80`).

*Vulnerability Criteria*:
- Conversion collapses below `minimum_simulated_conversion` (default 5%).
- Conversion drops by $>65\%$ relative to baseline ($\text{conv} < 0.35 \times \text{conv}_{\text{base}}$ when $\text{conv}_{\text{base}} > 0.10$).

*Failure Condition*: Scenario vulnerability triggers `REVIEW_REQUIRED`.

### Check D: Ranking Stability (`validate_ranking_stability`)
Re-evaluates all candidate prices evaluated by the ML optimizer using Common Random Numbers (CRN):
1. **Explicit Ranking Objective**:
   - If `unit_cost` is available: ranks candidates by simulated expected unit profit:
     $$E[\text{Profit}] = (\text{price} - \text{unit\_cost}) \times \text{sim\_conversion}$$
   - If `unit_cost` is unavailable (`None`): ranks candidates by simulated expected revenue:
     $$E[\text{Revenue}] = \text{price} \times \text{sim\_conversion}$$
   - Evaluates matching profit or revenue objectives for ML probabilities.
2. **Dual Stability Criteria**:
   - `rank_drop`: Evaluates position drop ($\text{sim\_rank} - \text{ml\_rank} \le \text{tolerance\_rank\_drop}$, default `2`, and $\text{sim\_rank} \le 3$).
   - `metric_gap`: Evaluates relative loss against simulation's top candidate:
     $$\text{metric\_gap} = \frac{\text{best\_sim\_metric} - \text{selected\_sim\_metric}}{\max(\text{best\_sim\_metric}, \epsilon)} \le \text{tolerance\_profit\_gap} \quad (\text{default: } 0.12)$$

*Failure Condition*: Exceeding either rank drop or metric gap tolerance triggers `REVIEW_REQUIRED`. Single-price mode marks ranking `available = False` without failing.

### Check E: ML Discrepancy & Anomaly Detection (`validate_ml_discrepancy`)
Quantifies discrepancy between ML probability and simulated purchase rate:
$$\text{gap} = |p_{\text{ML}} - \text{conv}_{\text{sim}}|$$
- `gap_exceeded`: $\text{gap} > \text{max\_allowed\_probability\_gap}$ (default: `0.15` / 15 pp).
- `severe_discrepancy`: $\text{gap} > 2 \times \text{max\_allowed\_probability\_gap}$ (default: `0.30` / 30 pp).
- Anomaly detection catches $p_{\text{ML}} \notin [0, 1]$, non-finite numbers, or zero-purchase collapses.

*Failure Condition*: Severe discrepancy ($>30$ pp) triggers `FAIL`. Moderate discrepancy ($15\text{ pp} < \text{gap} \le 30\text{ pp}$) triggers `REVIEW_REQUIRED`.

---

## 6. Edge Cases and Failure Handling

| Edge Case | Implementation Behaviour |
| :--- | :--- |
| **Missing Competitor Price** (`competitor_price = None`) | Competitor price is preserved as `None` without inventing synthetic prices. In `scenarios.py`, `competitor_undercut` scenario is marked `applicable = False` with diagnostic `"Scenario 'competitor_undercut' requires competitor_price, but offer has no competitor benchmark."` In `StandardDecisionPolicy`, competitor utility $U_{\text{comp}}$ evaluates to `0.0`. |
| **Missing Unit Cost** (`unit_cost = None`) | Candidate evaluation automatically falls back to expected revenue: $E[\text{Revenue}] = \text{price} \times \text{sim\_conversion}$. Ranking metric is labeled `"revenue"`. Does not crash with `TypeError`. |
| **Single Candidate Price** (`candidate_prices = None` or `len <= 1`) | Single-price mode: `validate_ranking_stability` returns `{"available": False, "reason": "Single-price validation: no candidate alternatives provided."}` Validation proceeds smoothly through the other four checks without failing. |
| **Invalid / Non-Finite Inputs** | `Offer` raises `ValueError` on non-positive prices, non-positive base price, non-positive perceived value, out-of-range urgency/scarcity, or negative unit cost. `ml_probability` must be in $[0.0, 1.0]$. In shadow mode, `run_shadow_validation` catches these exceptions and returns status `"ERROR"`. |
| **Degenerate / Cloned Population** | If population choice variance $\text{Var}(P_{\text{buy}}) < 0.0005$, `validate_population_heterogeneity` flags variance collapse, triggering `FAIL`. |
| **Confidence-Interval Boundaries** | Uses Wilson score interval with continuity properties. If $\text{ci\_upper} < \text{minimum\_simulated\_conversion}$, conversion is definitively below safety threshold $\to$ `FAIL`. If point estimate $\ge \text{threshold}$ but $\text{ci\_lower} < \text{threshold}$, conversion is borderline $\to$ `REVIEW_REQUIRED`. |
| **Ranking Ties & Negative Margins** | When all candidate profits are negative or zero, `metric_gap` handles division by zero safely using $\max(|\text{best\_metric}|, \epsilon)$. Ties in candidate metrics are broken deterministically by price proximity to base price, then lower price. |
| **Simulator Exceptions** | Wrapped in `try...except Exception as exc:` inside `run_shadow_validation`. Catches any unexpected Monte Carlo or runtime error, logs via `logger.error`, and returns a safe dictionary with `status: "ERROR"` and `error: str(exc)`. Pricing and checkout eligibility are never blocked. |

---

## 7. Decision Statuses and Feedback

### Outcome Matrix

```
                          ┌─────────────────────────────┐
                          │   All checks passed?        │
                          └──────────────┬──────────────┘
                                         │
                         Yes ┌───────────┴───────────┐ No
                             ▼                       ▼
                         ┌───────┐           Definitive Failure?
                         │ PASS  │         (CI upper < threshold,
                         └───────┘         Severe gap > 30 pp,
                                            Inverted demand)
                                                     │
                                         Yes ┌───────┴───────┐ No
                                             ▼               ▼
                                         ┌───────┐   ┌─────────────────┐
                                         │ FAIL  │   │ REVIEW_REQUIRED │
                                         └───────┘   └─────────────────┘
```

| Status | Exact Triggering Conditions |
| :--- | :--- |
| **`PASS`** | • Simulated conversion $\ge \text{minimum\_simulated\_conversion}$<br>• Probability gap $\le \text{max\_allowed\_probability\_gap}$<br>• Monotonicity & value sensitivity hold<br>• Population heterogeneity $\ge 0.002$<br>• Scenario robustness acceptable (no collapse or $>65\%$ drop)<br>• Candidate ranking stable |
| **`FAIL`** | • Simulated conversion is definitively below minimum ($\text{ci\_upper} < \text{minimum\_simulated\_conversion}$)<br>• Severe ML discrepancy ($\text{gap} > 2 \times \text{max\_allowed\_probability\_gap}$)<br>• Severe demand curve inversion (higher price converts substantially better)<br>• Cloned population ($\text{variance} < 0.0005$)<br>• Invalid inputs (negative prices, NaNs) |
| **`REVIEW_REQUIRED`** | • Moderate probability gap ($1\times \text{gap} < \text{gap} \le 2\times \text{gap}$)<br>• Borderline conversion ($\text{ci\_lower} < \text{minimum} \le \text{ci\_upper}$)<br>• Scenario vulnerability ($>65\%$ collapse in stress tests)<br>• Ranking instability (excessive rank drop or $\text{metric\_gap} > 12\%$) |
| **`ERROR`** | Internal simulator exception caught during execution. Diagnostic error string populated, pricing unaffected. |

### Structured Feedback Diagnostics (`FeedbackDiagnostic`)

`extract_feedback_diagnostics(report)` converts validation findings into structured flags:

```python
{
    "status": "PASS",                         # "PASS", "FAIL", "REVIEW_REQUIRED", or "ERROR"
    "conversion_threshold_failed": False,      # True if simulated conversion < minimum
    "discrepancy_exceeded": False,            # True if |p_ML - conv_sim| > tolerance
    "scenario_robustness_failed": False,      # True if stress scenarios collapsed conversion
    "ranking_unstable": False,                # True if candidate rank drop or metric gap exceeded
    "suggested_action": "PROCEED",            # "PROCEED", "EVALUATE_ALTERNATIVE_CANDIDATE",
                                              # "HUMAN_PRICING_REVIEW", "FALLBACK_TO_BASE_PRICE"
    "alternative_candidate_price": None,      # Best simulated price if ranking was unstable
    "retry_count": 0,                         # Iteration count in feedback loop
    "reason": "All configured validation checks passed successfully.",
    "recommended_steps": ["Price recommendation validated for downstream deployment."]
}
```

### Bounded Retries & Shadow-Mode Invariance
In standalone mode, `bounded_validation_pipeline(validation_fn, candidate_prices, max_retries=3)` provides an iterative candidate search. It stops when a candidate achieves `PASS` or after `max_retries` attempts (returning `REVIEW_REQUIRED` with action `"HUMAN_PRICING_REVIEW"`).

**Critical Shadow-Mode Guarantee**: In Monetize360 integration, the feedback loop is strictly advisory. It **never automatically modifies the optimizer's selected price** or changes validation thresholds. The price chosen by the optimizer is always returned.

---

## 8. Configuration Reference

All settings can be configured via `ValidationConfig` (`customer_agent_sim/models.py`) or JSON (`customer_agent_sim/config/default_config.json`).

| Setting | Type | Default | Enforced Range | Operational Effect |
| :--- | :--- | :--- | :--- | :--- |
| `population_size` | `int` | `1000` | $\ge 1$ | Number of simulated customer agents generated for Monte Carlo sampling. |
| `seed` | `int` | `42` | Any `int` | Random seed for deterministic reproducibility across populations and decisions. |
| `minimum_simulated_conversion` | `float` | `0.05` | $[0.0, 1.0]$ | Minimum acceptable simulated conversion rate (5%). Triggers `FAIL` if upper 95% CI is below this threshold. |
| `max_allowed_probability_gap` | `float` | `0.15` | $[0.0, 1.0]$ | Maximum allowed absolute percentage-point difference between ML probability and simulated conversion (15 pp). |
| `confidence_level` | `float` | `0.95` | $(0.0, 1.0)$ | Confidence level for Wilson score interval calculations. |
| `ranking_tolerance_rank_drop` | `int` | `2` | $\ge 0$ | Maximum allowable drop in simulation rank relative to ML rank before flagging ranking instability. |
| `ranking_tolerance_profit_gap` | `float` | `0.12` | $[0.0, 1.0]$ | Maximum allowable relative profit/revenue loss (12%) against simulation's top candidate before flagging instability. |
| `min_heterogeneity_variance` | `float` | `0.002` | $\ge 0.0$ | Minimum required variance in individual agent purchase probabilities across the population. |
| `max_retry_attempts` | `int` | `3` | $\ge 1$ | Maximum number of candidate re-evaluation iterations in bounded validation loops. |
| `scenarios` | `List[str]` | `[...]` | Valid keys in `STANDARD_SCENARIOS` | List of stress scenarios to evaluate (default: `budget_squeeze`, `competitor_undercut`, `high_urgency`, `perceived_value_drop`). |

---

## 9. Standalone Public API with Accurate Examples

### Example 1: Full Candidate Grid Validation

```python
from customer_agent_sim import (
    Offer,
    ValidationConfig,
    validate_offer,
    extract_feedback_diagnostics,
)

# 1. Define domain-neutral Offer
offer = Offer(
    price=105.00,             # ML Optimizer selected price
    base_price=100.00,        # Catalog reference price
    perceived_value=120.00,   # Customer perceived utility
    competitor_price=108.00,  # Competitor price (optional)
    urgency=0.35,             # Urgency signal in [0.0, 1.0]
    scarcity=0.25,            # Scarcity signal in [0.0, 1.0]
    unit_cost=60.00,          # Unit cost (optional)
    context={"channel": "direct", "region": "metro_west"},
)

# 2. Candidate grid and ML model predictions
candidate_prices = [85.00, 95.00, 100.00, 105.00, 115.00, 125.00]
ml_probabilities = {
    85.00: 0.82,
    95.00: 0.74,
    100.00: 0.69,
    105.00: 0.65,
    115.00: 0.52,
    125.00: 0.38,
}

# 3. Configure validation gate
config = ValidationConfig(
    population_size=1000,
    seed=42,
    minimum_simulated_conversion=0.08,
    max_allowed_probability_gap=0.15,
)

# 4. Execute validation
report = validate_offer(
    offer=offer,
    ml_probability=0.65,
    candidate_prices=candidate_prices,
    ml_probabilities=ml_probabilities,
    config=config,
)

# 5. Inspect report and structured diagnostics
print("Gate Status:", report.status.value)                 # PASS
print("Simulated Conversion:", f"{report.simulated_conversion:.2%}")
print(f"95% CI: [{report.ci_lower:.2%} - {report.ci_upper:.2%}]")
print(report.summary_text())

feedback = extract_feedback_diagnostics(report)
print("Action:", feedback.suggested_action)               # PROCEED
```

### Example 2: Single-Price Mode (Missing Competitor & Missing Grid)

```python
from customer_agent_sim import Offer, validate_offer

# Evaluates an offer without competitor benchmark or candidate alternatives
offer = Offer(
    price=99.00,
    base_price=100.00,
    perceived_value=100.00,
    competitor_price=None,  # Competitor scenarios marked inapplicable
    unit_cost=None,         # Ranking falls back to revenue
)

report = validate_offer(
    offer=offer,
    ml_probability=0.72,
    candidate_prices=None,  # Ranking stability marked unavailable
)

assert report.status in ("PASS", "REVIEW_REQUIRED", "FAIL")
assert report.ranking_results["available"] is False
assert report.scenario_results["scenarios"]["competitor_undercut"]["applicable"] is False
```

### Example 3: Population Reuse for High-Throughput Batch Processing

```python
from customer_agent_sim import PopulationGenerator, validate_offer

# Pre-generate 1,000 agents once
shared_population = PopulationGenerator().generate(size=1000, seed=42)

for offer, ml_prob in batch_quotes:
    report = validate_offer(
        offer=offer,
        ml_probability=ml_prob,
        population=shared_population,  # Reuses existing agents, avoiding regeneration
    )
```

---

## 10. Monetize360 Integration Contract

### Optimizer Integration Hooks
Shadow validation is integrated into Monetize360's constrained optimizer hierarchy:
- [`BaseConstrainedOptimizer.run_shadow_validation()`](file:///d:/HACKFEST/monetize360-datasets/ml/base_optimizer.py) (Shared validator)
- [`ConstrainedOptimizer.optimize_quote()`](file:///d:/HACKFEST/monetize360-datasets/ml/optimizer.py) (E-Commerce)
- [`HotelConstrainedOptimizer.optimize_quote()`](file:///d:/HACKFEST/monetize360-datasets/ml/hotel_optimizer.py) (Hospitality)
- [`BankingConstrainedOptimizer.optimize_quote()`](file:///d:/HACKFEST/monetize360-datasets/ml/banking_optimizer.py) (Banking / Lending)

### Explicit Input Mapping Contract

| `Offer` Field | Source Field (E-Commerce) | Source Field (Hospitality) | Source Field (Banking) | Missing Value Handling |
| :--- | :--- | :--- | :--- | :--- |
| `price` | `optimal_price` | `optimal_price` | `optimal_price` / `optimal_rate` | Required (selected by optimizer) |
| `base_price` | `base_price` | `base_price` | `base_rate` | Required (> 0) |
| `perceived_value` | `perceived_value` or context | `perceived_value` or context | `perceived_value` or context | Falls back to catalog anchor `base_price` |
| `competitor_price` | `competitor_price` / `competitor_min_price` | `competitor_price` | `competitor_rate` | **Explicitly `None`** (no synthetic pricing invented; marks competitor scenarios `inapplicable`) |
| `unit_cost` | `unit_cost` | `unit_cost` (operating cost) | `funding_cost` | **`None`** if absent (ranks candidates by expected revenue) |
| `urgency` | `urgency` / `demand_index` | `urgency` | `urgency` | Default `0.0` (clamped to $[0.0, 1.0]$) |
| `scarcity` | `scarcity` / `sell_through` | `scarcity` / occupancy | `scarcity` | Default `0.0` (clamped to $[0.0, 1.0]$) |
| `context` | Metadata dict | Metadata dict | Metadata dict | Default `{}` |

### Activating Shadow Validation

```python
from ml.optimizer import ConstrainedOptimizer

opt = ConstrainedOptimizer(
    preprocessor=preprocessor,
    model=gbt_model,
    enable_shadow_validation=True,      # Activates observational validation
    validation_population_size=500,     # Configurable population size
    validation_seed=42,                 # Configurable seed
)

result = opt.optimize_quote(
    candidate_prices=candidate_prices,
    unit_cost=unit_cost,
    base_price=base_price,
    min_price=min_price,
    max_price=max_price,
    model_probs=model_probs,
    inventory_on_hand=inventory,
    competitor_price=competitor_price,
    perceived_value=perceived_value,
)
```

### Optimizer Response Contract

The returned dictionary preserves all existing keys and appends `shadow_validation`:

```python
{
    # Standard optimizer outputs (strictly untouched)
    "optimal_price": 105.0,
    "expected_profit": 29.25,
    "optimal_index": 3,
    "is_price_feasible": True,
    "n_feasible_candidates": 6,
    "order_eligible": True,
    "can_checkout": True,
    "eligibility_reason": "in_stock",
    "inventory_on_hand": 10,
    "requested_quantity": 1,

    # Shadow validation diagnostic payload
    "shadow_validation": {
        "status": "PASS",               # "PASS", "FAIL", "REVIEW_REQUIRED", or "ERROR"
        "simulated_conversion": 0.683,  # Point estimate
        "probability_gap": 0.033,       # Absolute difference |p_ML - conv_sim|
        "confidence_interval": [0.6535, 0.7111], # 95% Wilson CI
        "report": { ... },              # Full ValidationReport dictionary
        "feedback": {                   # extract_feedback_diagnostics dictionary
            "status": "PASS",
            "conversion_threshold_failed": False,
            "discrepancy_exceeded": False,
            "scenario_robustness_failed": False,
            "ranking_unstable": False,
            "suggested_action": "PROCEED",
            "alternative_candidate_price": None,
            "retry_count": 0,
            "reason": "All configured validation checks passed successfully.",
            "recommended_steps": ["Price recommendation validated for downstream deployment."]
        },
        "error": None
    }
}
```

If a simulator error occurs:
```python
"shadow_validation": {
    "status": "ERROR",
    "simulated_conversion": None,
    "probability_gap": None,
    "confidence_interval": None,
    "report": None,
    "feedback": None,
    "error": "Error description string"
}
```

---

## 11. Testing and Demo

### Run Simulator Unit Tests (32 tests)
Verifies agent traits, decision policies, population diversity, scenarios, CRN ranking stability, Wilson intervals, and bounded feedback loops:
```powershell
python -m pytest customer_agent_sim/tests -v
```
*Actual result:* `32 passed in 3.42s`

### Run Shadow Validation Integration Tests (10 tests)
Verifies observational invariance, PASS, FAIL, REVIEW_REQUIRED, missing optional fields, revenue ranking fallback, simulator exception containment, population reuse, and multi-domain execution:
```powershell
python -m pytest tests/test_shadow_validation.py -v
```
*Actual result:* `10 passed in 4.42s`

### Run Complete Repository Test Suite (39 tests)
```powershell
python -m pytest tests -v
```
*Actual result:* `39 passed in 4.54s`

### Run Golden Pricing Engine Tests (32 golden cases)
Verifies that no changes were made to pricing logic, rules, or policies:
```powershell
python tests/test_golden.py
```
*Actual result:* `Results: 32 passed, 0 failed out of 32 cases. [OK]`

### Run Live Interactive Demo
```powershell
python customer_agent_sim/examples/run_demo.py
```
*Executes live validation across PASS, single-price mode, and overpriced candidate failure diagnostics.*

---

## 12. Limitations and Production Calibration

1. **Synthetic Sanity Check, Not Empirical Truth**:
   The customer agents are programmatic Monte Carlo simulators parameterized by utility functions. They provide an algorithmic safety gate to catch price-ladder inversions, catastrophic extrapolations, and severe sensitivity bugs. They **do not replace empirical demand testing** and should not be treated as real-world market proof.
2. **Epistemic vs. Sampling Uncertainty**:
   The 95% Wilson score interval measures statistical sampling uncertainty given the simulated model and sample size. It does not measure whether the structural model itself matches real-world customer psychology.
3. **Discrepancy Diagnostics**:
   A `FAIL` or `REVIEW_REQUIRED` discrepancy between the ML prediction and simulation highlights conflicting behavioral assumptions (e.g. an ML model extrapolating high conversion into a severe budget cliff). It indicates the recommendation needs review, not necessarily that the ML model is wrong.
4. **Production Parameter Calibration**:
   Before transitioning from shadow mode to active gating, behavioral coefficients (`price_sensitivity_weight`, `loss_aversion_factor`, `budget_penalty_weight`, `loyalty`) should be calibrated against empirical transaction telemetry, price elasticity studies, or holdout A/B testing data.
5. **Computational Latency**:
   Simulating 1,000 agents requires approximately 80–120ms per candidate. For latency-sensitive synchronous workflows, set `validation_population_size=250` or pass a pre-generated population (`population=shared_population`) to maintain sub-15ms latency.
