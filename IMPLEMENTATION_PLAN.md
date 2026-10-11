# Monetize360 — Implementation Plan v3 (Python / FastAPI)

_Revised 2026-10-11. Backend = **Python + FastAPI**. LLM = **Gemini or Grok** (keys supplied later; mock until then). Optimizer = **existing ML_Updated GBT model** (in-process) with a simulated fallback. Persona validation = **existing `customer_agent_sim`** (shadow mode) + named persona cohorts._
_Sources: "MONETIZE360 — MASTER BUILD PROMPT", the "ML_Updated" README, and the "customer_agent_sim" README. Do the phases **in order**. Each phase ends with acceptance checks that must pass before you start the next one._

**Status (2026-10-11):** ✅ All phases (1–8) implemented and verified — see `PROGRESS.md` for the verification table and `README.md` to run it. Only real Gemini/Grok calls remain unverified (no API keys yet). API port is 8010.

**Pipeline:** Business Portal → LLM Strategy Generator → validated config + approval → Universal Pricing Engine (rules, candidates, hard constraints) → ML Optimizer (P(buy) → E[profit] → select) → engine re-verify → **Customer-Agent Validation (shadow: PASS / FAIL / REVIEW_REQUIRED / ERROR + persona cohorts)** → Pricing API + Explanation → User Portal.

---

## 0. Audit summary

| Area | State | Decision |
|---|---|---|
| Frontend (React 18, Vite 5, TS strict, Tailwind, hand-written shadcn UI, Router, Zustand, Recharts) | ✅ builds | **Keep**. It becomes a pure client of the FastAPI backend. |
| `src/engine` (~1,900 lines TS, 48 passing vitest tests) | ✅ | **Port to Python** (`backend/app/engine`). This is the reference implementation, and its tests show the behaviour to reproduce. After Phase 7, delete everything in `src/engine` except `format.ts`. |
| `src/api` + `src/store/db.ts` | mock API over localStorage | Rewrite as `fetch` calls to `/api/v1`. Delete the mock. |
| Backend / DB | ❌ none | **Add** `backend/` (FastAPI + SQLAlchemy 2 + Alembic + SQLite + Pydantic v2 + pytest). |
| ML model (`ml_updated/`, GBT, 14 ratio features) | ❌ **not on this machine** (searched `C:\Users\97150` to depth 4) | Copy it into `Monetize360/ml_updated/` **read-only**. Load it in-process behind the `PriceOptimizer` interface. Never retrain it or overwrite its files. |
| Customer-agent simulator (`customer_agent_sim/`, numpy Monte Carlo, 5 checks, 32 tests) | ❌ **not on this machine** | Copy it into `Monetize360/customer_agent_sim/` **read-only**. Call its public API (`validate_offer`, `PopulationGenerator`, `extract_feedback_diagnostics`) from `backend/app/validation/`. Do **not** use its hooks into the domain-specific `ml/hotel_optimizer.py` / `banking_optimizer.py`. |
| LLM | mock only (`policyParser.ts`, `domainSuggest.ts`) | `GeminiProvider` + `GrokProvider` behind one interface. Port the mock parser as the labeled fallback. |
| Screens | 1/16 built | Consolidate to 7 business screens + 1 user portal. |
| Python | 3.13.7, scikit-learn 1.6.0 installed | Pin scikit-learn to **the version recorded in `ml_updated/artifacts/training_metadata.json`**. Pickles are version-sensitive. |

---

## 1. Ideation review — what's sound, what must be adjusted

### Sound (keep as designed)
- The ML model takes **ratios and 0..1 scores only**, so it is domain-agnostic in the same sense as the engine. It slots cleanly between "candidate generation" and "final decision".
- `E[profit] = P(buy) × (price − cost)`, argmax among guarded candidates. This matches the brief exactly.
- Separation: ML predicts behaviour, the optimizer decides on profit, and guardrails reject. This maps 1:1 onto engine → optimizer → re-verify.
- Monotonicity test, a LR baseline, and an edge-case suite (below-cost trap, total disqualification). These are good demo material.

### Issues to handle in the integration (not by retraining)
| # | Issue | Handling in this build |
|---|---|---|
| 1 | **Trained on synthetic data** produced by `generate_dataset.py` from `ground_truth_coefficients.json`. The model learned that simulator, not real customers. ROC-AUC 0.81 measures how well it fits the simulator. | Label every P(buy) as **"model trained on synthetic data"** (`provenance.trainingData="synthetic"`). Never call it real conversion. |
| 2 | **`price_rank_normalized` depends on which other candidates exist**. The same price can get a different P(buy) when the candidate set changes. | Compute the rank over the **full generated candidate set before eligibility filtering** (that matches training: 5 candidates per quote). Keep the candidate count at 5. Document it as a known model artifact and a candidate to drop in a future retrain. |
| 3 | **`perceived_value_gap` needs `perceived_value`**, which the engine has no source for. | Add an optional semantic signal `perceivedValue`. Default when unmapped = base price, so the gap = `(base − price)/base`. Flag the default in the decision warnings. |
| 4 | **`margin_ratio` feeds business cost into a customer-behaviour model** (a customer can't see cost). | Supply it as trained (required feature). Note it in the docs. Don't build anything else on it. |
| 5 | **Out-of-distribution inputs**: price_ratio was trained on 0.6–1.5, demand_pressure 0.2–2.0. | `features.py` adapter checks the ranges. If a value is outside, keep the prediction but add the warning `OOD_FEATURE:<name>` and show it in the UI. Strategy validation warns when `maxIncrease`/`maxDiscount` let ratios leave 0.6–1.5. |
| 6 | **Persona inputs (sensitivity, loyalty) are per-customer**, but quotes can be anonymous. | Domain config maps them from customer-profile attributes. If absent, use the domain default (0.5) plus a warning. In the Simulator, the persona panel can re-run the model **per persona** (sensitivity/loyalty from the persona). This is the clean link between "Persona Agent Simulation" and the ML model. |
| 7 | **Duplicate guardrails**: ML has its own (cost, ceiling, 10% conversion floor). | The engine's hard constraints stay authoritative. The ML cost/ceiling checks run as a second line. The 10% floor becomes `optimizerPolicy.conversionFloor` (default 0.10, configurable per strategy). The engine re-verifies the final pick either way. |
| 8 | GBT AUC (0.807) ≈ LR (0.808). GBT has higher accuracy. | Use GBT as primary (as designed). Expose `OPTIMIZER_MODEL=gbt|lr` so LR can be demoed as the baseline. |
| 9 | **Pickle = arbitrary code on load.** | Load only from the local `ML_ARTIFACTS_DIR`. Never accept uploaded pickles. Pin the sklearn version. Verify the artifact SHA-256 against a hash file recorded at first setup. |
| 10 | Brief says "simulate the optimizer". The model now exists, and Python can load it directly. | Keep **both** behind the interface: `MLPriceOptimizer` (default when artifacts are present) and `SimulatedPriceOptimizer` (fallback/tests, uses the **same 14 features** and a transparent logistic). Each response states which was used. |
| 11 | Recursive rule conditions (`all/any/not` trees) don't fit Gemini `response_schema` well (it doesn't handle recursion reliably). | The LLM output schema uses a **flat condition form** (`match: "all"|"any"`, `conditions: Leaf[]`, max one level of `not` per leaf). Server code converts it to the internal tree. The human editor can still build nested trees. |

**Verdict (ML):** the ideation is correct and can be implemented. None of the issues block it. All of them are handled at the adapter or labeling layer without retraining.

### 1b. Customer-agent validation (`customer_agent_sim`) review

**Sound (keep as designed)**
- **Shadow mode is exactly what the brief requires**: it runs after selection, never changes the price, eligibility or objective, and an `ERROR` never blocks a quote. That is the "personas must not decide the price" rule enforced by design.
- No LLM, seeded numpy, Common Random Numbers, Wilson CIs. It is deterministic and statistically honest, and labels itself as a synthetic sanity check.
- Domain-neutral `Offer` (price, base, perceived value, competitor, urgency, scarcity, cost) maps 1:1 onto our semantic signals.
- Handles missing values sensibly: no competitor → competitor scenario not applicable; no cost → ranks by revenue; one candidate → ranking check unavailable.

**Issues to handle in the integration**
| # | Issue | Handling in this build |
|---|---|---|
| P1 | **The brief asks for *named* personas** (budget-conscious, price-sensitive, loyal, urgent, premium, deal-seeker) with a description, concern and suggestion. The module simulates an **anonymous population** and only splits it by price sensitivity. | Add a **persona cohort layer** (`validation/cohorts.py`). Each named persona = a set of **trait/state overrides or distribution shifts** on top of `PopulationGenerator` (e.g. budget-conscious: `budget_ratio × 0.7`; loyal: `loyalty 1.8–2.5`; urgent: `need_intensity 1.8–2.5`; premium: `value_orientation 1.6–2.4`, `budget_ratio × 1.8`; deal-seeker: `patience 1.6–2.5`, `price_sensitivity × 1.6`). Generate each cohort with a fixed seed (`seed + cohort_index`) and run each candidate price on it via the module's runner (same CRN population per cohort). Output per persona × price: BUY/DEFER/REJECT shares, the main concern, a suggestion. If `customer_agent_sim` doesn't expose trait overrides, generate the population, then post-filter/rescale agent traits in **our** code. Never edit the module. |
| P2 | **"Main concern" needs to be explainable.** The module returns rates, not reasons. | Decompose mean utility per cohort into the documented terms (`U_price, U_budget, U_comp, U_surplus, U_urgency, U_risk`, defer pull). The most negative term picks the concern template ("price above my reference", "over my budget", "competitor is cheaper", "not worth it at this price", "I'd rather wait for a deal"). If the module doesn't expose per-term utilities, re-compute them in `cohorts.py` from the README formulas (documented as a mirror; test that the summed utility matches the module's P(BUY) on a few agents). |
| P3 | **Check A (monotonicity) tests the simulator's own price ladder**, which is monotone by construction, so it almost never fails. It says nothing about the ML model. | Add our own **ML ladder check** in `validation/ml_checks.py`: across the candidate set (sorted by price) assert `p_buy[i+1] ≤ p_buy[i] + 0.02`. Report it next to the module's 5 checks as "Check F: ML monotonicity". A violation → overall status at least `REVIEW_REQUIRED`. |
| P4 | **Two different synthetic behaviour models are compared** (ML trained on `generate_dataset.py`'s ground truth, agents use a separate utility). Check E's "ML discrepancy" means the two models disagree, not that either is right. | Label it "**model disagreement**" in the UI. Explanation copy: "Two independent synthetic behaviour models disagree by X pp. Review recommended; this is not evidence about real customers." |
| P5 | **Latency**: ~80–120 ms per evaluation at 1,000 agents. One validation runs ≈ 17 evaluations (6 candidates + 5-step ladder + 3 sensitivity probes + 4 scenarios), so **1.5–2 s**, plus cohorts. Too slow for the synchronous customer quote. | (a) **Shared populations** are generated once at startup per `(seed, size)` and per cohort, then cached in memory. (b) **Business Simulator**: run synchronously, population 1,000. (c) **Customer quote**: return the price immediately and run validation as a FastAPI `BackgroundTask` (population 250), storing it in `shadow_validations` linked to the decision. The decision page shows "validation pending" until it arrives. `VALIDATION_MODE=sync|background|off`. |
| P6 | **Feedback suggests actions** (`EVALUATE_ALTERNATIVE_CANDIDATE`, `FALLBACK_TO_BASE_PRICE`, `alternative_candidate_price`) and has a bounded-retry pipeline. Applying them would break shadow mode. | **Never call `bounded_validation_pipeline` in the pricing path.** Show suggestions as advisory text only. The UI marks an alternative or base-price suggestion as "not eligible" if it breaks a hard constraint (checked by our engine). `REVIEW_REQUIRED/FAIL` adds a business-side flag on the decision and the dashboard ("Decisions needing review"). It never changes `final_price`. Customers never see it. |
| P7 | The integration contract points at **domain-specific optimizers** (`HotelConstrainedOptimizer`, `BankingConstrainedOptimizer` in `d:/HACKFEST/monetize360-datasets/ml/`). That conflicts with domain independence, and it's a different codebase from `ml_updated/`. | Don't use those hooks. Our own `validation/service.py` builds the `Offer` from the **engine decision** (same mapping table as the README, sourced from semantic signals), for every domain. If that `ml/` repo is the one you actually want instead of `ml_updated/`, say so (open question). |
| P8 | Trait scales differ: agent `loyalty` is ~N(1, 0.35) in [0.1, 2.5], while the ML `customer_loyalty_score` is 0..1. | Never pass ML features into agent traits directly. Cohort definitions use the module's own scales. Customer-profile loyalty (0..1) maps to an agent-loyalty override via `0.1 + 2.4 × loyalty` only when "validate for this customer" is requested. |
| P9 | `Offer` raises on `perceived_value ≤ 0` or urgency/scarcity outside [0, 1]. | Our mapper fills perceived value from base price and clamps urgency/scarcity to [0, 1] (adding a mapping warning). Any remaining exception is caught → `ERROR` (shadow guarantee). |
| P10 | `NO_VALID_PRICE` has no price to validate. | Skip validation: `shadow_validation = {status: "SKIPPED", reason: "no valid price"}`. |

**Verdict (personas):** correct and implementable. It replaces the "persona reaction" component from v2 with something stronger. The two real gaps are named personas (P1–P2, solved by a cohort layer in our code) and latency (P5, solved by caching + background runs). Everything else is labeling and integration discipline.

---

## 2. Target layout

```
Monetize360/
  ml_updated/                  # COPIED IN, READ-ONLY. Do not edit, retrain, or regenerate.
  customer_agent_sim/          # COPIED IN, READ-ONLY. Imported as a package; its own 32 tests must still pass.
  backend/
    pyproject.toml             # or requirements.txt (see §2.1)
    alembic.ini
    alembic/versions/0001_init.py
    app/
      main.py                  # FastAPI app factory create_app(settings) ; CORS for :5173
      config.py                # pydantic-settings, reads .env
      db.py                    # SQLAlchemy engine/session; SQLite
      models.py                # ORM tables (§6.1)
      auth.py                  # dev role boundary (§6.3)
      audit.py
      lifecycle.py
      schemas/                 # Pydantic v2 contracts (§3)
        domain.py strategy.py rules.py constraints.py pricing.py optimizer.py persona.py llm.py common.py
      engine/                  # PURE, sync, no DB, no clock, no randomness, no domain names
        expression.py          # port of safe DSL (AST-based, NO eval)
        conditions.py rules.py conflicts.py pipeline.py
        signals.py             # semantic signal mapping + normalisation
        constraints.py candidates.py
        prepare.py finalize.py explain.py validate.py diff.py
      optimizer/
        base.py                # PriceOptimizer protocol
        features.py            # builds the 14 ML features from engine output (single source)
        ml.py                  # MLPriceOptimizer — loads ml_updated artifacts
        simulated.py           # SimulatedPriceOptimizer — same features, transparent logistic
        factory.py
      llm/
        base.py                # StrategyLlm protocol
        gemini.py grok.py mock.py  prompts.py  convert.py (flat→tree)
      validation/              # wraps customer_agent_sim — shadow mode only
        adapter.py             # Offer mapping from PricingDecision; calls validate_offer; catches → ERROR
        populations.py         # cached shared populations per (seed,size) and per cohort
        cohorts.py             # named persona cohorts, utility decomposition → concern/suggestion
        ml_checks.py           # Check F: ML monotonicity across candidates
        service.py             # sync/background orchestration, persistence
        narrative.py           # optional Gemini/Grok rewrite of persona text (validated, fallback)
      services/
        pricing.py strategy.py simulation.py domains.py
      routers/
        domains.py strategy.py versions.py pricing.py catalog.py audit.py health.py admin.py
      seed/seed.py
    tests/
      engine/ optimizer/ llm/ validation/ api/ test_pipeline_e2e.py conftest.py
  examples/domains/
    hospitality.json gpu-compute.json edtech.json car-rental.json   # car-rental NOT seeded
    invalid/*.json
  src/                         # frontend (React) — client only
  scripts/check-agnostic.py    # replaces .mjs
  .env.example
```

### 2.1 Python dependencies (`backend/requirements.txt`)
```
fastapi
uvicorn[standard]
pydantic>=2
pydantic-settings
sqlalchemy>=2
alembic
python-dotenv
httpx
google-genai           # Gemini
openai                 # Grok via xAI's OpenAI-compatible endpoint (base_url=https://api.x.ai/v1)
scikit-learn==<version from training_metadata.json>
numpy
pandas                 # only if ml_updated/features.py requires it
joblib
pytest
pytest-asyncio
```
Pin exact versions after installing (`pip freeze > requirements.lock`). Use a venv at `backend/.venv`.
**sklearn/Python compatibility:** if the training sklearn version has no wheel for Python 3.13, create the venv with the Python version used for training (`py -3.11 -m venv .venv`). Document this in the README.

### 2.2 Root scripts / tooling
- `package.json`: keep `dev`, `build`, `test` (vitest, frontend only). Add `"dev:all": "concurrently \"npm:api\" \"npm:dev\""` and `"api": "cd backend && .venv\\Scripts\\uvicorn app.main:app --reload --port 8000"`.
- `vite.config.ts`: add `server.proxy: { '/api': 'http://localhost:8000' }`.
- `.claude/launch.json`: add an `api` config (port 8000).
- `scripts/check-agnostic.py`: scans `backend/app` (excluding `seed/`), `src/pages`, `src/components`. Fails on `hotel|room|occupancy|loan|gpu|tutor|tour|car|airline` and on `domain ==`/`domain_id ==` comparisons inside `backend/app/engine`, `optimizer`, `personas`.

---

## 3. Contracts (Phase 1) — Pydantic v2 models in `backend/app/schemas/`

Use `model_config = ConfigDict(extra="forbid")` on every model, so unknown keys get rejected.

### 3.1 Domain
```python
SemanticSignal = Literal[
  "basePrice","unitCost","competitorPrice","perceivedValue",
  "customerSensitivity","customerLoyalty","urgency","scarcity","demandPressure",
  "competitorDensity","promotionActive","timeDecay",
  "capacityUtilization","supplyAvailability","risk"]          # last three: rule-only, not ML inputs

class Normalize(BaseModel): min: float; max: float; invert: bool = False; target_range: tuple[float,float] = (0,1)
   # demandPressure uses target_range (0.2, 2.0) to match training range

class Attribute(BaseModel):
  key: str (snake/camel id regex); label; type: Literal["number","text","boolean","date","enum"]
  source: Literal["request","profile","inventory","computed"]
  semantic: SemanticSignal | None; unit: str | None; required: bool = False
  customer_input: bool = False; allowed_values: list[str] | None; default: Scalar | None
  expression: str | None   # computed only, safe DSL
  normalize: Normalize | None; min/max/description/stage/synonyms (as in TS)

# Persona: see §3.4 (agent-cohort definition)

class Product(BaseModel): id; name; base_price>0; cost>=0; inventory: dict[str,Scalar] = {}; previous_price: float|None

class DomainConfig(BaseModel):
  id; name; description; currency: str (ISO 4217); output_unit: OutputUnit; product_noun: str
  attributes: list[Attribute]; pipeline: list[str]; personas: list[Persona]; products: list[Product]
  defaults: dict[SemanticSignal, float] = {}        # e.g. customerSensitivity 0.5
```
Validators: unique keys; each semantic signal mapped at most once; `enum` needs `allowed_values`; `normalize.min < max`; computed expressions parse.

### 3.2 Strategy
```python
Op = Literal[">",">=","<","<=","==","!=","in","not_in","between"]
Leaf = {attr, op, value};  Condition = Leaf | {all:[...]} | {any:[...]} | {not: Condition}   # depth ≤ 4
LlmCondition (flat) = {match: "all"|"any", conditions: list[Leaf & {negate: bool=False}]}

ActionType = Literal["percent","fixed","set","none","multiply","formula"]   # formula: manual edit only
Rule: id, name, description, condition, action{type, value, expression?}, priority:int, stage,
      enabled=True, effective_from/to, customer_message: str|None, explanation_template: str|None,
      policy_ref, assumptions, created_by, created_at

ConstraintType = Literal["floor","ceiling","margin_floor","markup_floor","max_discount",
                         "max_increase","max_daily_change","rounding"]
   # margin_floor (gross): (p-c)/p*100 >= v → p >= c/(1-v/100)
   # markup_floor:         (p-c)/c*100 >= v → p >= c*(1+v/100)
   # max_increase: p <= base*(1+v/100) ; max_discount: p >= base*(1-v/100)
   # implicit always: p > 0 and p >= cost (unless domain.allow_below_cost)

ConflictPolicy: mode Literal["stack","first_match_per_stage","highest_priority_wins"],
                combine Literal["additive","multiplicative"], max_total_adjustment_pct: float|None
   # order ALWAYS: stage index → priority asc → rule id asc
CandidatePolicy: count=5, offsets_pct=[-10,-5,0,5,10], include_base_price=True,
                 include_bound_pins=True, show_alternatives=True
ValidationPolicy: minimum_simulated_conversion=0.05, max_allowed_probability_gap=0.15, ranking_tolerance_rank_drop=2,
                  ranking_tolerance_profit_gap=0.12, scenarios=[budget_squeeze, competitor_undercut, high_urgency, perceived_value_drop],
                  personas: list[str] | None (ids; default = all domain personas)
OptimizerPolicy: objective Literal["max_expected_profit","max_expected_revenue"]="max_expected_profit",
                 conversion_floor: float = 0.10
StrategyConfig: domain_id, name, objective_text, rules, constraints, conflict_policy,
                candidate_policy, optimizer_policy, validation_policy
```

### 3.3 Engine / optimizer / decision
```python
GeneratedCandidate(price, reasons: list[str])
CheckedCandidate(GeneratedCandidate, eligible: bool, violations: list[Violation])

MLFeatures (exactly the 14 model columns, same names & order as ml_updated/features.py):
  price_ratio, log_price_ratio, price_vs_competitor, margin_ratio, price_rank_normalized,
  customer_sensitivity, customer_loyalty_score, urgency_score, scarcity_score, demand_pressure,
  competitor_density, is_promotion_active, time_decay_factor, perceived_value_gap

OptimizerRequest(domain_id, strategy_version_id, base_price, unit_cost, competitor_price,
                 perceived_value, signals: dict[str,float], candidates: list[CheckedCandidate],
                 objective, conversion_floor)
CandidateEvaluation(price, features: MLFeatures, p_buy, expected_revenue, expected_profit,
                    eligible, selection_status: Literal["SELECTED","NOT_SELECTED","INELIGIBLE",
                    "BELOW_CONVERSION_FLOOR"], explanation)
ModelInfo(name, version, kind: Literal["ml","simulated"], training_data: Literal["synthetic","n/a"],
          artifact_sha256: str|None)
OptimizerResponse(evaluations, selected_price: float|None, model: ModelInfo, warnings: list[str])

PricingDecision(decision_id, domain_id, strategy_version_id, version_number,
  status: Literal["OK","NO_VALID_PRICE"], final_price: float|None, currency, objective,
  reference_price, anchor_price, applied_rules, overridden_rules, skipped_rules,
  constraints_evaluated, candidates: list[CandidateEvaluation + reasons + violations],
  persona_summary: list[PersonaReaction]|None,
  explanation: {business: list[str], customer: list[str]},
  provenance: {optimizer: ModelInfo, personas: str|None, llm_generated_strategy: bool},
  context_snapshot, request, warnings, created_at)

CustomerQuote(decision_id, status, final_price, currency, unit_label, factors: list[str],
              alternatives: list[{price,label}] | None, notice: str)   # NO cost/margin/p_buy/rule ids
```

### 3.4 Persona cohorts & shadow validation
```python
# Domain config: Persona gains an agent-cohort definition (module's own trait scales, NOT ML 0..1 scales)
class TraitShift(BaseModel):            # applied to agents generated by PopulationGenerator
  trait: Literal["price_sensitivity","value_orientation","risk_aversion","patience","loyalty",
                 "decision_noise","budget_ratio","need_intensity","reference_price_bias"]
  op: Literal["multiply","set_range"]; value: float | None; low: float | None; high: float | None
class Persona(BaseModel):  # replaces 3.1 Persona
  id; name; description; preferences: list[str]; share: float
  cohort: list[TraitShift]
  ml_overrides: {customer_sensitivity, customer_loyalty_score, urgency_score} | None  # optional, for ML-per-persona view
  # 6 default personas shipped in backend/app/validation/default_personas.json; domains may override

PersonaReaction(persona_id, name, description, preferences, price,
  buy_share, defer_share, reject_share,               # simulated cohort shares
  reaction: Literal["DELIGHTED","ACCEPTS","HESITANT","REJECTS"],   # bands on buy_share: ≥.75/≥.5/≥.25/else
  main_concern: str, concern_driver: Literal["price","budget","competitor","value","wait","risk","none"],
  explanation: str, suggestion: str|None,
  source: Literal["agent_sim","agent_sim+llm_text"], simulated: Literal[True])

ValidationCheck(key: Literal["A_behaviour","B_heterogeneity","C_scenarios","D_ranking","E_model_disagreement","F_ml_monotonicity"],
  passed: bool|None, available: bool, severity: Literal["ok","review","fail"], summary: str, details: dict)

ShadowValidation(status: Literal["PASS","FAIL","REVIEW_REQUIRED","ERROR","SKIPPED","PENDING"],
  simulated_conversion: float|None, ci: tuple[float,float]|None, model_disagreement_pp: float|None,
  checks: list[ValidationCheck], scenarios: dict, feedback: dict|None,   # module's FeedbackDiagnostic, advisory
  personas: list[list[PersonaReaction]]|None,                            # persona × candidate
  population_size: int, seed: int, module_version: str, error: str|None, advisory_only: Literal[True])
# PricingDecision gains: shadow_validation: ShadowValidation  (business view only; NEVER in CustomerQuote)
```

**Acceptance (Phase 1):** `pytest tests/schemas` passes. All `examples/domains/*.json` parse. Every `examples/domains/invalid/*.json` fails with a path-based message (e.g. `rules.2.condition.op: Input should be '>', ...`).

---

## 4. Phase 2 — Engine port + refactor (`backend/app/engine`, pure Python)

Port from the TS files, keeping behaviour except for the listed changes. Use `decimal.Decimal` with a module-level context (`prec=28`, `ROUND_HALF_EVEN` unless the TS used half-up; match the TS tests).

| TS source | Python target | Notes |
|---|---|---|
| expression.ts | expression.py | Hand-written tokenizer + recursive-descent parser over a whitelist (numbers, attrs, + − × ÷, comparisons, `min max abs round daysBetween isWeekend if`). **No `eval`, no `ast.literal_eval` of user strings, no attribute access.** Port the "constructor" security test as "dunder/attribute access rejected". |
| conditions.ts | conditions.py | Add `not_in`. Return reasons for failures. |
| rules.ts | rules.py | Sort by (stage index, priority, id). Drop createdAt from ordering. |
| — | conflicts.py | **new**: `resolve(fired, policy) -> (applied, overridden[{rule_id, reason, overridden_by}])` |
| pipeline.ts | pipeline.py | Actions percent/fixed/set/none/multiply/formula. Apply `max_total_adjustment_pct` cap as a logged step. |
| context.ts | context.py | Plus `signals.py`: semantic mapping → raw values + normalised values in the target range. |
| constraints.ts | constraints.py | Add markup_floor, max_increase, implicit cost floor. `compute_bounds` → if `lower > upper` → conflict. **Remove clamping from the decision path** (keep `clamp()` only for computing bound pins). |
| optimise.ts (grid) | candidates.py | **new logic**: anchor offsets + base price + bound pins (only if the anchor is out of bounds) → round (step, then currency decimals) → dedupe merging reasons → sort. `check_candidates()` marks eligibility + violations. |
| price.ts | prepare.py / finalize.py | `prepare(domain, product, strategy, request, profile) -> Prepared` (status may already be NO_VALID_PRICE on conflicting bounds). `finalize(prepared, optimizer_response) -> PricingDecision`: re-check the selected price against the constraints + conversion floor, then recompute the argmax over eligible evaluations (ties → lower price). If the optimizer pick differs, override it and add the warning `OPTIMIZER_SELECTION_OVERRIDDEN`. No eligible candidate → NO_VALID_PRICE. |
| explain.ts | explain.py | Business sentences + customer sentences (from `customer_message` only). Add "why the winner": compare against the runner-up by expected profit. |
| validate.ts | validate.py | All existing checks + `not_in`/`none`/new constraints, unsupported operator, `formula` forbidden when `source="llm"`, rule references to an unmapped attribute, a priority tie that changes the price (port), and an OOD warning when constraints allow price_ratio outside 0.6–1.5. |
| diff.ts | diff.py | |
| policyParser.ts / domainSuggest.ts | llm/mock.py | Mock generator (Phase 4). |
| demand.ts / personas.ts | — | Superseded by `customer_agent_sim` + `validation/cohorts.py` (Phase 4c). Delete with the TS engine. |

Deterministic IDs: `decision_id = "D-" + sha256(canonical_json(request, version_id, model.version))[:16]`. The canonical JSON uses sorted keys.

**Tests** (`tests/engine/`): port the existing 48 vitest cases where the behaviour is unchanged (update the clamping/grid ones to the new semantics). Add: the 3 conflict modes + overridden log, deterministic tie-break, `not_in`, `none`, margin vs markup (cost 100 @20% → 125 vs 120), max_increase, dedupe merges reasons, default 5 candidates, ineligible candidates visible and not clamped, NO_VALID_PRICE (conflicting bounds; all candidates ineligible), finalize overrides a bad optimizer pick, repeatability (deep-equal), and the expression security tests.

**Acceptance:** `pytest tests/engine` green. `python scripts/check-agnostic.py` green.

---

## 5. Phase 3 — Optimizer (`backend/app/optimizer`)

```python
class PriceOptimizer(Protocol):
    info: ModelInfo
    def evaluate(self, req: OptimizerRequest) -> OptimizerResponse: ...
```
- **`features.py`** is the single place that builds the 14 features. First **read `ml_updated/features.py`** and mirror its formulas and column order exactly (import it directly if it has no side effects: add `ml_updated` to `sys.path` and import the feature function). Rank is computed over **all** generated candidates (see §1 #2). `perceived_value` falls back to base price with the warning `DEFAULT_PERCEIVED_VALUE`. Missing scores use `domain.defaults`, else 0.5 (demand_pressure 1.0, promotion 0, competitor_price = base) with a `DEFAULT_SIGNAL:<name>` warning. Range check → `OOD_FEATURE:<name>`.
- **`ml.py` `MLPriceOptimizer`**: loads `artifacts/model_gbt.pkl` (or `model_lr.pkl` + `scaler.pkl` if `OPTIMIZER_MODEL=lr`) with joblib, **once at startup**. Verifies the SHA-256 against `ML_ARTIFACT_SHA256` (env) when set. `predict_proba(X)[:,1]` with a DataFrame in the training column order. `ModelInfo(name="gbt", version=<from training_metadata.json>, kind="ml", training_data="synthetic")`. Applies the conversion floor + objective argmax over eligible candidates. **Never** writes to `ml_updated/`.
- **`simulated.py` `SimulatedPriceOptimizer`**: same features. `z = 1.2 − 4·sens·log_price_ratio − 2.5·max(0, price_vs_competitor−1) + 0.6·urgency + 0.4·scarcity + 0.3·(demand_pressure−1) + 0.5·loyalty + 1.5·perceived_value_gap + 0.3·promo`, `p = sigmoid(z)`. Deterministic. `kind="simulated"`.
- **`factory.py`**: `OPTIMIZER_PROVIDER=ml|simulated|auto` (default `auto` = ml if artifacts load, else simulated with a startup log line). `/health` reports which one is active.
- If the copied ML code exposes an HTTP service instead, add `http.py` with the same protocol. Not required.

**Tests:** both optimizers return schema-valid responses. Expected-profit math. The highest **profit** wins, not the highest p_buy. Conversion floor. Determinism. Feature vector order equals the model's `feature_names_in_` (skip with `pytest.mark.skipif` when the artifacts are absent). The README demo case (base 100, cost 40, comp 95, sens 0.65, loyalty 0.30, candidates 80–120) runs through the full engine and gives a valid decision (assert structure, not the exact $90). Finalize rejects a fake optimizer that selects an ineligible price.

---

## 6. Phase 4 — Persistence, lifecycle, auth, LLM, API

### 6.1 Tables (SQLAlchemy 2 + Alembic; JSON columns validated by Pydantic on write and read)
```
domains(id PK, name, config JSON, created_by, created_at, updated_at)
strategy_drafts(domain_id PK FK, config JSON, status CHECK IN('DRAFT','VALIDATED'), content_hash,
                source CHECK IN('manual','llm','mock'), updated_by, updated_at)
strategy_versions(id PK, domain_id FK, number, status CHECK IN('APPROVED','ACTIVE','RETIRED'),
                  config JSON, domain_snapshot JSON, content_hash, created_by, approved_by,
                  approved_at, published_at, retired_at, notes, UNIQUE(domain_id, number))
   + partial unique index uq_one_active ON strategy_versions(domain_id) WHERE status='ACTIVE'
validation_runs(id, domain_id, content_hash, result JSON, created_by, created_at)
simulation_runs(id, domain_id, content_hash, version_id NULL, request JSON, result JSON, created_by, created_at)
decisions(id PK, domain_id, version_id, status, final_price, request JSON, result JSON, created_at)
candidate_evaluations(id, decision_id FK, price, eligible, violations JSON, p_buy, expected_revenue,
                      expected_profit, selection_status)
shadow_validations(id, owner_type CHECK IN('simulation','decision'), owner_id, status, result JSON,
                   population_size, seed, created_at)      -- includes persona cohort results
audit_logs(id, at, actor, role, operation, resource_type, resource_id, version, outcome, details JSON)
customers(id, domain_id NULL, profile JSON)
```

### 6.2 Lifecycle (`lifecycle.py`, transition table + one function per transition)
```
draft DRAFT --validate ok--> VALIDATED ; any PUT --> DRAFT (hash changed)
approve: needs VALIDATED and validation_run.content_hash == draft.content_hash
         and approver role manager|admin and approver != draft.updated_by  → new version APPROVED
publish(version_id): APPROVED|RETIRED → ACTIVE; previous ACTIVE → RETIRED   (single DB transaction)
rollback(version_id): RETIRED → ACTIVE; current ACTIVE → RETIRED (transaction, op="ROLLBACK")
```
Invalid → HTTP 409 `{"error":{"code":"INVALID_TRANSITION","from":..,"to":..}}`. Version rows are immutable apart from status/timestamps (repository has no update method for `config`). Every mutation writes audit (`outcome="FAILED"` on errors, in a separate session so the rollback doesn't erase it).

### 6.3 Auth (dev-grade, stated as such in the README)
`Authorization: Bearer dev:<role>:<user>`, roles `analyst|manager|admin|customer`, as a FastAPI dependency `require_role(...)`. 401 when missing or malformed, 403 when the role is wrong. Customer may only use `GET /catalog` and `POST /pricing/quote` (always receives a `CustomerQuote`).

### 6.4 LLM strategy generator (`llm/`)
```python
class StrategyLlm(Protocol):
    name: Literal["gemini","grok","mock"]
    def generate_strategy(self, domain: DomainConfig, description: str) -> dict: ...
    def suggest_domain(self, description: str) -> dict: ...
```
- **Gemini** (`google-genai`): `client.models.generate_content(model=GEMINI_MODEL, contents=..., config={"response_mime_type":"application/json","response_schema": LlmStrategyOut})`. Use the **flat** `LlmStrategyOut` schema (no recursion, no `formula`). Check the current google-genai docs for exact params.
- **Grok** (`openai` SDK, `base_url="https://api.x.ai/v1"`, `XAI_API_KEY`, `GROK_MODEL`): `response_format={"type":"json_schema","json_schema":{"name":"strategy","schema":LlmStrategyOut.model_json_schema(),"strict":True}}`. If strict schema is rejected, fall back to `{"type":"json_object"}` + Pydantic validation. Check the current xAI docs for model names.
- **Pipeline:** provider → `LlmStrategyOut.model_validate` → `convert.py` (flat→tree, assign ids, `source="llm"`) → `validate_strategy(strict=True, allow_formula=False)` → if errors, **one repair call** with the error list → still invalid → HTTP 422 with errors (the draft is unchanged).
- System prompt: produce configuration only. Use only the listed attribute keys, operators, action types and constraint types. Never output code or a final price. Percent values are in percentage points.
- `LLM_PROVIDER=gemini|grok|mock`. If the key is missing → mock with `provider="mock"` and a UI badge "Mock generator — no LLM configured". **Never label mock output as LLM.**
- Tests use a fake provider (junk JSON, wrong types, unknown operator, `formula`, code-looking strings) → 422. Real provider calls are tested only manually when a key exists; report them as unverified otherwise.

### 6.5 Customer-agent validation + personas (`validation/`) — shadow mode, never fed into selection
Runs **after** `finalize()`. Wraps the copied `customer_agent_sim` package (import it, never edit it).
- **`adapter.py`**: `to_offer(decision) -> Offer` using the README mapping, sourced from semantic signals: price = final price, base_price = reference, perceived_value (fallback: base), competitor_price (**None** if unmapped, never invented), urgency/scarcity (normalised 0..1, clamped, with warning), unit_cost, context = {domain_id, version}. `ml_probability` = selected candidate's p_buy. `candidate_prices` / `ml_probabilities` = **all evaluated candidates** (eligible and ineligible both, so ranking sees the full set). Config from the strategy's `validation_policy` (defaults = module defaults: min conversion 0.05, gap 0.15, rank drop 2, profit gap 0.12, seed 42). Calls `validate_offer(..., population=cached)` + `extract_feedback_diagnostics`. **Any exception → `status="ERROR"`, logged, quote unaffected.**
- **`populations.py`**: in-memory LRU of `PopulationGenerator().generate(size, seed)` keyed by `(size, seed)` and per cohort `(persona_id, size, seed+i)`. Warmed at startup for the seeded domains.
- **`cohorts.py`**: builds each named persona's cohort (§1b P1), evaluates every candidate price on it with CRN, computes buy/defer/reject shares, decomposes utility (§1b P2) → `concern_driver`, `main_concern`, `suggestion` from templates. Default 6 personas in `default_personas.json`.
- **`ml_checks.py`**: Check F (ML monotonicity across candidates, tolerance 0.02).
- **Overall status** = module status, escalated to at least `REVIEW_REQUIRED` if Check F fails. `NO_VALID_PRICE` → `SKIPPED`.
- **`service.py`**: `VALIDATION_MODE=sync|background|off`. The Simulator endpoint always runs sync (population `VALIDATION_POP_SIM`, default 1000). Customer quotes use background (`VALIDATION_POP_QUOTE`, default 250) via FastAPI `BackgroundTasks`, writing `shadow_validations` with the decision id. Until it lands, the decision shows `PENDING`.
- **`narrative.py`** (optional, `PERSONA_TEXT_PROVIDER=gemini|grok|none`): rewrites only `main_concern/explanation/suggestion` text from the structured numbers. Pydantic-validated. Falls back to template text on any failure. Source becomes `agent_sim+llm_text`.
- **Never** call `bounded_validation_pipeline`, never apply `suggested_action` or `alternative_candidate_price`. Shown as advisory only, with an engine eligibility check on the suggested price.
- **Tests (`tests/validation/`)**: shadow invariance (`final_price`, candidates and statuses deep-equal with validation on vs off), exception containment (monkeypatch `validate_offer` to raise → ERROR, quote 200), SKIPPED on NO_VALID_PRICE, competitor None → competitor scenario not applicable, determinism (same seed → identical report), cohort shifts move shares the expected way (budget persona buy share < premium persona at the same high price), every persona REJECTing doesn't change `final_price`, Check F flags a fake optimizer with an increasing p_buy ladder, customer quote never contains `shadow_validation`, `PENDING` → stored result after the background task. Also run `python -m pytest customer_agent_sim/tests` to confirm the module's own 32 tests still pass in our venv.

### 6.6 Routes (`/api/v1`, error shape `{"error":{"code","message","details"}}`)
| Method & path | Role | Notes |
|---|---|---|
| POST `/domains` | analyst+ | create from DomainConfig JSON (also the 4th-domain import) |
| GET `/domains`, GET/PUT `/domains/{id}` | analyst+ | |
| POST `/domains/suggest` | analyst+ | NL → DomainConfig draft |
| POST `/domains/{id}/generate-strategy` | analyst+ | → draft DRAFT, `{draft, provider, warnings, assumptions}` |
| GET/PUT `/domains/{id}/strategy` | analyst+ | |
| POST `/domains/{id}/validate` | analyst+ | stores validation_run |
| POST `/domains/{id}/simulate` | analyst+ | draft or `version_id`. Scenarios. Includes a **sync** shadow validation + persona cohorts × all candidates. Stores simulation_run + shadow_validations |
| GET `/pricing/decisions/{id}/validation` | analyst+ | shadow validation for a decision (PENDING / result) |
| GET `/personas/defaults` | analyst+ | default persona cohort definitions |
| POST `/domains/{id}/approve` | manager+ | |
| POST `/domains/{id}/publish` `{version_id}` | manager+ | |
| GET `/domains/{id}/versions`, GET `/domains/{id}/versions/{a}/diff/{b}` | analyst+ | |
| POST `/domains/{id}/rollback` `{version_id}` | manager+ | |
| GET `/catalog` | any | active domains, products (no cost), customer_input attributes |
| POST `/pricing/quote` | any | `{domain_id, product_id, customer_id?, attributes, as_of?, include_personas?}`. Model has `extra="forbid"`, so a `price`/`final_price` field → 422. Uses the ACTIVE version |
| GET `/pricing/decisions`, GET `/pricing/decisions/{id}` | analyst+ | |
| GET `/audit-logs` | analyst+ | filters |
| GET `/health` | none | active llm / optimizer (+ model version) / validation mode + customer_agent_sim version |
| POST `/admin/reseed` | admin | dev only |

`as_of` defaults to today in the router. The engine never reads the clock.

**Acceptance (Phase 4):** `pytest tests/api` with `TestClient` on a temp SQLite file covers CRUD, mock generate, malformed LLM → 422, PUT invalidates validation, approve-without-validation 409, maker-checker 403, publish/rollback keeps exactly one ACTIVE, published version immutable, quote stores version id, customer quote deep-scan has no `cost|margin|p_buy|rule` keys, `final_price` in request → 422, 401/403 matrix, audit rows per mutation, NO_VALID_PRICE → 200 with that status.
**E2E** `tests/test_pipeline_e2e.py`: import car-rental.json → generate (mock) → edit a rule → validate → approve (other user) → publish → quote → get decision → publish v2 → rollback v1 → new quote cites v1.

---

## 7. Phase 5 — Seed data & domain independence
- `hospitality.json`: base, cost, occupancy→capacityUtilization (rule-only), demand_index→demandPressure (normalize to 0.2–2.0), competitor_rate→competitorPrice, days_to_arrival→urgency (normalize invert), loyalty from profile. Rules: demand high AND occupancy>80 → +12%; demand low → −8%. Constraints: margin_floor 25, max_increase 20, rounding 50. INR.
- `gpu-compute.json`: hourly price, utilization→capacityUtilization, available_gpus→scarcity (invert), queue_demand→demandPressure, commitment enum with `not_in`. markup_floor 30, ceiling, rounding 0.05. USD.
- `edtech.json`: session price, tutor_availability→scarcity (invert), demand→demandPressure, hours_until_session→urgency/timeDecay, student_tier enum, promotion flag→promotionActive. floor, max_discount 15, rounding 10. INR.
- `car-rental.json`: **not seeded**. Used in the demo + e2e test.
- `seed.py`: reset DB, load the 3 domains + a starter strategy each, v1 approved by `seed-manager` and published, demo customers with sensitivity/loyalty. Descriptions say "synthetic demo data".
- Test: quote all 3 seeded domains through the API → status OK, distinct currencies, and the same code path (assert `provenance.optimizer` identical and no domain branches, which the agnostic script also checks).

---

## 8. Phase 6–7 — Frontend (React stays)
**API client:** `src/api/http.ts` (`fetch('/api/v1'+path)`, `Authorization` from the role switcher, error → `ApiError`). Rewrite `src/api/*.ts` to use it, with snake_case ↔ camelCase in one mapper, or switch the TS types to snake_case. Replace the zustand-db hooks with a tiny `useQuery/invalidate` helper. Delete `src/store/db.ts`, `src/api/router.ts`, `src/data/seed/`, and all `src/engine/*` except `format.ts`. Frontend must never compute a price. `check-agnostic.py` fails if `src/pages|components` import anything from `src/engine` other than `format`.

**Business portal `/business/*`** (keep AppLayout + role switcher, add the Customer role):
1. Dashboard: domains, active versions, recent decisions, NO_VALID_PRICE count, **decisions needing review** (shadow FAIL/REVIEW_REQUIRED), provider badges (LLM gemini/grok/mock · Optimizer ML-GBT/simulated · "trained on synthetic data").
2. Domains: list, New-domain wizard (describe → AI suggest → attributes with semantic mapping/normalize/customer_input → products → personas), **Import JSON** (4th-domain demo, shows validation errors).
3. Strategy Studio: NL objective → Generate (provider badge, assumptions) → rule table editor (add/edit/delete/enable/priority/stage, condition builder all/any/not, action editor, customer message) → conflict/candidate/optimizer policy + constraints → Validate (errors linked to rows) → DRAFT/VALIDATED pill.
4. Simulator: product + attributes + presets → run on draft/version → candidate table (price, reasons, eligible ✓/✗ + violation, **P(buy)**, exp. revenue, exp. profit, status, OOD warnings), selected row highlighted, rejected rows visible → Recharts price vs E[profit] → applied/overridden rules → business explanation → **Validation panel**: overall status chip (PASS / REVIEW_REQUIRED / FAIL / ERROR), simulated conversion with 95% CI, "model disagreement" in pp, checks A–F as rows (✓ / ⚠ / ✗ + one-line summary), scenario table (budget squeeze, competitor undercut, high urgency, value drop, with conversion vs baseline and "n/a" when not applicable), advisory feedback (suggested action + alternative price, marked "eligible" or "violates <constraint>") → **Persona grid**: persona × candidate cells with buy/defer/reject bar + reaction chip, click → concern + suggestion. Banner: "Simulated customer agents (shadow mode). Advisory only — does not change the selected price."
5. Versions & Approvals: approve (disabled with a reason for analyst/self), publish, rollback, diff.
6. Decisions: list (filter by validation status) + full trace by id, including the shadow validation (polls while PENDING).
7. Audit log.

**User portal `/portal`** (separate layout, customer role): catalog → form from customer_input attributes → Get price → price + unit, factors, alternatives, notice "Demo engine; purchase predictions come from a model trained on synthetic data." NO_VALID_PRICE → "Not available for these details."

**Acceptance:** `npm run build` OK. `npm run dev:all`, then the §10 demo runs in the browser preview with no console errors. Narrow width OK.

---

## 9. Phase 8 — Docs, env, verification
`.env.example`:
```
DATABASE_URL=sqlite:///./data/monetize360.db
LLM_PROVIDER=mock              # gemini | grok | mock
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash  # confirm current model id
XAI_API_KEY=
GROK_MODEL=grok-4              # confirm current model id
PERSONA_TEXT_PROVIDER=none     # gemini | grok | none
VALIDATION_MODE=background     # sync | background | off  (customer quotes; Simulator is always sync)
VALIDATION_POP_SIM=1000
VALIDATION_POP_QUOTE=250
VALIDATION_SEED=42
OPTIMIZER_PROVIDER=auto        # auto | ml | simulated
OPTIMIZER_MODEL=gbt            # gbt | lr
ML_ARTIFACTS_DIR=../ml_updated/artifacts
ML_ARTIFACT_SHA256=
CORS_ORIGINS=http://localhost:5173
```
README: venv setup (Windows PowerShell commands), `alembic upgrade head`, `python -m app.seed.seed`, `uvicorn app.main:app --reload --port 8000`, `npm run dev`, `pytest`, `npm test`, a sample quote via `curl`/`Invoke-RestMethod`, architecture diagram (Portal → LLM → Engine → ML Optimizer → Personas → API → User Portal), the conflict policy, and the ML integration notes from §1. PROGRESS.md: verification table (feature | test | result | command | limitation). Only claim what was run.

## 10. Demo (5–7 min)
1. Dashboard: 3 domains, one engine, ML-GBT badge.
2. Import car-rental.json, or create "Coworking desks" in the wizard.
3. Strategy Studio: NL objective → Generate (Gemini/Grok) → edit + disable a rule → Validate (show an error, fix it).
4. Simulator "high demand": 5 candidates, one rejected by max_increase (reason visible). Chart shows why the winner has the max E[profit].
5. Validation panel: status PASS with CI. Scenario table shows the competitor-undercut drop. Persona grid: budget-conscious rejects, urgent buyer accepts, with concerns. Toggle validation off and on: price unchanged (shadow mode). Optional: a strategy with an aggressive +20% price → REVIEW_REQUIRED from the ranking/scenario checks.
6. Approve as Manager → Publish.
7. User portal quote → customer-safe factors.
8. Decision trace (version, rules, constraints, model version, features) + audit row.
9. GPU Compute: same flow, USD, different signals.
10. Rollback → re-quote cites v1. Optional: "Case F" all-below-cost scenario → NO_VALID_PRICE.

## 11. Phase order & model guidance
| # | Phase | Size | Model |
|---|---|---|---|
| 1 | Pydantic contracts + example/invalid JSONs | M | Sonnet |
| 2 | Engine port + refactor + tests | **XL** | **Sonnet** (split: 2a expression/conditions/rules/conflicts/pipeline; 2b constraints/candidates/prepare/finalize/explain/validate) |
| 3 | Optimizer (features, ML adapter, simulator) | M | Sonnet (needs `ml_updated/` copied in first) |
| 4 | DB, lifecycle, auth, LLM (Gemini/Grok), shadow validation + personas, routers, API + e2e tests | L | Sonnet (split 4a DB/lifecycle/routers, 4b LLM, 4c validation/cohorts; needs `customer_agent_sim/` copied in) |
| 5 | Seed JSONs + seeding + 3-domain test | S | Haiku OK |
| 6 | Frontend API client swap + cleanup | M | Sonnet |
| 7 | Business screens + user portal | L | Sonnet, or Haiku per screen |
| 8 | Docs, env, verification, demo dry-run | S | Haiku OK |

**Rules for the implementing model:** run `pytest` (+ `npm test`/`npm run build` for frontend phases) and `python scripts/check-agnostic.py` at the end of every phase, and never claim a pass you didn't run. No domain branching in engine/optimizer/personas. Never `eval`/`exec` anything, never execute LLM output. Optimizer/personas never override hard constraints. Never label mock/simulated/synthetic-trained output as real. **Never modify, retrain or regenerate anything in `ml_updated/` or `customer_agent_sim/`.** Shadow validation never changes a price, eligibility or status, and never blocks a quote. Files under ~300 lines.

## 12. Prerequisites before Phase 3
1. Copy the `ml_updated/` folder (code + `artifacts/`) into `C:\Users\97150\Monetize360\ml_updated\`. `data/training_data.csv` is optional.
2. Note the scikit-learn and Python versions from `training_metadata.json`.
3. Copy `customer_agent_sim/` (with `config/` and `tests/`) into `C:\Users\97150\Monetize360\customer_agent_sim\` before Phase 4c.
4. Gemini or xAI API key: **will be supplied later**. Phases 1–8 must complete and pass with `LLM_PROVIDER=mock`. When keys arrive: set them in `backend/.env`, flip `LLM_PROVIDER`, and run the manual LLM checks (generate strategy on all 3 domains + the repair-retry path). Record the result in the verification table.

## 13. Open questions (defaults assumed)
1. ML codebase: the persona README references `d:/HACKFEST/monetize360-datasets/ml/` (domain-specific optimizers), while the ML README describes `ml_updated/` (domain-agnostic). **Default: use `ml_updated/` only.**
2. Should shadow `FAIL` ever block publishing a strategy (i.e. a Simulator FAIL on the preset scenarios blocks approval)? **Default: no, it's a warning on the approval screen only**, which keeps shadow mode intact.
