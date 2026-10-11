# Monetize360 — Universal, Configurable Dynamic Pricing Engine

One deterministic pricing engine for any industry. **Business behaviour lives in validated configuration**
(domains, rules, constraints, policies); the engine, optimizer interface, shadow validation and explanations are shared.

```
Business Portal ─► Market research (SerpApi web search) ─► LLM Strategy Generator (Groq) ─► validated config + approval (DRAFT→VALIDATED→APPROVED→ACTIVE)
      ─► Universal Pricing Engine (context → rules → conflicts → anchor → candidates → hard constraints)
      ─► ML Optimizer (ml_updated GBT: P(buy) → E[profit] → select)  ─► engine re-verifies the pick
      ─► Customer-agent shadow validation (customer_agent_sim: PASS / REVIEW / FAIL + persona cohorts, advisory only)
      ─► Pricing API + explanation ─► User Portal (customer-safe view)
```

| Layer | Where | Notes |
|---|---|---|
| Contracts | `backend/app/schemas` | Pydantic v2, `extra="forbid"` everywhere (a caller can't inject `final_price`) |
| Engine | `backend/app/engine` | Pure Python: no DB, clock, randomness or industry words. `prepare()` → optimizer → `finalize()` |
| Optimizer | `backend/app/optimizer` | `MLPriceOptimizer` (local `ml_updated` artifacts, read-only), `SimulatedPriceOptimizer` fallback, `HttpPriceOptimizer` |
| Market research | `backend/app/market` | SerpApi or Serper: competitor prices per product + market snippets, cached on disk, bounded per run |
| LLM | `backend/app/llm` | **Groq** (`openai/gpt-oss-120b`, default), Gemini or Grok. Prompt in `llm/prompts.py`. One repair retry, then 422. Labelled **mock** generator when no key |
| Validation | `backend/app/validation` | Wraps `customer_agent_sim` (unmodified) + named persona cohorts + Check F (ML monotonicity) |
| API | `backend/app/routers`, `services` | FastAPI `/api/v1`, SQLite + Alembic, dev-grade role tokens, audit log |
| UI | `src/` | React + Vite + Tailwind. Business portal `/business/*`, user portal `/portal`. Never computes prices |
| Domains | `examples/domains/*.json` | hospitality, gpu-compute, edtech, ecommerce, banking (seeded) + car-rental (imported live) + 15 invalid configs |

## Run it (Windows PowerShell)

Prerequisites: Python 3.13, Node 22.

```powershell
# 1. Backend dependencies (scikit-learn is pinned to 1.6.1 because the model pickles were trained with it)
cd backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env          # optional; every setting has a safe default
cd ..

# 2. Frontend dependencies
npm install

# 3. Start both (API on :8010, UI on :5173). The database is created, migrated and seeded on first start.
npm run dev:all
```

Open http://localhost:5173 (business portal) and http://localhost:5173/portal (user portal).
Port 8010 is the API default; to use another port set `M360_API_PORT` for Vite and pass `--port` to uvicorn.

Other commands:

| Task | Command |
|---|---|
| API only | `npm run api` (or `cd backend; .venv\Scripts\python -m uvicorn app.main:app --reload --port 8010`) |
| UI only | `npm run dev` |
| Reset + reseed the database | `npm run db:seed` (or **Reset demo data** in the role menu as Admin) |
| Backend tests (189) | `npm run test:backend` |
| customer_agent_sim's own tests (32) | `npm run test:sim` |
| Frontend tests / types / build | `npm test` · `npm run typecheck` · `npm run build` |
| Domain-independence check | `npm run check:agnostic` |
| API docs | http://localhost:8010/docs |

### Sample quote

```powershell
$h = @{ Authorization = "Bearer dev:customer:guest-gold" }
$body = '{"domain_id":"hospitality","product_id":"DLX","attributes":{"check_in":"2026-10-24","nights":2}}'
Invoke-RestMethod -Method Post -Uri http://localhost:8010/api/v1/pricing/quote -Headers $h -ContentType "application/json" -Body $body
```
```bash
curl -s -X POST http://localhost:8010/api/v1/pricing/quote -H "Authorization: Bearer dev:analyst:priya" \
  -H "Content-Type: application/json" -d '{"domain_id":"gpu-compute","product_id":"ACC-B","attributes":{"commitment":"reserved_1m","hours":200}}'
```
The customer token returns a `CustomerQuote` (price, factors, alternatives). A business token returns the full `PricingDecision`.

## Configuration (`backend/.env`)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `groq`, `gemini`, `grok` or `mock`. Needs `GROQ_API_KEY` / `GEMINI_API_KEY` / `XAI_API_KEY`; falls back to the labelled mock without a key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Any Groq chat model; gpt-oss-120b follows the JSON schema reliably |
| `SEARCH_PROVIDER` | `auto` | `serpapi`, `serper`, `none`; auto uses whichever of `SERPAPI_API_KEY` / `SERPER_API_KEY` is set |
| `MARKET_MAX_PRODUCTS`, `MARKET_CACHE_TTL_H` | 3, 24 | Search budget per generation (≈ 1–2 searches per product + 1) and cache lifetime |
| `GEMINI_MODEL`, `GROK_MODEL` | `gemini-2.5-flash`, `grok-4` | Confirm current model ids with your provider |
| `PERSONA_TEXT_PROVIDER` | `none` | Optional LLM rephrasing of persona concerns (numbers never change) |
| `OPTIMIZER_PROVIDER` | `auto` | `auto` = ML model if artifacts load, else simulator; `ml` (strict), `simulated`, `http` |
| `OPTIMIZER_MODEL` | `gbt` | `gbt` (primary) or `lr` (baseline) |
| `ML_ARTIFACTS_DIR` / `ML_ARTIFACT_SHA256` | `ml_updated/artifacts` / empty | Pin the model hash outside a demo: `1031c5bf8a9bd86404d50a6f4847816b9fd39863aa3eb87c1765b0688b6ba5ea` (model_gbt.pkl) |
| `VALIDATION_MODE` | `background` | Customer quotes: `sync`, `background`, `off`. The Simulator always validates synchronously |
| `VALIDATION_POP_SIM` / `VALIDATION_POP_QUOTE` | 1000 / 250 | Agent population sizes |
| `DATABASE_URL` | `backend/data/monetize360.db` | Any SQLAlchemy URL |

## Key rules the code enforces

- **No domain branching** in the engine/optimizer/validation (checked by `scripts/check-agnostic.py`). New domain = new JSON.
- **Hard constraints are authoritative.** Candidates that break one stay visible as *rejected* with the reason; nothing is clamped.
  If nothing is valid, the result is `NO_VALID_PRICE`. Implicit: price > 0 and price ≥ unit cost.
- **Margin vs markup:** `margin_floor` = (price − cost) / price; `markup_floor` = (price − cost) / cost.
- **Selection:** highest expected profit (or revenue, if configured) among eligible candidates at or above the conversion floor; ties → lower price.
  The engine re-checks the optimizer's pick and overrides it if needed (`OPTIMIZER_SELECTION_OVERRIDDEN`).
- **Conflict resolution:** order is always stage → priority (ascending) → rule id. Modes: `stack`, `first_match_per_stage`,
  `highest_priority_wins`; overridden rules are logged. Optional `max_total_adjustment_pct` cap.
- **LLM output is configuration only:** flat schema, no formulas, no code, validated, never publishes anything.
- **Shadow validation never changes a price**, and errors become `status: ERROR` without blocking the quote.
- **Labels:** P(buy) comes from a model trained on *synthetic* data; persona/agent results are *simulated*. The UI says so.
- **Auth is development-grade** (`Bearer dev:<role>:<user>`, roles analyst / manager / admin / customer) with maker-checker on approval. Not production security.

## Market-grounded strategy generation (Groq + SerpApi)

When a business user clicks **Generate draft** in Strategy Studio (with "Research the market first" on):

1. **Market research** (`app/market`): for up to `MARKET_MAX_PRODUCTS` products it searches Google Shopping (or, for services
   and rates, web results) using only the domain's own names, unit and currency, so it works for any industry. Prices in the
   domain's currency (or percentages for rate-priced domains like banking) are extracted, outliers outside 0.25×–4× of the
   base price are dropped, and min / median / max, confidence and our position vs the market are computed. Results are
   cached for 24 h in `backend/data/market_cache.json` to save search credits.
2. **Groq** receives the strategist prompt (`app/llm/prompts.py::STRATEGY_SYSTEM`) plus a JSON brief: domain, pipeline,
   attributes (the only keys it may use), products, vocabulary, the market summary (no links) and the objective. It returns
   rules, constraints, policies, assumptions and **market_observations** (which numbers it relied on).
3. The output is converted and run through the full validator (no formulas, known attributes only); on errors the model gets
   one repair attempt, otherwise the draft is left unchanged (422).
4. The business user sees the draft rules, assumptions, the market table with sources, and can **Apply medians** to the
   domain's competitor-price attribute. Nothing is published until validated and approved by a different person.

Market data is unverified web data and is labelled that way; listings may not be comparable (different models, sellers).

## Plugging in a different ML model

Implement `PriceOptimizer.evaluate(OptimizerRequest) -> OptimizerResponse` (`backend/app/optimizer/base.py`), or run a service
that accepts/returns that JSON and set `OPTIMIZER_PROVIDER=http`, `OPTIMIZER_URL=...`. Features are built in one place
(`optimizer/features.py`, mirrors `ml_updated/features.py` and is tested against it). The engine, API and UI don't change.

## Demo walkthrough (5–7 min)

1. **Dashboard** (`/business`): 3 domains, one engine; badges show *LLM: mock* and *Optimizer: ML GBT · synthetic training data*.
2. **Domains → Import JSON**: paste `examples/domains/car-rental.json`; a 4th domain appears with no code change.
3. **Strategy Studio** (Hospitality): objective text is prefilled → **Generate draft** (mock label, assumptions, warnings) →
   edit a rule (+10% → +12%), toggle one off → **Validate**.
4. **Simulator**: Deluxe Room, preset *High demand*, check-in date → **Run**. Show the 5 candidates and reasons, any rejected
   by a constraint, P(buy) and expected profit, the chart, and why the winner beats the runner-up.
5. Same page: **Customer-agent validation** (PASS, CI, checks A–F, stress scenarios) and the **persona grid**
   (budget-conscious hesitant/rejects, urgent buyer delighted). Banner: advisory only.
6. **Versions & Approvals**: as Analyst, *Approve* is disabled (role); switch to **Manager** → Approve → Publish.
7. **User portal** (`/portal`): get a price → customer-safe factors, alternatives, synthetic-data notice.
8. **Decisions**: open that decision: version, inputs, rules, constraints, model version, validation. **Audit log** shows every step.
9. Switch the domain to **E-commerce** in Strategy Studio and generate with market research on: Groq drafts rules from
   live competitor prices (table + sources + "how the market shaped the rules"). Then **Banking**: the same engine prices an
   interest rate (shown as 11.50%).
10. **Versions**: roll back to v1 → new quotes cite v1.

## Honest scope

- Real: engine, constraints, lifecycle, persistence, audit, API, both portals, the **trained** GBT model (loaded locally), and
  the customer-agent simulator (run as shipped).
- Simulated/synthetic: the GBT was trained on synthetic data; agent and persona feedback is simulation; all domain data is demo data.
- Verified live: Groq (`openai/gpt-oss-120b`) and SerpApi, end to end in the UI (e-commerce) and via script (banking).
  Gemini and Grok (xAI) clients exist but were not run (no keys).
- Secrets live only in `backend/.env` (gitignored). Rotate keys that have been shared in chat or tickets.
