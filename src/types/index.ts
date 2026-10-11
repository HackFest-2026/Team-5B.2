// TypeScript mirror of the backend Pydantic contracts (snake_case, exactly as the API returns them).
// The frontend never computes prices: these are display types only.

export type Scalar = number | string | boolean | null;
export type Role = "analyst" | "manager" | "admin" | "customer";

export interface OutputUnit { label: string; prefix: string; suffix: string; decimals: number }

export type AttributeType = "number" | "text" | "boolean" | "date" | "enum";
export type AttributeSource = "request" | "profile" | "inventory" | "computed";
export const SEMANTIC_SIGNALS = [
  "competitorPrice", "perceivedValue", "customerSensitivity", "customerLoyalty", "urgency", "scarcity", "demandPressure",
  "competitorDensity", "promotionActive", "timeDecay", "capacityUtilization", "supplyAvailability", "risk",
] as const;
export type SemanticSignal = (typeof SEMANTIC_SIGNALS)[number];

export interface Normalize { min: number; max: number; invert?: boolean; target_range?: [number, number] }

export interface Attribute {
  key: string;
  label: string;
  type: AttributeType;
  source: AttributeSource;
  semantic?: SemanticSignal | null;
  unit?: string | null;
  required?: boolean;
  customer_input?: boolean;
  allowed_values?: string[] | null;
  default?: Scalar;
  expression?: string | null;
  normalize?: Normalize | null;
  min?: number | null;
  max?: number | null;
  description?: string | null;
  stage?: string | null;
}

export interface TraitShift { trait: string; op: "multiply" | "set_range"; value?: number | null; low?: number | null; high?: number | null }
export interface Persona { id: string; name: string; description: string; preferences: string[]; share: number; cohort: TraitShift[] }
export interface Product { id: string; name: string; base_price: number; cost: number; inventory: Record<string, Scalar>; previous_price?: number | null }

export interface DomainConfig {
  id: string;
  name: string;
  description: string;
  currency: string;
  output_unit: OutputUnit;
  product_noun: string;
  attributes: Attribute[];
  pipeline: string[];
  personas: Persona[];
  products: Product[];
  defaults?: Record<string, number>;
  allow_below_cost?: boolean;
}

export interface DomainSummary {
  id: string; name: string; description: string; currency: string; products: number;
  draft_status: "DRAFT" | "VALIDATED" | null; active_version: string | null; created_at: string;
}

// ---------- rules & strategy ----------
export type Op = ">" | ">=" | "<" | "<=" | "==" | "!=" | "in" | "not_in" | "between";
export const OPS: Op[] = [">", ">=", "<", "<=", "==", "!=", "in", "not_in", "between"];
export interface LeafCondition { attr: string; op: Op; value: Scalar | Scalar[] }
export type Condition = LeafCondition | { all: Condition[] } | { any: Condition[] } | { not: Condition };

export type ActionType = "percent" | "fixed" | "set" | "none" | "multiply" | "formula";
export interface RuleAction { type: ActionType; value?: number | null; expression?: string | null }
export interface Rule {
  id: string; name: string; description?: string; condition: Condition; action: RuleAction; priority: number; stage: string;
  enabled: boolean; effective_from?: string | null; effective_to?: string | null; customer_message?: string | null;
  created_by?: string; created_at?: string | null; assumptions?: string[]; policy_ref?: string | null; explanation_template?: string | null;
}
export type ConstraintType = "floor" | "ceiling" | "margin_floor" | "markup_floor" | "max_discount" | "max_increase" | "max_daily_change" | "rounding";
export const CONSTRAINT_TYPES: ConstraintType[] = ["floor", "ceiling", "margin_floor", "markup_floor", "max_discount", "max_increase", "max_daily_change", "rounding"];
export interface Constraint { type: ConstraintType; value: number }

export interface StrategyConfig {
  domain_id: string;
  name: string;
  objective_text: string;
  rules: Rule[];
  constraints: Constraint[];
  conflict_policy: { mode: "stack" | "first_match_per_stage" | "highest_priority_wins"; combine: "additive" | "multiplicative"; max_total_adjustment_pct?: number | null };
  candidate_policy: { count: number; offsets_pct: number[]; include_base_price: boolean; include_bound_pins: boolean; show_alternatives: boolean };
  optimizer_policy: { objective: "max_expected_profit" | "max_expected_revenue"; conversion_floor: number };
  validation_policy: Record<string, unknown>;
  source: "manual" | "llm" | "mock";
}

export interface ValidationIssue { id: string; severity: "error" | "warning" | "info"; code: string; message: string; path?: string | null; rule_ids: string[]; details?: Record<string, unknown> | null }
export interface ValidationReport { ok: boolean; issues: ValidationIssue[]; counts: Record<"error" | "warning" | "info", number> }

export interface DraftView {
  domain_id: string; status: "DRAFT" | "VALIDATED"; source: "manual" | "llm" | "mock"; content_hash: string;
  updated_by: string; updated_at: string; strategy: StrategyConfig; issues?: ValidationIssue[] | null;
  report?: ValidationReport; provider?: string; provider_label?: string; assumptions?: string[]; warnings?: string[];
  market?: MarketContext | null; market_observations?: string[];
}

export interface VersionView {
  id: string; domain_id: string; number: number; status: "APPROVED" | "ACTIVE" | "RETIRED"; content_hash: string;
  created_by: string; approved_by: string; approved_at: string; published_at: string | null; retired_at: string | null; notes: string;
  strategy: StrategyConfig;
}

export interface StrategyDiff {
  added: Rule[]; removed: Rule[]; changed: { before: Rule; after: Rule; fields: string[] }[];
  constraints_before: Constraint[]; constraints_after: Constraint[]; constraints_changed: boolean;
  settings: { field: string; before: unknown; after: unknown }[]; has_changes: boolean;
}

// ---------- pricing ----------
export type SelectionStatus = "SELECTED" | "NOT_SELECTED" | "INELIGIBLE" | "BELOW_CONVERSION_FLOOR";
export interface Violation { constraint: string; message: string }
export interface DecisionCandidate {
  price: number; reasons: string[]; eligible: boolean; violations: Violation[]; p_buy: number | null;
  expected_revenue: number | null; expected_profit: number | null; selection_status: SelectionStatus; explanation: string;
}
export interface AppliedRule { rule_id: string; name: string; stage?: string | null; before: number; after: number; delta: number; reason: string }
export interface ConstraintEval { constraint: string; value: number | null; kind: "lower" | "upper" | "step"; bound: number | null; passed: boolean; message: string }
export interface ModelInfo { name: string; version: string; kind: "ml" | "simulated"; training_data: "synthetic" | "n/a"; artifact_sha256?: string | null }

export type Reaction = "DELIGHTED" | "ACCEPTS" | "HESITANT" | "REJECTS";
export interface PersonaReaction {
  persona_id: string; name: string; description: string; preferences: string[]; price: number; buy_share: number; defer_share: number;
  reject_share: number; reaction: Reaction; main_concern: string; concern_driver: string; explanation: string; suggestion: string | null;
  source: string; simulated: true;
}
export interface ValidationCheck { key: string; passed: boolean | null; available: boolean; severity: "ok" | "review" | "fail"; summary: string; details: Record<string, unknown> }
export type ShadowStatus = "PASS" | "FAIL" | "REVIEW_REQUIRED" | "ERROR" | "SKIPPED" | "PENDING";
export interface ShadowValidation {
  status: ShadowStatus; simulated_conversion: number | null; ci: [number, number] | null; model_disagreement_pp: number | null;
  checks: ValidationCheck[]; scenarios: Record<string, { applicable?: boolean; conversion_rate?: number; reason?: string; description?: string }>;
  feedback: Record<string, unknown> | null; personas: PersonaReaction[][] | null; population_size: number; seed: number;
  module_version: string; error: string | null; reason: string | null; advisory_only: true;
}

export interface PricingDecision {
  decision_id: string; domain_id: string; product_id: string; strategy_version_id: string; version_number: number | null;
  status: "OK" | "NO_VALID_PRICE"; status_reason: string | null; final_price: number | null; currency: string; unit_label: string;
  unit_prefix: string; unit_suffix: string; decimals: number; objective: string; reference_price: number; anchor_price: number;
  bounds: { lower: number | null; upper: number | null; lower_by: string[]; upper_by: string[]; conflict: boolean };
  applied_rules: AppliedRule[]; overridden_rules: { rule_id: string; reason: string; overridden_by: string | null }[];
  skipped_rules: { rule_id: string; reason: string }[]; constraints_evaluated: ConstraintEval[]; candidates: DecisionCandidate[];
  persona_summary: PersonaReaction[] | null; shadow_validation: ShadowValidation | null;
  explanation: { business: string[]; customer: string[] };
  provenance: { optimizer: ModelInfo | null; personas: string | null; llm_generated_strategy: boolean; simulated: boolean };
  signals: Record<string, number>; context_snapshot: Record<string, Scalar>;
  request: { domain_id: string; product_id: string; customer_id: string | null; attributes: Record<string, Scalar>; as_of: string };
  warnings: string[]; created_at: string | null;
}

export interface DecisionSummary {
  decision_id: string; domain_id: string; product_id: string; version_id: string; status: string; final_price: number | null;
  currency: string; decimals?: number; unit_prefix?: string; unit_suffix?: string; created_at: string; created_by: string; validation_status: ShadowStatus | null;
}

export interface CustomerQuote {
  decision_id: string; status: "OK" | "NO_VALID_PRICE"; final_price: number | null; currency: string; unit_label: string; unit_prefix: string; unit_suffix: string; decimals: number;
  factors: string[]; alternatives: { price: number; label: string }[] | null; notice: string;
}

export interface CatalogEntry {
  domain_id: string; name: string; description: string; currency: string; unit_label: string; product_noun: string;
  products: { id: string; name: string }[]; inputs: Attribute[];
}

export interface SimulationResult {
  simulation_id: string; version_id: string; content_hash: string; validation: ValidationReport;
  results: { label: string; decision: PricingDecision | null; error: { message: string; details?: string[] | null } | null }[];
}

export interface AuditEntry {
  id: number; at: string; actor: string; role: string; operation: string; resource_type: string; resource_id: string | null;
  version: string | null; outcome: string; details: Record<string, unknown>;
}

export interface Customer { id: string; domain_id: string | null; name: string; profile: Record<string, Scalar> }

export interface PriceObservation { price: number; title: string; source: string; link: string; kind: "shopping" | "web" }
export interface ProductMarket {
  product_id: string; name: string; base_price: number; query: string; observations: PriceObservation[]; n: number;
  min: number | null; median: number | null; max: number | null; confidence: "none" | "low" | "medium" | "high";
  position: "below_market" | "in_line" | "above_market" | null; base_vs_median_pct: number | null;
}
export interface MarketContext {
  provider: string; region: string; currency: string; queried_at: string; products: ProductMarket[];
  insights: { title: string; snippet: string; source: string; link: string }[]; searches_used: number; cache_hits: number;
  warnings: string[]; disclaimer: string;
}

export interface Health {
  status: string;
  llm: { provider: string; model: string | null; mock: boolean };
  optimizer: { name: string; version: string; kind: "ml" | "simulated"; training_data: string; fallback_reason: string | null };
  market: { provider: string | null; enabled: boolean; max_products: number };
  validation: { mode: string; module: string; advisory_only: boolean; persona_text: string };
  auth: string;
}
