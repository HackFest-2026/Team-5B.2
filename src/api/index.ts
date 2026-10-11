// Typed client for every backend endpoint used by the portals.
import type {
  AuditEntry,
  CatalogEntry,
  Customer,
  CustomerQuote,
  DecisionSummary,
  DomainConfig,
  DomainSummary,
  DraftView,
  Health,
  MarketContext,
  Persona,
  PricingDecision,
  Scalar,
  ShadowValidation,
  SimulationResult,
  StrategyConfig,
  StrategyDiff,
  VersionView,
} from "@/types";
import { request } from "./http";

export { ApiError } from "./http";
export { invalidate, useQuery } from "./query";

const enc = encodeURIComponent;
const qs = (p: Record<string, string | number | undefined | null>) => {
  const s = Object.entries(p).filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => `${k}=${enc(String(v))}`).join("&");
  return s ? `?${s}` : "";
};

export const api = {
  health: () => request<Health>("GET", "/health"),

  domains: () => request<DomainSummary[]>("GET", "/domains"),
  domain: (id: string) => request<DomainConfig>("GET", `/domains/${enc(id)}`),
  createDomain: (payload: unknown) => request<DomainConfig>("POST", "/domains", payload),
  updateDomain: (id: string, domain: DomainConfig) => request<DomainConfig>("PUT", `/domains/${enc(id)}`, domain),
  suggestDomain: (description: string) => request<{ domain: DomainConfig; provider: string }>("POST", "/domains/suggest", { description }),

  draft: (id: string) => request<DraftView>("GET", `/domains/${enc(id)}/strategy`),
  saveDraft: (id: string, strategy: StrategyConfig) => request<DraftView>("PUT", `/domains/${enc(id)}/strategy`, strategy),
  generate: (id: string, description: string, use_market_research = true) =>
    request<DraftView>("POST", `/domains/${enc(id)}/generate-strategy`, { description, use_market_research }),
  marketResearch: (id: string) => request<MarketContext>("POST", `/domains/${enc(id)}/market-research`),
  validate: (id: string) => request<DraftView>("POST", `/domains/${enc(id)}/validate`),
  simulate: (id: string, body: { version_id?: string | null; as_of: string; include_personas?: boolean; scenarios: { label: string; product_id: string; customer_id?: string | null; attributes: Record<string, Scalar> }[] }) =>
    request<SimulationResult>("POST", `/domains/${enc(id)}/simulate`, body),

  approve: (id: string, notes = "") => request<VersionView>("POST", `/domains/${enc(id)}/approve`, { notes }),
  publish: (id: string, version_id: string) => request<VersionView>("POST", `/domains/${enc(id)}/publish`, { version_id }),
  rollback: (id: string, version_id: string) => request<VersionView>("POST", `/domains/${enc(id)}/rollback`, { version_id }),
  versions: (id: string) => request<VersionView[]>("GET", `/domains/${enc(id)}/versions`),
  diff: (id: string, a: string, b: string) => request<StrategyDiff>("GET", `/domains/${enc(id)}/versions/${enc(a)}/diff/${enc(b)}`),

  quote: (body: { domain_id: string; product_id: string; customer_id?: string | null; attributes: Record<string, Scalar>; as_of?: string; include_personas?: boolean }) =>
    request<PricingDecision>("POST", "/pricing/quote", body),
  decisions: (p: { domain_id?: string; status?: string; limit?: number } = {}) => request<DecisionSummary[]>("GET", `/pricing/decisions${qs(p)}`),
  decision: (id: string) => request<PricingDecision>("GET", `/pricing/decisions/${enc(id)}`),
  decisionValidation: (id: string) => request<ShadowValidation>("GET", `/pricing/decisions/${enc(id)}/validation`),

  auditLogs: (p: { resource_type?: string; resource_id?: string; actor?: string; operation?: string; limit?: number } = {}) =>
    request<AuditEntry[]>("GET", `/audit-logs${qs(p)}`),
  customers: (domain_id?: string) => request<Customer[]>("GET", `/customers${qs({ domain_id })}`),
  personaDefaults: () => request<Persona[]>("GET", "/personas/defaults"),
  reseed: () => request<{ seeded: string[] }>("POST", "/admin/reseed"),

  // customer portal (always sent with the customer role)
  catalog: () => request<CatalogEntry[]>("GET", "/catalog", undefined, "customer"),
  customerQuote: (body: { domain_id: string; product_id: string; attributes: Record<string, Scalar>; as_of?: string }) =>
    request<CustomerQuote>("POST", "/pricing/quote", body, "customer"),
};
