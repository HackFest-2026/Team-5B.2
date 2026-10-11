import { AlertTriangle, CheckCircle2, MinusCircle, ShieldAlert, Users, XCircle } from "lucide-react";
import type { PersonaReaction, PricingDecision, Reaction, ShadowValidation } from "@/types";
import { formatPercent, formatPrice } from "@/engine/format";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/primitives";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/overlays";
import { ShadowBadge } from "@/components/common";

const CHECK_LABEL: Record<string, string> = {
  A_behaviour: "A · Behavioural consistency", B_heterogeneity: "B · Population variety", C_scenarios: "C · Stress scenarios",
  D_ranking: "D · Ranking stability", E_model_disagreement: "E · Model disagreement", F_ml_monotonicity: "F · ML price monotonicity",
};
const SCENARIO_LABEL: Record<string, string> = {
  baseline: "Baseline", budget_squeeze: "Budget squeeze (−25%)", competitor_undercut: "Competitor undercut (−20%)",
  high_urgency: "High urgency", perceived_value_drop: "Perceived value drop (−20%)",
};

function Banner() {
  return (
    <div className="flex items-start gap-2 rounded-md border border-warning/40 bg-warning/10 p-2.5 text-xs text-warning-foreground">
      <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
      <span><b>Simulated customer agents (shadow mode).</b> Advisory only: this never changes the selected price. Synthetic agents are not evidence about real customers.</span>
    </div>
  );
}

export function ValidationPanel({ v, unit }: { v: ShadowValidation | null; unit: { unit_prefix: string; unit_suffix: string; decimals: number } }) {
  if (!v) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 text-base">Customer-agent validation <ShadowBadge status={v.status} /></CardTitle>
        <CardDescription>{v.population_size ? `${v.population_size} simulated agents · seed ${v.seed}` : v.reason ?? ""}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <Banner />
        {v.status === "PENDING" && <div className="text-muted-foreground">Validation is running in the background…</div>}
        {v.status === "ERROR" && <div className="text-destructive">The simulator failed ({v.error}). The quote is unaffected.</div>}
        {v.status === "SKIPPED" && <div className="text-muted-foreground">{v.reason}</div>}
        {v.simulated_conversion !== null && (
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-md border p-3">
              <div className="text-xs text-muted-foreground">Simulated conversion</div>
              <div className="tabular text-xl font-semibold">{formatPercent(v.simulated_conversion, 1)}</div>
              {v.ci && <div className="text-[11px] text-muted-foreground">95% CI {formatPercent(v.ci[0], 1)} – {formatPercent(v.ci[1], 1)}</div>}
            </div>
            <div className="rounded-md border p-3">
              <div className="text-xs text-muted-foreground">Model disagreement</div>
              <div className="tabular text-xl font-semibold">{v.model_disagreement_pp?.toFixed(1)} pp</div>
              <div className="text-[11px] text-muted-foreground">ML vs agents (both synthetic)</div>
            </div>
            <div className="rounded-md border p-3">
              <div className="text-xs text-muted-foreground">Suggested action (advisory)</div>
              <div className="font-medium">{String(v.feedback?.suggested_action ?? "—").replace(/_/g, " ").toLowerCase()}</div>
              {v.feedback?.alternative_candidate_price != null && (
                <div className="text-[11px]">
                  Alternative {formatPrice(Number(v.feedback.alternative_candidate_price), unit)}{" "}
                  {v.feedback.alternative_candidate_eligible ? <Badge variant="success">eligible</Badge> : <Badge variant="destructive">violates a constraint</Badge>}
                </div>
              )}
            </div>
          </div>
        )}
        {v.checks.length > 0 && (
          <div className="divide-y rounded-md border">
            {v.checks.map((c) => (
              <div key={c.key} className="flex items-start gap-2 p-2.5 text-xs" data-testid="validation-check">
                {!c.available ? <MinusCircle className="mt-0.5 h-4 w-4 text-muted-foreground" />
                  : c.severity === "ok" ? <CheckCircle2 className="mt-0.5 h-4 w-4 text-success" />
                  : c.severity === "review" ? <AlertTriangle className="mt-0.5 h-4 w-4 text-warning" />
                  : <XCircle className="mt-0.5 h-4 w-4 text-destructive" />}
                <div><div className="font-medium">{CHECK_LABEL[c.key] ?? c.key}</div><div className="text-muted-foreground">{c.summary}</div></div>
              </div>
            ))}
          </div>
        )}
        {Object.keys(v.scenarios).length > 0 && (
          <div>
            <div className="mb-1 text-xs font-medium uppercase text-muted-foreground">Stress scenarios (same agents, one shock each)</div>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
              {Object.entries(v.scenarios).map(([k, s]) => (
                <div key={k} className="rounded-md border p-2 text-xs" title={s.description ?? s.reason}>
                  <div className="text-muted-foreground">{SCENARIO_LABEL[k] ?? k}</div>
                  <div className="tabular text-base font-semibold">{s.applicable === false ? "n/a" : formatPercent(s.conversion_rate ?? null, 1)}</div>
                  {s.applicable === false && <div className="text-[10px] text-muted-foreground">no competitor price</div>}
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

const REACTION_STYLE: Record<Reaction, string> = {
  DELIGHTED: "bg-success/20 text-success", ACCEPTS: "bg-success/10 text-success", HESITANT: "bg-warning/20 text-warning-foreground", REJECTS: "bg-destructive/15 text-destructive",
};

function Cell({ r, selected, d }: { r: PersonaReaction; selected: boolean; d: PricingDecision }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className={cn("rounded-md border p-1.5 text-center", selected && "ring-2 ring-success")}>
          <div className={cn("rounded px-1 text-[10px] font-semibold", REACTION_STYLE[r.reaction])}>{r.reaction.toLowerCase()}</div>
          <div className="mt-1 flex h-1.5 overflow-hidden rounded-full bg-muted" aria-label="buy / wait / walk away">
            <div className="bg-success" style={{ width: `${r.buy_share * 100}%` }} />
            <div className="bg-warning" style={{ width: `${r.defer_share * 100}%` }} />
            <div className="bg-destructive/70" style={{ width: `${r.reject_share * 100}%` }} />
          </div>
          <div className="tabular mt-0.5 text-[10px] text-muted-foreground">{formatPercent(r.buy_share)} buy</div>
        </div>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs text-xs">
        <div className="font-medium">{r.name} at {formatPrice(r.price, d)}</div>
        <div>{r.explanation}</div>
        <div className="mt-1"><b>Concern:</b> {r.main_concern}</div>
        {r.suggestion && <div><b>Suggestion:</b> {r.suggestion}</div>}
      </TooltipContent>
    </Tooltip>
  );
}

export function PersonaGrid({ d }: { d: PricingDecision }) {
  const grid = d.shadow_validation?.personas;
  if (!grid?.length) return null;
  const prices = grid[0].map((r) => r.price);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base"><Users className="h-4 w-4" /> Persona reactions across candidates</CardTitle>
        <CardDescription>Hover a cell for the persona's concern and a suggestion. The outlined column is the selected price.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <Banner />
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] border-separate border-spacing-1 text-xs">
            <thead>
              <tr>
                <th className="text-left font-medium text-muted-foreground">Persona</th>
                {prices.map((p) => <th key={p} className={cn("tabular font-medium", p === d.final_price ? "text-success" : "text-muted-foreground")}>{formatPrice(p, d)}</th>)}
              </tr>
            </thead>
            <tbody>
              {grid.map((row) => (
                <tr key={row[0].persona_id}>
                  <td className="pr-2 align-top"><div className="font-medium">{row[0].name}</div><div className="text-[10px] text-muted-foreground">{row[0].description}</div></td>
                  {row.map((r) => <td key={r.price}><Cell r={r} selected={r.price === d.final_price} d={d} /></td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {d.persona_summary && (
          <div className="grid gap-2 md:grid-cols-2">
            {d.persona_summary.filter((r) => r.concern_driver !== "none").map((r) => (
              <div key={r.persona_id} className="rounded-md border p-2 text-xs">
                <b>{r.name}:</b> “{r.main_concern}” {r.suggestion && <span className="text-muted-foreground">→ {r.suggestion}</span>}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

