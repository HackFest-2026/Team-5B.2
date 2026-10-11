import { CheckCircle2, CircleSlash, Info, XCircle } from "lucide-react";
import { CartesianGrid, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from "recharts";
import type { DecisionCandidate, PricingDecision } from "@/types";
import { formatPercent, formatPrice } from "@/engine/format";
import { cn } from "@/lib/utils";
import { Badge, Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/primitives";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SimulatedBadge, StatusPill } from "@/components/common";

const STATUS_LABEL: Record<DecisionCandidate["selection_status"], string> = {
  SELECTED: "Selected", NOT_SELECTED: "Eligible", INELIGIBLE: "Rejected", BELOW_CONVERSION_FLOOR: "Below conversion floor",
};

export function DecisionSummaryCard({ d }: { d: PricingDecision }) {
  const money = (v: number | null) => formatPrice(v, d);
  const opt = d.provenance.optimizer;
  return (
    <Card>
      <CardContent className="grid gap-4 p-5 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <div className="text-xs text-muted-foreground">Final price</div>
          <div className="tabular text-3xl font-semibold" data-testid="final-price">{d.final_price === null ? "No valid price" : formatPrice(d.final_price, d, true)}</div>
          <div className="text-xs text-muted-foreground">{d.unit_label}</div>
        </div>
        <div className="space-y-1 text-sm">
          <div className="text-xs text-muted-foreground">Status</div>
          <StatusPill status={d.status} />
          {d.status_reason && <div className="text-xs text-destructive">{d.status_reason}</div>}
        </div>
        <div className="space-y-1 text-sm">
          <div className="text-xs text-muted-foreground">Reference → anchor</div>
          <div className="tabular">{money(d.reference_price)} → {money(d.anchor_price)}</div>
          <div className="text-xs text-muted-foreground">Bounds {d.bounds.lower === null ? "—" : money(d.bounds.lower)} … {d.bounds.upper === null ? "∞" : money(d.bounds.upper)}</div>
        </div>
        <div className="space-y-1 text-sm">
          <div className="text-xs text-muted-foreground">Provenance</div>
          <div className="flex flex-wrap gap-1">
            <Badge variant="outline">{d.strategy_version_id}</Badge>
            {opt && <Badge variant={opt.kind === "ml" ? "info" : "warning"}>{opt.name}</Badge>}
            {d.provenance.simulated && <SimulatedBadge label={opt?.training_data === "synthetic" ? "synthetic-trained" : "simulated"} />}
          </div>
          <div className="truncate text-[11px] text-muted-foreground" title={d.decision_id}>{d.decision_id}</div>
        </div>
      </CardContent>
    </Card>
  );
}

export function CandidateTable({ d }: { d: PricingDecision }) {
  const money = (v: number | null) => formatPrice(v, d);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">Candidate prices <SimulatedBadge label="P(buy) is a model estimate" /></CardTitle>
        <CardDescription>
          Every candidate is checked against the hard constraints first. The optimizer's P(buy) gives expected profit, and the engine picks the highest
          {d.objective === "max_expected_revenue" ? " expected revenue" : " expected profit"} among eligible candidates.
        </CardDescription>
      </CardHeader>
      <CardContent className="p-0 pb-2">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Price</TableHead>
              <TableHead>Why this candidate</TableHead>
              <TableHead>Eligibility</TableHead>
              <TableHead className="text-right">P(buy)</TableHead>
              <TableHead className="text-right">Exp. revenue</TableHead>
              <TableHead className="text-right">Exp. profit</TableHead>
              <TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {d.candidates.map((c) => (
              <TableRow key={c.price} className={cn(c.selection_status === "SELECTED" && "bg-success/10 hover:bg-success/15", !c.eligible && "text-muted-foreground")} data-testid="candidate-row">
                <TableCell className="tabular font-medium">{money(c.price)}</TableCell>
                <TableCell className="max-w-[220px] text-xs">{c.reasons.join(" · ")}</TableCell>
                <TableCell className="max-w-[280px] text-xs">
                  {c.eligible ? (
                    <span className="inline-flex items-center gap-1 text-success"><CheckCircle2 className="h-3.5 w-3.5" /> passes all constraints</span>
                  ) : (
                    <span className="inline-flex items-start gap-1 text-destructive"><XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {c.violations.map((v) => v.message).join("; ")}</span>
                  )}
                </TableCell>
                <TableCell className="tabular text-right">{formatPercent(c.p_buy, 1)}</TableCell>
                <TableCell className="tabular text-right">{money(c.expected_revenue)}</TableCell>
                <TableCell className="tabular text-right font-medium">{money(c.expected_profit)}</TableCell>
                <TableCell>
                  <Badge variant={c.selection_status === "SELECTED" ? "success" : c.selection_status === "NOT_SELECTED" ? "secondary" : c.selection_status === "INELIGIBLE" ? "destructive" : "warning"}>
                    {STATUS_LABEL[c.selection_status]}
                  </Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

export function ProfitChart({ d }: { d: PricingDecision }) {
  const pts = d.candidates.filter((c) => c.expected_profit !== null).map((c) => ({
    price: c.price, profit: c.expected_profit, pbuy: c.p_buy !== null ? +(c.p_buy * 100).toFixed(1) : null, eligible: c.eligible, selected: c.selection_status === "SELECTED",
  }));
  if (pts.length < 2) return null;
  const sel = pts.find((p) => p.selected);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Price vs expected profit</CardTitle>
        <CardDescription>Cheaper prices sell more often but earn less per sale. The selected price is where P(buy) × margin peaks. Grey dots are rejected by a hard constraint.</CardDescription>
      </CardHeader>
      <CardContent className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={pts} margin={{ left: 8, right: 16, top: 8, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis dataKey="price" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => formatPrice(v, { ...d, decimals: d.decimals > 2 ? 2 : d.decimals })} fontSize={11} />
            <YAxis yAxisId="p" tickFormatter={(v) => formatPrice(v, { ...d, decimals: d.decimals > 2 ? 2 : d.decimals })} fontSize={11} width={70} />
            <YAxis yAxisId="b" orientation="right" unit="%" fontSize={11} width={40} />
            <RTooltip formatter={(v: number, n: string) => (n === "profit" ? formatPrice(v, d) : `${v}%`)} labelFormatter={(v) => `Price ${formatPrice(v as number, d)}`} />
            <Line yAxisId="p" dataKey="profit" name="profit" stroke="hsl(var(--primary))" strokeWidth={2}
              dot={(props: { cx?: number; cy?: number; payload?: { eligible: boolean; price: number } }) => (
                <circle key={props.payload?.price} cx={props.cx} cy={props.cy} r={4} fill={props.payload?.eligible ? "hsl(var(--primary))" : "hsl(var(--muted-foreground))"} />
              )} />
            <Line yAxisId="b" dataKey="pbuy" name="P(buy)" stroke="hsl(var(--warning))" strokeDasharray="4 3" dot={false} />
            {sel && <ReferenceDot yAxisId="p" x={sel.price} y={sel.profit ?? 0} r={7} fill="hsl(var(--success))" stroke="white" />}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

export function RulesAndConstraints({ d }: { d: PricingDecision }) {
  const money = (v: number) => formatPrice(v, d);
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader><CardTitle className="text-base">Rules</CardTitle></CardHeader>
        <CardContent className="space-y-2 text-sm">
          {d.applied_rules.length === 0 && <div className="text-muted-foreground">No rule changed the price.</div>}
          {d.applied_rules.map((r, i) => (
            <div key={i} className="flex items-start justify-between gap-3 rounded-md border p-2">
              <div>
                <div className="font-medium">{r.name} <span className="text-xs text-muted-foreground">({r.rule_id}{r.stage ? ` · ${r.stage}` : ""})</span></div>
                <div className="text-xs text-muted-foreground">{r.reason}</div>
              </div>
              <div className="tabular whitespace-nowrap text-xs">{money(r.before)} → <b>{money(r.after)}</b></div>
            </div>
          ))}
          {d.overridden_rules.map((r) => (
            <div key={r.rule_id} className="flex items-start gap-2 rounded-md border border-dashed p-2 text-xs text-muted-foreground">
              <CircleSlash className="mt-0.5 h-3.5 w-3.5" /> <span><b>{r.rule_id}</b> matched but was overridden: {r.reason}</span>
            </div>
          ))}
          {d.skipped_rules.length > 0 && (
            <details className="text-xs text-muted-foreground">
              <summary className="cursor-pointer">{d.skipped_rules.length} rule(s) did not apply</summary>
              <ul className="mt-1 space-y-0.5 pl-4">{d.skipped_rules.map((r) => <li key={r.rule_id}><b>{r.rule_id}</b>: {r.reason}</li>)}</ul>
            </details>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle className="text-base">Hard constraints (final price)</CardTitle></CardHeader>
        <CardContent className="space-y-1.5 text-sm">
          {d.constraints_evaluated.length === 0 && <div className="text-muted-foreground">No final price, so nothing to check.</div>}
          {d.constraints_evaluated.map((c, i) => (
            <div key={i} className="flex items-start gap-2 text-xs">
              {c.passed ? <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 text-success" /> : <XCircle className="mt-0.5 h-3.5 w-3.5 text-destructive" />}
              <span>{c.message}</span>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

export function ExplanationCard({ d }: { d: PricingDecision }) {
  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Explanation</CardTitle></CardHeader>
      <CardContent className="grid gap-4 text-sm lg:grid-cols-2">
        <div>
          <div className="mb-1 text-xs font-medium uppercase text-muted-foreground">Business (internal)</div>
          <ol className="list-decimal space-y-1 pl-5">{d.explanation.business.map((l, i) => <li key={i}>{l}</li>)}</ol>
        </div>
        <div>
          <div className="mb-1 text-xs font-medium uppercase text-muted-foreground">Customer-safe</div>
          <ul className="list-disc space-y-1 pl-5">{d.explanation.customer.map((l, i) => <li key={i}>{l}</li>)}</ul>
          {d.warnings.length > 0 && (
            <div className="mt-3 rounded-md bg-muted p-2 text-xs">
              <div className="mb-1 flex items-center gap-1 font-medium"><Info className="h-3.5 w-3.5" /> Engine notes</div>
              <ul className="space-y-0.5">{d.warnings.map((w, i) => <li key={i} className="break-words">{w}</li>)}</ul>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
