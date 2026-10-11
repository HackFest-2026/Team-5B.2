import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowDown, ArrowUp, CheckCircle2, Pencil, Plus, Save, ShieldCheck, Sparkles, Trash2, Wand2 } from "lucide-react";
import { toast } from "sonner";
import { api, invalidate, useQuery } from "@/api";
import { useCurrentDomain } from "@/api/hooks";
import type { Constraint, ConstraintType, DraftView, Rule, StrategyConfig, ValidationIssue } from "@/types";
import { CONSTRAINT_TYPES } from "@/types";
import { actionText, conditionText, CONSTRAINT_HELP } from "@/engine/readback";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge, Input, Switch, Table, TableBody, TableCell, TableHead, TableHeader, TableRow, Textarea } from "@/components/ui/primitives";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/overlays";
import { AsyncButton, EmptyState, ErrorBox, Field, Loading, PageHeader, StatusPill, toastError } from "@/components/common";
import { RuleDialog } from "./RuleDialog";
import { MarketPanel } from "@/components/pricing/MarketPanel";

function IssueList({ issues, onSelectRule }: { issues: ValidationIssue[]; onSelectRule: (id: string) => void }) {
  if (!issues.length) return null;
  return (
    <div className="space-y-1.5" data-testid="validation-issues">
      {issues.map((i) => (
        <div key={i.id} className={cn("rounded-md border p-2 text-xs", i.severity === "error" ? "border-destructive/40 bg-destructive/5" : i.severity === "warning" ? "border-warning/40 bg-warning/10" : "bg-muted/40")}>
          <Badge variant={i.severity === "error" ? "destructive" : i.severity === "warning" ? "warning" : "secondary"} className="mr-2">{i.severity}</Badge>
          {i.message}
          {i.rule_ids.map((r) => <button key={r} className="ml-2 underline" onClick={() => onSelectRule(r)}>{r}</button>)}
          {i.path && <span className="ml-2 text-muted-foreground">({i.path})</span>}
        </div>
      ))}
    </div>
  );
}

export default function Strategy() {
  const { id, domain } = useCurrentDomain();
  const draftQ = useQuery(id ? `draft:${id}` : null, () => api.draft(id!));
  const [s, setS] = useState<StrategyConfig | null>(null);
  const [dirty, setDirty] = useState(false);
  const [issues, setIssues] = useState<ValidationIssue[]>([]);
  const [gen, setGen] = useState<DraftView | null>(null);
  const [objective, setObjective] = useState("");
  const [editing, setEditing] = useState<Rule | null>(null);
  const [highlight, setHighlight] = useState<string | null>(null);
  const health = useQuery("health", api.health);
  const marketOn = !!health.data?.market?.enabled;
  const [useMarket, setUseMarket] = useState(true);

  useEffect(() => {
    if (draftQ.data && !dirty) {
      setS(structuredClone(draftQ.data.strategy));
      setObjective((o) => o || draftQ.data!.strategy.objective_text);
      if (draftQ.data.issues) setIssues(draftQ.data.issues);
    }
  }, [draftQ.data, dirty]);
  const lastId = useRef(id);
  useEffect(() => {
    if (lastId.current === id) return; // only reset when the user switches domain
    lastId.current = id;
    setDirty(false); setGen(null); setObjective(""); setIssues([]);
  }, [id]);

  const stages = domain?.pipeline ?? [];
  const sorted = useMemo(() => [...(s?.rules ?? [])].sort((a, b) => stages.indexOf(a.stage) - stages.indexOf(b.stage) || a.priority - b.priority || a.id.localeCompare(b.id)), [s, stages]);
  if (!id) return <EmptyState title="No domain" description="Create a domain first." />;
  if (draftQ.error) return <ErrorBox error={draftQ.error} onRetry={draftQ.reload} />;
  if (!s || !domain || !draftQ.data) return <Loading />;

  const update = (patch: Partial<StrategyConfig>) => { setS({ ...s, ...patch }); setDirty(true); };
  const setRules = (rules: Rule[]) => update({ rules });
  const save = async () => {
    const r = await api.saveDraft(id, { ...s, objective_text: objective });
    setIssues(r.issues ?? []);
    setDirty(false);
    invalidate(`draft:${id}`);
    invalidate("domains");
    return r;
  };
  const move = (rule: Rule, dir: -1 | 1) => {
    const same = sorted.filter((r) => r.stage === rule.stage);
    const idx = same.findIndex((r) => r.id === rule.id);
    const other = same[idx + dir];
    if (!other) return;
    setRules(s.rules.map((r) => (r.id === rule.id ? { ...r, priority: other.priority } : r.id === other.id ? { ...r, priority: rule.priority === other.priority ? rule.priority + dir : rule.priority } : r)));
  };
  const newRule = (): Rule => ({
    id: `R-${Date.now().toString(36).toUpperCase()}`, name: "New rule", condition: { all: [] }, action: { type: "percent", value: 0 },
    priority: (Math.max(0, ...s.rules.map((r) => r.priority)) || 0) + 10, stage: stages[0], enabled: true, customer_message: null,
  });
  const draft = draftQ.data;

  return (
    <div className="space-y-4">
      <PageHeader icon={Sparkles} title="Strategy Studio" description={`Describe the objective, generate a draft, review every rule, then validate. Domain: ${domain.name}.`}
        actions={<>
          <StatusPill status={dirty ? "UNSAVED" : draft.status} />
          <Badge variant={draft.source === "manual" ? "secondary" : draft.source === "mock" ? "warning" : "info"}>source: {draft.source}</Badge>
          <AsyncButton variant="outline" disabled={!dirty} onClick={async () => { try { await save(); toast.success("Draft saved"); } catch (e) { toastError(e); } }}><Save /> Save draft</AsyncButton>
          <AsyncButton onClick={async () => {
            try {
              if (dirty) await save();
              const r = await api.validate(id);
              setIssues(r.report?.issues ?? []);
              invalidate(`draft:${id}`); invalidate("domains");
              r.report?.ok ? toast.success("Strategy validated. A manager can approve it in Versions & Approvals.") : toast.error(`${r.report?.counts.error} validation error(s)`);
            } catch (e) { toastError(e); }
          }}><ShieldCheck /> Validate</AsyncButton>
        </>} />

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base"><Wand2 className="h-4 w-4" /> Generate from a business objective</CardTitle>
          <CardDescription>The generator returns configuration only (never code or final prices). Its output is validated before it replaces the draft.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea className="h-24" value={objective} onChange={(e) => setObjective(e.target.value)} placeholder="e.g. Increase prices when demand is high and utilization is above 80%. Discount when demand is low, but never below a 25% margin. Limit any increase to 20%." />
          <div className="flex flex-wrap items-center gap-2">
            <AsyncButton disabled={!objective.trim()} onClick={async () => {
              try {
                const r = await api.generate(id, objective, marketOn && useMarket);
                setGen(r); setDirty(false); invalidate(`draft:${id}`); invalidate("domains");
                setS(structuredClone(r.strategy)); setIssues([]);
                toast.success(`Draft generated by ${r.provider_label}. Review the rules below.`);
              } catch (e) { toastError(e, "Generation failed"); }
            }}><Sparkles /> Generate draft</AsyncButton>
            {marketOn && (
              <label className="flex items-center gap-2 text-sm">
                <Switch checked={useMarket} onCheckedChange={setUseMarket} /> Research the market first ({health.data?.market.provider} web search)
              </label>
            )}
            {dirty && <span className="text-xs text-warning-foreground">Unsaved edits will be replaced.</span>}
          </div>
          {gen && (
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-md border p-3 text-xs">
                <Badge variant={gen.provider === "mock" ? "warning" : "info"}>{gen.provider_label}</Badge>
                <div className="mt-2 font-medium">Assumptions</div>
                <ul className="list-disc pl-4">{(gen.assumptions ?? []).map((a, i) => <li key={i}>{a}</li>)}{!gen.assumptions?.length && <li>None</li>}</ul>
              </div>
              <div className="rounded-md border p-3 text-xs">
                <div className="font-medium">Warnings</div>
                <ul className="list-disc pl-4">{(gen.warnings ?? []).map((a, i) => <li key={i}>{a}</li>)}{!gen.warnings?.length && <li>None</li>}</ul>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {gen?.market && (
        <MarketPanel market={gen.market} observations={gen.market_observations} domain={domain}
          onApplyCompetitorPrices={async () => {
            const attr = domain.attributes.find((a) => a.semantic === "competitorPrice" && a.source === "inventory");
            if (!attr || !gen.market) return;
            const medians = Object.fromEntries(gen.market.products.filter((p) => p.median !== null).map((p) => [p.product_id, p.median]));
            const updated = { ...domain, products: domain.products.map((p) => (p.id in medians ? { ...p, inventory: { ...p.inventory, [attr.key]: medians[p.id] } } : p)) };
            try {
              await api.updateDomain(domain.id, updated);
              invalidate("domain");
              toast.success(`${attr.label} updated for ${Object.keys(medians).length} item(s). Re-validate the draft before approval.`);
            } catch (e) { toastError(e); }
          }} />
      )}

      {issues.length > 0 && (
        <Card><CardHeader><CardTitle className="text-base">Validation</CardTitle></CardHeader><CardContent><IssueList issues={issues} onSelectRule={setHighlight} /></CardContent></Card>
      )}
      {draft.status === "VALIDATED" && !dirty && (
        <div className="flex items-center gap-2 rounded-md border border-success/40 bg-success/10 p-3 text-sm text-success">
          <CheckCircle2 className="h-4 w-4" /> This draft passed validation. <Link to="/business/simulator" className="underline">Simulate it</Link> or <Link to="/business/versions" className="underline">send it for approval</Link>.
        </div>
      )}

      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <div><CardTitle className="text-base">Rules ({s.rules.length})</CardTitle><CardDescription>Applied in stage order: {stages.join(" → ")}</CardDescription></div>
          <Button size="sm" onClick={() => setEditing(newRule())}><Plus /> Add rule</Button>
        </CardHeader>
        <CardContent className="p-0 pb-2">
          <Table>
            <TableHeader><TableRow><TableHead>On</TableHead><TableHead>Stage</TableHead><TableHead>Prio</TableHead><TableHead>Rule</TableHead><TableHead>Action</TableHead><TableHead>Customer message</TableHead><TableHead /></TableRow></TableHeader>
            <TableBody>
              {sorted.map((r) => (
                <TableRow key={r.id} className={cn(!r.enabled && "opacity-50", highlight === r.id && "bg-warning/15")} data-testid="rule-row">
                  <TableCell><Switch checked={r.enabled} onCheckedChange={(c) => setRules(s.rules.map((x) => (x.id === r.id ? { ...x, enabled: c } : x)))} aria-label={`Enable ${r.name}`} /></TableCell>
                  <TableCell><Badge variant="outline">{r.stage}</Badge></TableCell>
                  <TableCell className="tabular">{r.priority}</TableCell>
                  <TableCell><div className="font-medium">{r.name} <span className="text-xs text-muted-foreground">{r.id}</span></div><div className="text-xs text-muted-foreground">If {conditionText(r.condition, domain.attributes)}</div></TableCell>
                  <TableCell className="tabular whitespace-nowrap font-medium">{actionText(r.action, domain.output_unit)}</TableCell>
                  <TableCell className="max-w-[200px] text-xs">{r.customer_message ?? <span className="text-muted-foreground">—</span>}</TableCell>
                  <TableCell className="whitespace-nowrap text-right">
                    <Button size="icon" variant="ghost" aria-label="Move up" onClick={() => move(r, -1)}><ArrowUp /></Button>
                    <Button size="icon" variant="ghost" aria-label="Move down" onClick={() => move(r, 1)}><ArrowDown /></Button>
                    <Button size="icon" variant="ghost" aria-label={`Edit ${r.name}`} onClick={() => setEditing(r)}><Pencil /></Button>
                    <Button size="icon" variant="ghost" aria-label={`Delete ${r.name}`} onClick={() => setRules(s.rules.filter((x) => x.id !== r.id))}><Trash2 /></Button>
                  </TableCell>
                </TableRow>
              ))}
              {!s.rules.length && <TableRow><TableCell colSpan={7} className="text-center text-sm text-muted-foreground">No rules yet. Generate a draft or add one.</TableCell></TableRow>}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <div><CardTitle className="text-base">Hard constraints</CardTitle><CardDescription>Authoritative. Price &gt; 0 and price ≥ unit cost always apply.</CardDescription></div>
            <Button size="sm" variant="outline" onClick={() => update({ constraints: [...s.constraints, { type: "max_increase", value: 20 }] })}><Plus /> Constraint</Button>
          </CardHeader>
          <CardContent className="space-y-2">
            {s.constraints.map((c, i) => (
              <div key={i} className="flex items-center gap-2">
                <Select value={c.type} onValueChange={(t) => update({ constraints: s.constraints.map((x, j) => (j === i ? { ...x, type: t as ConstraintType } : x)) })}>
                  <SelectTrigger className="h-8 w-44" aria-label="Constraint type"><SelectValue /></SelectTrigger>
                  <SelectContent>{CONSTRAINT_TYPES.map((t) => <SelectItem key={t} value={t}>{t}</SelectItem>)}</SelectContent>
                </Select>
                <Input className="h-8 w-28" type="number" value={c.value} onChange={(e) => update({ constraints: s.constraints.map((x, j): Constraint => (j === i ? { ...x, value: Number(e.target.value) } : x)) })} />
                <span className="flex-1 text-xs text-muted-foreground">{CONSTRAINT_HELP[c.type]}</span>
                <Button size="icon" variant="ghost" aria-label="Remove constraint" onClick={() => update({ constraints: s.constraints.filter((_, j) => j !== i) })}><Trash2 /></Button>
              </div>
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="text-base">Conflict, candidate and optimizer policy</CardTitle></CardHeader>
          <CardContent className="grid gap-3 sm:grid-cols-2">
            <Field label="When several rules match">
              <Select value={s.conflict_policy.mode} onValueChange={(m) => update({ conflict_policy: { ...s.conflict_policy, mode: m as StrategyConfig["conflict_policy"]["mode"] } })}>
                <SelectTrigger aria-label="Conflict mode"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="stack">Stack: apply all</SelectItem>
                  <SelectItem value="first_match_per_stage">First match per stage</SelectItem>
                  <SelectItem value="highest_priority_wins">Highest priority wins</SelectItem>
                </SelectContent>
              </Select>
            </Field>
            <Field label="Percent combination">
              <Select value={s.conflict_policy.combine} onValueChange={(m) => update({ conflict_policy: { ...s.conflict_policy, combine: m as "additive" | "multiplicative" } })}>
                <SelectTrigger aria-label="Combine"><SelectValue /></SelectTrigger>
                <SelectContent><SelectItem value="multiplicative">Multiplicative (compound)</SelectItem><SelectItem value="additive">Additive (vs base)</SelectItem></SelectContent>
              </Select>
            </Field>
            <Field label="Max total rule adjustment (±%)">
              <Input type="number" value={s.conflict_policy.max_total_adjustment_pct ?? ""} placeholder="no cap" onChange={(e) => update({ conflict_policy: { ...s.conflict_policy, max_total_adjustment_pct: e.target.value === "" ? null : Number(e.target.value) } })} />
            </Field>
            <Field label="Candidate count">
              <Input type="number" min={1} max={15} value={s.candidate_policy.count} onChange={(e) => update({ candidate_policy: { ...s.candidate_policy, count: Number(e.target.value) } })} />
            </Field>
            <Field label="Candidate offsets around the anchor (%)" className="sm:col-span-2">
              <Input value={s.candidate_policy.offsets_pct.join(", ")} onChange={(e) => update({ candidate_policy: { ...s.candidate_policy, offsets_pct: e.target.value.split(",").map((x) => Number(x.trim())).filter((x) => !Number.isNaN(x)) } })} />
            </Field>
            <Field label="Objective">
              <Select value={s.optimizer_policy.objective} onValueChange={(o) => update({ optimizer_policy: { ...s.optimizer_policy, objective: o as "max_expected_profit" | "max_expected_revenue" } })}>
                <SelectTrigger aria-label="Objective"><SelectValue /></SelectTrigger>
                <SelectContent><SelectItem value="max_expected_profit">Max expected profit</SelectItem><SelectItem value="max_expected_revenue">Max expected revenue</SelectItem></SelectContent>
              </Select>
            </Field>
            <Field label="Conversion floor (min P(buy))">
              <Input type="number" step="0.01" min={0} max={1} value={s.optimizer_policy.conversion_floor} onChange={(e) => update({ optimizer_policy: { ...s.optimizer_policy, conversion_floor: Number(e.target.value) } })} />
            </Field>
            <label className="flex items-center gap-2 text-sm"><Switch checked={s.candidate_policy.include_base_price} onCheckedChange={(c) => update({ candidate_policy: { ...s.candidate_policy, include_base_price: c } })} /> Include base price</label>
            <label className="flex items-center gap-2 text-sm"><Switch checked={s.candidate_policy.include_bound_pins} onCheckedChange={(c) => update({ candidate_policy: { ...s.candidate_policy, include_bound_pins: c } })} /> Pin to nearest bound</label>
            <label className="flex items-center gap-2 text-sm"><Switch checked={s.candidate_policy.show_alternatives} onCheckedChange={(c) => update({ candidate_policy: { ...s.candidate_policy, show_alternatives: c } })} /> Show alternatives to customers</label>
          </CardContent>
        </Card>
      </div>

      <RuleDialog open={!!editing} rule={editing} attrs={domain.attributes} stages={stages} onClose={() => setEditing(null)}
        onSave={(r) => { setRules(s.rules.some((x) => x.id === r.id) ? s.rules.map((x) => (x.id === r.id ? r : x)) : [...s.rules, r]); setEditing(null); }} />
    </div>
  );
}
