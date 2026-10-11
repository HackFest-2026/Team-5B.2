import { useEffect, useMemo, useState } from "react";
import { Calculator, Play } from "lucide-react";
import { api, invalidate, useQuery } from "@/api";
import { useCurrentDomain } from "@/api/hooks";
import type { Attribute, Scalar, SimulationResult } from "@/types";
import { useUi } from "@/store/ui";
import { Button } from "@/components/ui/button";
import { Badge, Switch } from "@/components/ui/primitives";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/overlays";
import { AsyncButton, EmptyState, ErrorBox, Field, Loading, PageHeader, toastError } from "@/components/common";
import { AttributeInputs, cleanValues } from "@/components/pricing/AttributeInputs";
import { DecisionDetail } from "@/components/pricing";

const PRESSURE_SIGNALS = ["demandPressure", "capacityUtilization", "scarcity"];

/** Generic presets driven by semantic signals, so they work for any domain. */
function preset(attrs: Attribute[], level: number): Record<string, Scalar> {
  const out: Record<string, Scalar> = {};
  for (const a of attrs) {
    if (a.type !== "number" || !a.semantic || !PRESSURE_SIGNALS.includes(a.semantic) || a.source === "computed") continue;
    const lo = a.normalize?.min ?? a.min ?? 0;
    const hi = a.normalize?.max ?? a.max ?? 100;
    const t = a.normalize?.invert ? 1 - level : level;
    out[a.key] = Math.round((lo + t * (hi - lo)) * 100) / 100;
  }
  return out;
}

export default function Simulator() {
  const { id, domain } = useCurrentDomain();
  const asOf = useUi((s) => s.asOf);
  const versions = useQuery(id ? `versions:${id}` : null, () => api.versions(id!));
  const customers = useQuery(id ? `customers:${id}` : null, () => api.customers(id!));
  const [target, setTarget] = useState("draft");
  const [productId, setProductId] = useState("");
  const [customerId, setCustomerId] = useState("none");
  const [values, setValues] = useState<Record<string, Scalar>>({});
  const [personas, setPersonas] = useState(true);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [active, setActive] = useState("Custom");

  useEffect(() => {
    setProductId(domain?.products[0]?.id ?? "");
    setValues({}); setResult(null); setTarget("draft"); setCustomerId("none");
  }, [domain?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const inputs = useMemo(() => (domain?.attributes ?? []).filter((a) => a.source === "request" || a.source === "inventory"), [domain]);
  const product = domain?.products.find((p) => p.id === productId);
  if (!id) return <EmptyState title="No domain" />;
  if (!domain) return <Loading />;

  const run = async () => {
    try {
      const r = await api.simulate(id, {
        version_id: target === "draft" ? null : target, as_of: asOf, include_personas: personas,
        scenarios: [{ label: active, product_id: productId, customer_id: customerId === "none" ? null : customerId, attributes: cleanValues(values) }],
      });
      setResult(r);
      invalidate("audit");
    } catch (e) { toastError(e, "Simulation failed"); }
  };
  const res = result?.results[0];

  return (
    <div className="space-y-4">
      <PageHeader icon={Calculator} title="Simulator" description="Price a scenario against the draft or any version before it goes live. Nothing here is published or shown to customers." />
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Scenario</CardTitle>
          <CardDescription>Inventory values default to the {domain.product_noun.toLowerCase()}'s stored data; anything you type overrides them for this simulation only.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Field label="Strategy">
              <Select value={target} onValueChange={setTarget}>
                <SelectTrigger aria-label="Strategy version"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="draft">Current draft</SelectItem>
                  {(versions.data ?? []).map((v) => <SelectItem key={v.id} value={v.id}>{v.id} ({v.status.toLowerCase()})</SelectItem>)}
                </SelectContent>
              </Select>
            </Field>
            <Field label={domain.product_noun}>
              <Select value={productId} onValueChange={setProductId}>
                <SelectTrigger aria-label="Product"><SelectValue /></SelectTrigger>
                <SelectContent>{domain.products.map((p) => <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>)}</SelectContent>
              </Select>
            </Field>
            <Field label="Customer profile">
              <Select value={customerId} onValueChange={setCustomerId}>
                <SelectTrigger aria-label="Customer"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">Anonymous</SelectItem>
                  {(customers.data ?? []).map((c) => <SelectItem key={c.id} value={c.id}>{c.name}</SelectItem>)}
                </SelectContent>
              </Select>
            </Field>
            <Field label="Pricing date"><div className="flex h-9 items-center text-sm">{asOf} <span className="ml-2 text-xs text-muted-foreground">(top bar)</span></div></Field>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-muted-foreground">Presets:</span>
            {[["Custom", null], ["High demand", 0.92], ["Normal", 0.5], ["Low demand", 0.12]].map(([label, lvl]) => (
              <Button key={label as string} size="sm" variant={active === label ? "default" : "outline"} onClick={() => {
                setActive(label as string);
                if (lvl !== null) setValues((v) => ({ ...v, ...preset(domain.attributes, lvl as number) }));
              }}>{label}</Button>
            ))}
            <Button size="sm" variant="ghost" onClick={() => { setValues({}); setActive("Custom"); }}>Clear</Button>
          </div>
          <AttributeInputs attributes={inputs} values={values} onChange={(k, v) => { setValues({ ...values, [k]: v }); setActive("Custom"); }}
            placeholders={product?.inventory as Record<string, Scalar>} />
          <div className="flex flex-wrap items-center gap-4">
            <AsyncButton onClick={run} disabled={!productId}><Play /> Run simulation</AsyncButton>
            <label className="flex items-center gap-2 text-sm"><Switch checked={personas} onCheckedChange={setPersonas} /> Persona cohorts</label>
            {result && <Badge variant="outline">{result.version_id.startsWith("draft") ? "draft" : result.version_id} · {result.simulation_id}</Badge>}
          </div>
        </CardContent>
      </Card>
      {result && !result.validation.ok && (
        <Card><CardContent className="p-4 text-sm text-destructive">This strategy has {result.validation.counts.error} validation error(s); fix them in Strategy Studio before approval. Results are shown for exploration only.</CardContent></Card>
      )}
      {res?.error && <ErrorBox error={{ message: res.error.message, lines: res.error.details ?? [] }} />}
      {res?.decision && <DecisionDetail d={res.decision} />}
    </div>
  );
}
