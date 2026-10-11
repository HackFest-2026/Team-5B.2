import { useEffect, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import type { ActionType, Attribute, Condition, LeafCondition, Op, Rule, Scalar } from "@/types";
import { OPS } from "@/types";
import { Button } from "@/components/ui/button";
import { Input, Switch } from "@/components/ui/primitives";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/overlays";
import { Field } from "@/components/common";

const ACTIONS: { value: ActionType; label: string }[] = [
  { value: "percent", label: "Percent change (±%)" }, { value: "fixed", label: "Fixed change (±amount)" },
  { value: "set", label: "Target price" }, { value: "multiply", label: "Multiply" }, { value: "none", label: "No adjustment" },
  { value: "formula", label: "Formula (safe expression)" },
];

function parseValue(raw: string, attr?: Attribute): Scalar {
  if (attr?.type === "number") return raw === "" ? null : Number(raw);
  if (attr?.type === "boolean") return raw === "true";
  return raw;
}

function LeafEditor({ leaf, attrs, onChange }: { leaf: LeafCondition; attrs: Attribute[]; onChange: (l: LeafCondition) => void }) {
  const attr = attrs.find((a) => a.key === leaf.attr);
  const isList = leaf.op === "in" || leaf.op === "not_in";
  const values = Array.isArray(leaf.value) ? leaf.value : [leaf.value];
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Select value={leaf.attr} onValueChange={(v) => onChange({ ...leaf, attr: v })}>
        <SelectTrigger className="h-8 w-48" aria-label="Attribute"><SelectValue placeholder="Attribute" /></SelectTrigger>
        <SelectContent>{attrs.map((a) => <SelectItem key={a.key} value={a.key}>{a.label}</SelectItem>)}</SelectContent>
      </Select>
      <Select value={leaf.op} onValueChange={(op) => {
        const o = op as Op;
        const v = o === "between" ? [values[0] ?? 0, values[1] ?? values[0] ?? 0] : o === "in" || o === "not_in" ? values.filter((x) => x !== null) : values[0];
        onChange({ ...leaf, op: o, value: v as Scalar | Scalar[] });
      }}>
        <SelectTrigger className="h-8 w-24" aria-label="Operator"><SelectValue /></SelectTrigger>
        <SelectContent>{OPS.map((o) => <SelectItem key={o} value={o}>{o}</SelectItem>)}</SelectContent>
      </Select>
      {leaf.op === "between" ? (
        <>
          <Input className="h-8 w-24" value={String(values[0] ?? "")} onChange={(e) => onChange({ ...leaf, value: [parseValue(e.target.value, attr), values[1] ?? null] })} />
          <span className="text-xs">and</span>
          <Input className="h-8 w-24" value={String(values[1] ?? "")} onChange={(e) => onChange({ ...leaf, value: [values[0] ?? null, parseValue(e.target.value, attr)] })} />
        </>
      ) : isList ? (
        <Input className="h-8 w-56" placeholder={attr?.allowed_values?.join(", ") ?? "a, b, c"} value={values.join(", ")}
          onChange={(e) => onChange({ ...leaf, value: e.target.value.split(",").map((s) => s.trim()).filter(Boolean).map((s) => parseValue(s, attr)) })} />
      ) : attr?.type === "enum" || attr?.type === "boolean" ? (
        <Select value={String(leaf.value ?? "")} onValueChange={(v) => onChange({ ...leaf, value: parseValue(v, attr) })}>
          <SelectTrigger className="h-8 w-40" aria-label="Value"><SelectValue placeholder="Value" /></SelectTrigger>
          <SelectContent>{(attr.type === "boolean" ? ["true", "false"] : attr.allowed_values ?? []).map((o) => <SelectItem key={o} value={o}>{o}</SelectItem>)}</SelectContent>
        </Select>
      ) : (
        <Input className="h-8 w-32" type={attr?.type === "number" ? "number" : attr?.type === "date" ? "date" : "text"} value={String(leaf.value ?? "")}
          onChange={(e) => onChange({ ...leaf, value: parseValue(e.target.value, attr) })} />
      )}
    </div>
  );
}

/** Recursive condition builder: leaf, ALL / ANY groups and NOT. */
export function ConditionEditor({ value, attrs, onChange, depth = 0 }: { value: Condition; attrs: Attribute[]; onChange: (c: Condition) => void; depth?: number }) {
  const newLeaf = (): LeafCondition => ({ attr: attrs[0]?.key ?? "", op: ">", value: 0 });
  if ("attr" in value) return <LeafEditor leaf={value} attrs={attrs} onChange={onChange} />;
  if ("not" in value) {
    return (
      <div className="rounded-md border border-dashed p-2">
        <div className="mb-1 text-xs font-semibold text-muted-foreground">NOT</div>
        <ConditionEditor value={value.not} attrs={attrs} depth={depth + 1} onChange={(c) => onChange({ not: c })} />
      </div>
    );
  }
  const mode = "all" in value ? "all" : "any";
  const items: Condition[] = "all" in value ? value.all : value.any;
  const setItems = (next: Condition[]) => onChange(mode === "all" ? { all: next } : { any: next });
  return (
    <div className="space-y-2 rounded-md border bg-muted/20 p-2">
      <div className="flex items-center gap-2 text-xs">
        <Select value={mode} onValueChange={(m) => onChange(m === "all" ? { all: items } : { any: items })}>
          <SelectTrigger className="h-7 w-44" aria-label="Group mode"><SelectValue /></SelectTrigger>
          <SelectContent><SelectItem value="all">ALL of (and)</SelectItem><SelectItem value="any">ANY of (or)</SelectItem></SelectContent>
        </Select>
        {items.length === 0 && <span className="text-muted-foreground">{mode === "all" ? "Always applies" : "Never applies"}</span>}
      </div>
      {items.map((c, i) => (
        <div key={i} className="flex items-start gap-2">
          <div className="flex-1"><ConditionEditor value={c} attrs={attrs} depth={depth + 1} onChange={(n) => setItems(items.map((x, j) => (j === i ? n : x)))} /></div>
          <Button size="icon" variant="ghost" className="h-8 w-8" title="Negate" onClick={() => setItems(items.map((x, j) => (j === i ? ("not" in x ? x.not : { not: x }) : x)))}>¬</Button>
          <Button size="icon" variant="ghost" className="h-8 w-8" aria-label="Remove condition" onClick={() => setItems(items.filter((_, j) => j !== i))}><Trash2 /></Button>
        </div>
      ))}
      <div className="flex gap-2">
        <Button size="sm" variant="outline" className="h-7" onClick={() => setItems([...items, newLeaf()])}><Plus /> Condition</Button>
        {depth < 2 && <Button size="sm" variant="outline" className="h-7" onClick={() => setItems([...items, { any: [newLeaf()] }])}><Plus /> Group</Button>}
      </div>
    </div>
  );
}

export function RuleDialog({ open, rule, attrs, stages, onClose, onSave }: {
  open: boolean; rule: Rule | null; attrs: Attribute[]; stages: string[]; onClose: () => void; onSave: (r: Rule) => void;
}) {
  const [r, setR] = useState<Rule | null>(rule);
  useEffect(() => setR(rule ? { ...rule, condition: "attr" in rule.condition ? { all: [rule.condition] } : rule.condition } : null), [rule]);
  if (!r) return null;
  const set = (p: Partial<Rule>) => setR({ ...r, ...p });
  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-3xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{rule?.name ? `Edit rule ${rule.id}` : "New rule"}</DialogTitle>
          <DialogDescription>Rules are configuration. The engine evaluates them in stage order, then priority (lower first), then id.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-3 sm:grid-cols-4">
          <Field label="Name" className="sm:col-span-2"><Input value={r.name} onChange={(e) => set({ name: e.target.value })} /></Field>
          <Field label="Stage">
            <Select value={r.stage} onValueChange={(v) => set({ stage: v })}>
              <SelectTrigger aria-label="Stage"><SelectValue /></SelectTrigger>
              <SelectContent>{stages.map((s) => <SelectItem key={s} value={s}>{s}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
          <Field label="Priority (lower first)"><Input type="number" min={0} value={r.priority} onChange={(e) => set({ priority: Number(e.target.value) })} /></Field>
        </div>
        <Field label="When"><ConditionEditor value={r.condition} attrs={attrs} onChange={(c) => set({ condition: c })} /></Field>
        <div className="grid gap-3 sm:grid-cols-3">
          <Field label="Then">
            <Select value={r.action.type} onValueChange={(t) => set({ action: { type: t as ActionType, value: t === "none" || t === "formula" ? null : r.action.value ?? 0, expression: t === "formula" ? r.action.expression ?? "price" : null } })}>
              <SelectTrigger aria-label="Action"><SelectValue /></SelectTrigger>
              <SelectContent>{ACTIONS.map((a) => <SelectItem key={a.value} value={a.value}>{a.label}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
          {r.action.type === "formula" ? (
            <Field label="Expression" hint="Variables: price, base, base_price, cost and attribute keys" className="sm:col-span-2">
              <Input className="font-mono" value={r.action.expression ?? ""} onChange={(e) => set({ action: { ...r.action, expression: e.target.value } })} />
            </Field>
          ) : r.action.type !== "none" ? (
            <Field label={r.action.type === "percent" ? "Percentage points (−10 = 10% off)" : "Value"}>
              <Input type="number" value={r.action.value ?? 0} onChange={(e) => set({ action: { ...r.action, value: Number(e.target.value) } })} />
            </Field>
          ) : null}
        </div>
        <Field label="Customer-facing message (optional, no cost or margin details)"><Input value={r.customer_message ?? ""} onChange={(e) => set({ customer_message: e.target.value || null })} /></Field>
        <label className="flex items-center gap-2 text-sm"><Switch checked={r.enabled} onCheckedChange={(c) => set({ enabled: c })} /> Enabled</label>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={() => {
            const cond = "all" in r.condition && r.condition.all.length === 1 && "attr" in r.condition.all[0] ? r.condition.all[0] : r.condition;
            onSave({ ...r, condition: cond });
          }}>Apply to draft</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
