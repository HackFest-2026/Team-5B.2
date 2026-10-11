import { Plus, Trash2 } from "lucide-react";
import type { Attribute, AttributeType, DomainConfig, Product } from "@/types";
import { SEMANTIC_SIGNALS } from "@/types";
import { Button } from "@/components/ui/button";
import { Input, Switch } from "@/components/ui/primitives";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/overlays";
import { Field } from "@/components/common";

const TYPES: AttributeType[] = ["number", "enum", "boolean", "date", "text"];
const SOURCES = ["request", "inventory", "profile", "computed"] as const;

export const BLANK_DOMAIN: DomainConfig = {
  id: "new-domain", name: "New domain", description: "", currency: "USD",
  output_unit: { label: "USD per unit", prefix: "$", suffix: "", decimals: 2 }, product_noun: "Product",
  attributes: [], pipeline: ["market", "customer"], personas: [], products: [{ id: "STANDARD", name: "Standard offering", base_price: 100, cost: 60, inventory: {} }],
};

function Sel<T extends string>({ value, options, onChange, label, allowNone }: { value: T | null | undefined; options: readonly T[]; onChange: (v: T | null) => void; label: string; allowNone?: boolean }) {
  return (
    <Select value={value ?? "__none"} onValueChange={(v) => onChange(v === "__none" ? null : (v as T))}>
      <SelectTrigger className="h-8" aria-label={label}><SelectValue /></SelectTrigger>
      <SelectContent>
        {allowNone && <SelectItem value="__none">—</SelectItem>}
        {options.map((o) => <SelectItem key={o} value={o}>{o}</SelectItem>)}
      </SelectContent>
    </Select>
  );
}

const num = (s: string): number | null => (s === "" ? null : Number(s));

export function DomainEditor({ value, onChange, lockId }: { value: DomainConfig; onChange: (d: DomainConfig) => void; lockId?: boolean }) {
  const set = (patch: Partial<DomainConfig>) => onChange({ ...value, ...patch });
  const setAttr = (i: number, patch: Partial<Attribute>) => set({ attributes: value.attributes.map((a, j) => (i === j ? { ...a, ...patch } : a)) });
  const setProd = (i: number, patch: Partial<Product>) => set({ products: value.products.map((p, j) => (i === j ? { ...p, ...patch } : p)) });

  return (
    <div className="space-y-6">
      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Field label="Domain id"><Input value={value.id} disabled={lockId} onChange={(e) => set({ id: e.target.value })} /></Field>
        <Field label="Name"><Input value={value.name} onChange={(e) => set({ name: e.target.value })} /></Field>
        <Field label="Currency (ISO)"><Input value={value.currency} maxLength={3} onChange={(e) => set({ currency: e.target.value.toUpperCase() })} /></Field>
        <Field label="Product noun"><Input value={value.product_noun} onChange={(e) => set({ product_noun: e.target.value })} /></Field>
        <Field label="Description" className="sm:col-span-2 lg:col-span-4"><Input value={value.description} onChange={(e) => set({ description: e.target.value })} /></Field>
        <Field label="Unit label"><Input value={value.output_unit.label} onChange={(e) => set({ output_unit: { ...value.output_unit, label: e.target.value } })} /></Field>
        <Field label="Prefix / suffix">
          <div className="flex gap-2">
            <Input value={value.output_unit.prefix} onChange={(e) => set({ output_unit: { ...value.output_unit, prefix: e.target.value } })} />
            <Input value={value.output_unit.suffix} onChange={(e) => set({ output_unit: { ...value.output_unit, suffix: e.target.value } })} />
          </div>
        </Field>
        <Field label="Decimals"><Input type="number" min={0} max={6} value={value.output_unit.decimals} onChange={(e) => set({ output_unit: { ...value.output_unit, decimals: Number(e.target.value) } })} /></Field>
        <Field label="Pipeline stages (in order, comma-separated)">
          <Input value={value.pipeline.join(", ")} onChange={(e) => set({ pipeline: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })} />
        </Field>
      </section>

      <section>
        <div className="mb-2 flex items-center justify-between">
          <div>
            <div className="font-medium">Attributes</div>
            <div className="text-xs text-muted-foreground">Keep your own names. Map each one to a reusable semantic signal where it fits; normalisation puts it on the signal's scale.</div>
          </div>
          <Button size="sm" variant="outline" onClick={() => set({ attributes: [...value.attributes, { key: `attr_${value.attributes.length + 1}`, label: "New attribute", type: "number", source: "request" }] })}>
            <Plus /> Attribute
          </Button>
        </div>
        <div className="overflow-x-auto rounded-md border">
          <table className="w-full min-w-[1100px] text-xs">
            <thead className="bg-muted/50 text-left text-muted-foreground">
              <tr>{["Key", "Label", "Type", "Source", "Semantic signal", "Values / formula", "Min", "Max", "Normalize min–max", "Invert", "Customer input", ""].map((h) => <th key={h} className="px-2 py-1.5 font-medium">{h}</th>)}</tr>
            </thead>
            <tbody>
              {value.attributes.map((a, i) => (
                <tr key={i} className="border-t align-top">
                  <td className="p-1"><Input className="h-8" value={a.key} onChange={(e) => setAttr(i, { key: e.target.value })} /></td>
                  <td className="p-1"><Input className="h-8" value={a.label} onChange={(e) => setAttr(i, { label: e.target.value })} /></td>
                  <td className="p-1 w-28"><Sel label="type" value={a.type} options={TYPES} onChange={(v) => setAttr(i, { type: v ?? "number", allowed_values: v === "enum" ? a.allowed_values ?? ["A", "B"] : null })} /></td>
                  <td className="p-1 w-28"><Sel label="source" value={a.source} options={SOURCES} onChange={(v) => setAttr(i, { source: v ?? "request", expression: v === "computed" ? a.expression ?? "" : null, customer_input: v === "request" ? a.customer_input : false })} /></td>
                  <td className="p-1 w-40"><Sel label="semantic" allowNone value={a.semantic ?? null} options={SEMANTIC_SIGNALS} onChange={(v) => setAttr(i, { semantic: v })} /></td>
                  <td className="p-1">
                    {a.type === "enum" ? <Input className="h-8" value={(a.allowed_values ?? []).join(", ")} onChange={(e) => setAttr(i, { allowed_values: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })} />
                      : a.source === "computed" ? <Input className="h-8 font-mono" value={a.expression ?? ""} placeholder="daysBetween(as_of, start)" onChange={(e) => setAttr(i, { expression: e.target.value })} />
                      : <span className="text-muted-foreground">—</span>}
                  </td>
                  <td className="p-1 w-20"><Input className="h-8" type="number" value={a.min ?? ""} onChange={(e) => setAttr(i, { min: num(e.target.value) })} /></td>
                  <td className="p-1 w-20"><Input className="h-8" type="number" value={a.max ?? ""} onChange={(e) => setAttr(i, { max: num(e.target.value) })} /></td>
                  <td className="p-1 w-36">
                    <div className="flex gap-1">
                      <Input className="h-8" type="number" value={a.normalize?.min ?? ""} onChange={(e) => {
                        const v = num(e.target.value);
                        setAttr(i, { normalize: v === null ? null : { ...(a.normalize ?? { max: 100 }), min: v } });
                      }} />
                      <Input className="h-8" type="number" value={a.normalize?.max ?? ""} onChange={(e) => {
                        const v = num(e.target.value);
                        setAttr(i, { normalize: v === null ? null : { ...(a.normalize ?? { min: 0 }), max: v } });
                      }} />
                    </div>
                  </td>
                  <td className="p-1 text-center"><Switch checked={!!a.normalize?.invert} disabled={!a.normalize} onCheckedChange={(c) => a.normalize && setAttr(i, { normalize: { ...a.normalize, invert: c } })} /></td>
                  <td className="p-1 text-center"><Switch checked={!!a.customer_input} disabled={a.source !== "request"} onCheckedChange={(c) => setAttr(i, { customer_input: c })} /></td>
                  <td className="p-1"><Button size="icon" variant="ghost" aria-label="Remove attribute" onClick={() => set({ attributes: value.attributes.filter((_, j) => j !== i) })}><Trash2 /></Button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <div className="mb-2 flex items-center justify-between">
          <div className="font-medium">{value.product_noun}s</div>
          <Button size="sm" variant="outline" onClick={() => set({ products: [...value.products, { id: `P${value.products.length + 1}`, name: "New", base_price: 100, cost: 60, inventory: {} }] })}><Plus /> {value.product_noun}</Button>
        </div>
        <div className="overflow-x-auto rounded-md border">
          <table className="w-full min-w-[720px] text-xs">
            <thead className="bg-muted/50 text-left text-muted-foreground"><tr>{["Id", "Name", "Base price", "Unit cost", "Inventory values (JSON)", ""].map((h) => <th key={h} className="px-2 py-1.5 font-medium">{h}</th>)}</tr></thead>
            <tbody>
              {value.products.map((p, i) => (
                <tr key={i} className="border-t">
                  <td className="p-1 w-28"><Input className="h-8" value={p.id} onChange={(e) => setProd(i, { id: e.target.value })} /></td>
                  <td className="p-1"><Input className="h-8" value={p.name} onChange={(e) => setProd(i, { name: e.target.value })} /></td>
                  <td className="p-1 w-28"><Input className="h-8" type="number" value={p.base_price} onChange={(e) => setProd(i, { base_price: Number(e.target.value) })} /></td>
                  <td className="p-1 w-28"><Input className="h-8" type="number" value={p.cost} onChange={(e) => setProd(i, { cost: Number(e.target.value) })} /></td>
                  <td className="p-1"><InventoryInput value={p.inventory} onChange={(inv) => setProd(i, { inventory: inv })} /></td>
                  <td className="p-1"><Button size="icon" variant="ghost" aria-label="Remove product" onClick={() => set({ products: value.products.filter((_, j) => j !== i) })}><Trash2 /></Button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function InventoryInput({ value, onChange }: { value: Record<string, unknown>; onChange: (v: Product["inventory"]) => void }) {
  return (
    <Input
      className="h-8 font-mono"
      defaultValue={JSON.stringify(value)}
      onBlur={(e) => {
        try {
          const parsed = JSON.parse(e.target.value || "{}");
          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) onChange(parsed);
        } catch {
          e.target.value = JSON.stringify(value);
        }
      }}
    />
  );
}
