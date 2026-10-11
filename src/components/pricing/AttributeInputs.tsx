import type { Attribute, Scalar } from "@/types";
import { Input } from "@/components/ui/primitives";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/overlays";
import { Field } from "@/components/common";

/** Form fields generated from attribute definitions (no domain-specific UI code). */
export function AttributeInputs({ attributes, values, onChange, placeholders }: {
  attributes: Attribute[];
  values: Record<string, Scalar>;
  onChange: (key: string, value: Scalar) => void;
  placeholders?: Record<string, Scalar>;
}) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {attributes.map((a) => {
        const v = values[a.key];
        const ph = placeholders?.[a.key] ?? a.default;
        const label = `${a.label}${a.unit ? ` (${a.unit})` : ""}${a.required ? " *" : ""}`;
        const hint = a.description ?? (a.min != null || a.max != null ? `Range ${a.min ?? "…"}–${a.max ?? "…"}` : undefined);
        if (a.type === "enum") {
          return (
            <Field key={a.key} label={label} hint={hint}>
              <Select value={v == null ? "__default" : String(v)} onValueChange={(x) => onChange(a.key, x === "__default" ? null : x)}>
                <SelectTrigger aria-label={a.label}><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__default">{ph != null ? `Default (${ph})` : "Not set"}</SelectItem>
                  {(a.allowed_values ?? []).map((o) => <SelectItem key={o} value={o}>{o}</SelectItem>)}
                </SelectContent>
              </Select>
            </Field>
          );
        }
        if (a.type === "boolean") {
          return (
            <Field key={a.key} label={label} hint={hint}>
              <Select value={v == null ? "__default" : String(v)} onValueChange={(x) => onChange(a.key, x === "__default" ? null : x === "true")}>
                <SelectTrigger aria-label={a.label}><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__default">{ph != null ? `Default (${ph ? "yes" : "no"})` : "Not set"}</SelectItem>
                  <SelectItem value="true">Yes</SelectItem>
                  <SelectItem value="false">No</SelectItem>
                </SelectContent>
              </Select>
            </Field>
          );
        }
        return (
          <Field key={a.key} label={label} hint={hint}>
            <Input
              aria-label={a.label}
              type={a.type === "number" ? "number" : a.type === "date" ? "date" : "text"}
              value={v == null ? "" : String(v)}
              placeholder={ph == null ? "" : String(ph)}
              min={a.min ?? undefined}
              max={a.max ?? undefined}
              onChange={(e) => {
                const raw = e.target.value;
                onChange(a.key, raw === "" ? null : a.type === "number" ? Number(raw) : raw);
              }}
            />
          </Field>
        );
      })}
    </div>
  );
}

/** Drop empty values so the backend applies its defaults. */
export function cleanValues(values: Record<string, Scalar>): Record<string, Scalar> {
  return Object.fromEntries(Object.entries(values).filter(([, v]) => v !== null && v !== "" && v !== undefined));
}
