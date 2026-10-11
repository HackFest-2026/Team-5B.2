// Plain-English read-back of conditions and actions (display only; no pricing logic here).
import type { Attribute, Condition, Constraint, OutputUnit, RuleAction, Scalar } from "../types";
import { formatUnit } from "./format";

const OP_WORDS: Record<string, string> = {
  ">": "is more than", ">=": "is at least", "<": "is less than", "<=": "is at most", "==": "is", "!=": "is not",
  in: "is one of", not_in: "is not one of", between: "is between",
};

const show = (v: Scalar | Scalar[]): string => (Array.isArray(v) ? v.join(", ") : typeof v === "boolean" ? (v ? "yes" : "no") : String(v));

export function conditionText(c: Condition, attrs: Attribute[]): string {
  const label = (k: string) => attrs.find((a) => a.key === k)?.label ?? k;
  if ("attr" in c) {
    if (c.op === "between" && Array.isArray(c.value)) return `${label(c.attr)} is between ${show(c.value[0])} and ${show(c.value[1])}`;
    return `${label(c.attr)} ${OP_WORDS[c.op] ?? c.op} ${show(c.value)}`;
  }
  if ("all" in c) return c.all.length ? c.all.map((x) => wrap(x, attrs)).join(" and ") : "always";
  if ("any" in c) return c.any.length ? c.any.map((x) => wrap(x, attrs)).join(" or ") : "never";
  return `not (${conditionText(c.not, attrs)})`;
}

function wrap(c: Condition, attrs: Attribute[]) {
  const s = conditionText(c, attrs);
  return "attr" in c || "not" in c ? s : `(${s})`;
}

export function actionText(a: RuleAction, unit: OutputUnit): string {
  const v = a.value ?? 0;
  switch (a.type) {
    case "percent": return v < 0 ? `−${Math.abs(v)}%` : `+${v}%`;
    case "fixed": return v < 0 ? `−${formatUnit(Math.abs(v), unit)}` : `+${formatUnit(v, unit)}`;
    case "multiply": return `× ${v}`;
    case "set": return `set to ${formatUnit(v, unit)}`;
    case "none": return "no change";
    case "formula": return `= ${a.expression ?? ""}`;
  }
}

export const CONSTRAINT_HELP: Record<Constraint["type"], string> = {
  floor: "Absolute minimum price",
  ceiling: "Absolute maximum price",
  margin_floor: "Gross margin % = (price − cost) / price",
  markup_floor: "Markup % = (price − cost) / cost",
  max_discount: "Max % below the base price",
  max_increase: "Max % above the base price",
  max_daily_change: "Max % move vs the previous price",
  rounding: "Round candidates to this step",
};
