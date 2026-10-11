// Display formatting only. The frontend never calculates prices; it formats values the API returned.
import type { OutputUnit } from "../types";

export function formatUnit(value: number | null | undefined, unit: Pick<OutputUnit, "prefix" | "suffix" | "decimals">): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const body = value.toLocaleString("en-IN", { minimumFractionDigits: unit.decimals, maximumFractionDigits: unit.decimals });
  return `${unit.prefix ?? ""}${body}${unit.suffix ?? ""}`;
}

const CURRENCY_PREFIX: Record<string, string> = { INR: "₹", USD: "$", EUR: "€", GBP: "£" };

/** Format money by ISO currency when the domain's output unit isn't at hand. */
export function formatMoney(value: number | null | undefined, currency: string, decimals = 2): string {
  return formatUnit(value, { prefix: CURRENCY_PREFIX[currency] ?? `${currency} `, suffix: "", decimals });
}

/** A price in the domain's own unit. Suffixes that start with a space (" /night") are shown only when `full`. */
export function formatPrice(value: number | null | undefined, u: { unit_prefix?: string; unit_suffix?: string; decimals?: number; currency?: string }, full = false): string {
  if (u.unit_prefix === undefined && u.currency) return formatMoney(value, u.currency, u.decimals ?? 2);
  const suffix = u.unit_suffix ?? "";
  return formatUnit(value, { prefix: u.unit_prefix ?? "", suffix: full || !suffix.startsWith(" ") ? suffix : "", decimals: u.decimals ?? 2 });
}

export function formatPercent(value: number | null | undefined, decimals = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatDelta(value: number, unit: Pick<OutputUnit, "prefix" | "suffix" | "decimals">): string {
  const sign = value > 0 ? "+" : value < 0 ? "−" : "±";
  return `${sign}${formatUnit(Math.abs(value), unit)}`;
}

export function shortTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
