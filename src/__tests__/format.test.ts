import { describe, expect, it } from "vitest";
import { formatMoney, formatPercent, formatUnit } from "@/engine/format";
import { actionText, conditionText } from "@/engine/readback";
import type { Attribute } from "@/types";

describe("display formatting (the UI never computes prices)", () => {
  it("formats money and percentages", () => {
    expect(formatMoney(6950, "INR", 0)).toBe("₹6,950");
    expect(formatMoney(2.4, "USD")).toBe("$2.40");
    expect(formatUnit(null, { prefix: "$", suffix: "", decimals: 2 })).toBe("—");
    expect(formatPercent(0.722, 1)).toBe("72.2%");
  });
  it("reads conditions and actions back in plain English", () => {
    const attrs: Attribute[] = [{ key: "occupancy", label: "Occupancy", type: "number", source: "inventory" }];
    expect(conditionText({ all: [{ attr: "occupancy", op: ">", value: 80 }, { not: { attr: "x", op: "in", value: ["a", "b"] } }] }, attrs))
      .toBe("Occupancy is more than 80 and not (x is one of a, b)");
    expect(actionText({ type: "percent", value: -8 }, { label: "", prefix: "$", suffix: "", decimals: 0 })).toBe("−8%");
  });
});
