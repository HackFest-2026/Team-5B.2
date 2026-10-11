"""Domain-agnostic price extraction from search results, plus outlier filtering and summary stats."""

from __future__ import annotations

import re
from statistics import median
from typing import Optional

CURRENCY_TOKENS = {
    "INR": r"(?:₹|rs\.?|inr)", "USD": r"(?:\$|usd|us\$)", "EUR": r"(?:€|eur)", "GBP": r"(?:£|gbp)",
    "AED": r"(?:aed|dhs?)", "JPY": r"(?:¥|jpy)", "AUD": r"(?:a\$|aud)", "CAD": r"(?:c\$|cad)", "SGD": r"(?:s\$|sgd)",
}
REGION = {"INR": "in", "USD": "us", "EUR": "de", "GBP": "uk", "AED": "ae", "JPY": "jp", "AUD": "au", "CAD": "ca", "SGD": "sg"}
NUM = r"(\d{1,3}(?:[,\s]\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
# Accept prices within this band of the business's own base price; everything else is treated as noise
# (accessories, bundles, monthly vs yearly, unrelated pages). Ratios keep the filter domain-agnostic.
BAND = (0.25, 4.0)


def region_for(currency: str) -> str:
    return REGION.get(currency.upper(), "us")


def _to_float(s: str) -> Optional[float]:
    try:
        return float(re.sub(r"[,\s]", "", s))
    except ValueError:
        return None


def extract_prices(text: str, currency: str) -> list[float]:
    """Amounts written with the domain's currency (symbol or code, before or after the number)."""
    tok = CURRENCY_TOKENS.get(currency.upper(), re.escape(currency.lower()))
    t = text.lower().replace(" ", " ")
    out = []
    for m in re.finditer(rf"{tok}\s?{NUM}|{NUM}\s?{tok}\b", t):
        v = _to_float(m.group(1) or m.group(2))
        if v is not None and v > 0:
            out.append(v)
    return out


def percent_values(text: str) -> list[float]:
    """Percentages such as '10.5% p.a.' (for rate-priced domains whose unit is a percentage)."""
    return [v for v in (_to_float(m) for m in re.findall(r"(\d{1,2}(?:\.\d{1,2})?)\s?%", text)) if v is not None and 0 < v < 100]


def in_band(price: float, base: float) -> bool:
    return base > 0 and BAND[0] <= price / base <= BAND[1]


def summarize(prices: list[float]) -> dict:
    if not prices:
        return {"n": 0, "min": None, "median": None, "max": None, "confidence": "none"}
    ps = sorted(prices)
    # Trim extremes once there are enough points.
    if len(ps) >= 6:
        k = max(1, len(ps) // 10)
        ps = ps[k:-k]
    n = len(prices)
    return {"n": n, "min": round(ps[0], 4), "median": round(median(ps), 4), "max": round(ps[-1], 4),
            "confidence": "high" if n >= 8 else "medium" if n >= 4 else "low"}
