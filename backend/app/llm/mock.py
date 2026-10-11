"""Deterministic, keyword-based MOCK generator, used when no LLM key is configured.

It is clearly labelled ``provider="mock"`` everywhere and is never presented as an LLM. It knows
nothing about any industry: it only matches the domain's own attribute labels, keys, synonyms and
allowed values. Output goes through the same validation as real LLM output.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from app.schemas.domain import Attribute, DomainConfig

NUM = r"(-?\d[\d,]*(?:\.\d+)?)"
NEG = ("discount", "off", "reduce", "reduction", "lower", "decrease", "cut", "cheaper", "rebate", "less")
POS = ("surcharge", "increase", "raise", "premium", "markup", "uplift", "extra", "higher", "charge more", "fee")
HIGH = ("high", "higher", "strong", "peak", "busy", "heavy", "scarce", "tight", "full")
LOW = ("low", "lower", "weak", "quiet", "slow", "light", "plenty", "spare", "empty")
COMPARATORS: list[tuple[str, str]] = [
    (rf"(?:between|from)\s+{NUM}\s*%?\s+(?:and|to|-)\s+{NUM}", "between"),
    (rf"(?:at least|minimum of|no less than|or more)\s*{NUM}", ">="),
    (rf"(?:at most|maximum of|no more than|up to|not more than)\s*{NUM}", "<="),
    (rf"(?:more than|over|above|greater than|exceeds?|exceeding|>)\s*{NUM}", ">"),
    (rf"(?:less than|under|below|fewer than|<)\s*{NUM}", "<"),
]
DEFAULT_PCT = 10.0
DEFAULT_MARGIN = 20.0


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _words(key: str) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", key).replace("_", " ").replace("-", " ").lower()


def _terms(a: Attribute) -> list[str]:
    terms = {a.label.lower(), _words(a.key), *(s.lower() for s in a.synonyms)}
    first = a.label.lower().split()[0] if a.label else ""
    if len(first) >= 5:
        terms.add(first)
    return sorted((t for t in terms if len(t) > 2), key=len, reverse=True)


def _find(text: str, term: str) -> int:
    m = re.search(rf"\b{re.escape(term)}", text, re.I)
    return m.start() if m else -1


def _mentions(text: str, attrs: list[Attribute]) -> list[tuple[int, Attribute, list[str]]]:
    out = []
    for a in attrs:
        idx = [i for i in (_find(text, t) for t in _terms(a)) if i >= 0]
        enum_hits = [v for v in (a.allowed_values or []) if _find(text, v.lower()) >= 0] if a.type == "enum" else []
        idx += [_find(text, v.lower()) for v in enum_hits]
        if idx:
            out.append((min(idx), a, enum_hits))
    return sorted(out, key=lambda m: m[0])


def _blank_comparators(t: str) -> str:
    for pattern, _ in COMPARATORS:
        t = re.sub(pattern, lambda m: " " * len(m.group(0)), t, flags=re.I)
    return t


def _has(text: str, words: tuple[str, ...]) -> bool:
    return any(_find(text, w) >= 0 for w in words)


def _constraint(t: str) -> Optional[dict[str, Any]]:
    t = t.lower()
    if m := re.search(rf"markup[^.]*?{NUM}\s*%", t):
        return {"type": "markup_floor", "value": _num(m.group(1))}
    if "margin" in t:
        m = re.search(rf"margin[^.]*?{NUM}\s*%", t) or re.search(rf"{NUM}\s*%[^.]*?margin", t)
        return {"type": "margin_floor", "value": _num(m.group(1)) if m else DEFAULT_MARGIN, "_assumed": m is None}
    if m := (re.search(rf"(?:limit|cap|restrict)[^.]*?increase[^.]*?{NUM}\s*%", t)
             or re.search(rf"increase[^.]*?(?:no more than|at most|limit(?:ed)? to|cap(?:ped)? at|maximum of|not exceed)\s*{NUM}\s*%", t)
             or re.search(rf"(?:max(?:imum)?|total)\s+(?:price\s+)?increase[^.]*?{NUM}\s*%", t)):
        return {"type": "max_increase", "value": _num(m.group(1))}
    if m := (re.search(rf"(?:max(?:imum)?|total)\s+discount[^.]*?{NUM}\s*%", t)
             or re.search(rf"(?:never|not|no)[^.]*?discount[^.]*?more than\s*{NUM}\s*%", t)
             or re.search(rf"discounts?[^.]*?(?:never|not)\s+exceed\s*{NUM}\s*%", t)):
        return {"type": "max_discount", "value": _num(m.group(1))}
    if m := re.search(rf"round(?:ed)?\s+(?:prices?\s+)?(?:to|off to)\s+(?:the\s+)?(?:nearest\s+)?[^\d]{{0,4}}{NUM}", t):
        return {"type": "rounding", "value": _num(m.group(1))}
    if (m := re.search(rf"(?:never|not|no)\s[^.]*?(?:below|less than|under)\s+[^\d%-]{{0,4}}{NUM}(?!\s*%)", t)):
        return {"type": "floor", "value": _num(m.group(1))}
    if (m := re.search(rf"(?:never|not|no)\s[^.]*?(?:above|exceed|more than|over)\s+[^\d%-]{{0,4}}{NUM}(?!\s*%)", t)):
        return {"type": "ceiling", "value": _num(m.group(1))}
    return None


def _adjustment(t: str) -> Optional[tuple[dict[str, Any], Optional[str]]]:
    cleaned = _blank_comparators(t.lower())
    neg, pos = _has(cleaned, NEG), _has(cleaned, POS)
    if m := re.search(rf"(?:set|fix)\s+(?:the\s+)?price\s+(?:to|at)\s+[^\d-]{{0,4}}{NUM}", cleaned):
        return {"type": "set", "value": _num(m.group(1))}, None
    if m := re.search(rf"{NUM}\s*(?:%|percent)", cleaned):
        v = abs(_num(m.group(1)))
        note = None
        if neg and pos:
            note = "Both increase and decrease words found; assumed a decrease."
        elif not neg and not pos:
            note = "No direction word found; assumed an increase."
        return {"type": "percent", "value": -v if neg else v}, note
    if neg or pos:
        return ({"type": "percent", "value": -DEFAULT_PCT if neg and not pos else DEFAULT_PCT},
                f"No amount given; assumed {'-' if neg and not pos else '+'}{DEFAULT_PCT:g}%.")
    return None


def _range(a: Attribute) -> tuple[float, float]:
    if a.min is not None and a.max is not None:
        return a.min, a.max
    if a.normalize is not None:
        return a.normalize.min, a.normalize.max
    return 0.0, 100.0


def _conditions(t: str, mentions, assumptions: list[str]) -> list[dict[str, Any]]:
    lower = t.lower()
    leaves: list[dict[str, Any]] = []
    numeric = [(i, a) for i, a, _ in mentions if a.type == "number"]
    used: set[str] = set()
    for pattern, op in COMPARATORS:
        for m in re.finditer(pattern, lower):
            if not numeric:
                continue
            idx, a = min(numeric, key=lambda x: abs(x[0] - m.start()))
            value: Any = [_num(m.group(1)), _num(m.group(2))] if op == "between" else _num(m.group(1))
            leaves.append({"attr": a.key, "op": op, "value": value})
            used.add(a.key)
    for idx, a, enum_hits in mentions:
        if a.key in used:
            continue
        window = lower[idx: idx + len(a.label) + 25]
        if a.type == "number":
            lo, hi = _range(a)
            if _has(window, HIGH):
                v = round(lo + 0.7 * (hi - lo), 2)
                leaves.append({"attr": a.key, "op": ">=", "value": v})
                assumptions.append(f'Read "high {a.label.lower()}" as {a.label} ≥ {v:g}.')
            elif _has(window, LOW):
                v = round(lo + 0.3 * (hi - lo), 2)
                leaves.append({"attr": a.key, "op": "<", "value": v})
                assumptions.append(f'Read "low {a.label.lower()}" as {a.label} < {v:g}.')
        elif a.type == "enum" and enum_hits:
            leaves.append({"attr": a.key, "op": "in", "value": enum_hits} if len(enum_hits) > 1
                          else {"attr": a.key, "op": "==", "value": enum_hits[0]})
        elif a.type == "boolean":
            before = lower[max(0, idx - 12): idx]
            leaves.append({"attr": a.key, "op": "==", "value": not re.search(r"\b(?:not|no|without|non)\b[\s-]*$", before)})
    return leaves


def _title(sentence: str) -> str:
    words = re.sub(r"[.;:,]+$", "", sentence).split()
    s = " ".join(words[:7])
    return (s[:1].upper() + s[1:] + ("…" if len(words) > 7 else ""))[:120]


def generate_strategy(domain: DomainConfig, description: str, market: Optional[dict] = None) -> dict[str, Any]:
    """Return a dict shaped like LlmStrategyOut. Market data is only summarised, not reasoned about."""
    rules: list[dict[str, Any]] = []
    constraints: list[dict[str, Any]] = []
    assumptions: list[str] = []
    warnings: list[str] = []
    priority = 10
    sentences = [s.strip(" -*•\t") for s in re.split(r"(?<=[.;])\s+|\n+", description) if s.strip()]
    for sentence in sentences:
        # Split "X, but never Y" into the rule part and the constraint part.
        parts = re.split(r",?\s+but\s+|;\s*", sentence)
        for part in parts:
            c = _constraint(part)
            mentions = _mentions(part, domain.attributes)
            adj = _adjustment(part)
            if c and (adj is None or not mentions or c["type"] in ("max_increase", "max_discount")):
                if c.pop("_assumed", False):
                    assumptions.append(f"No margin percentage given; assumed a {DEFAULT_MARGIN:g}% gross margin floor.")
                if not any(x["type"] == c["type"] for x in constraints):
                    constraints.append(c)
                continue
            if adj is None:
                if len(part) > 3:
                    warnings.append(f'Could not turn "{part}" into a rule or constraint.')
                continue
            action, note = adj
            if note:
                assumptions.append(f'"{_title(part)}": {note}')
            leaves = _conditions(part, mentions, assumptions)
            if not leaves:
                assumptions.append(f'"{_title(part)}": no condition recognised, so the rule always applies.')
            stage = next((a.stage for _, a, _ in mentions if a.stage in domain.pipeline), None) or domain.pipeline[0]
            rules.append({
                "name": _title(part), "description": part,
                "condition": {"match": "all", "conditions": leaves},
                "action": action, "priority": priority, "stage": stage,
                "customer_message": ("Price reflects current demand." if action.get("value", 0) > 0
                                     else "A discount has been applied."),
            })
            priority += 10
    if not rules and not constraints:
        warnings.append("Nothing in the description matched this domain's attributes.")
    observations = []
    for m in (market or {}).get("products", []):
        if m.get("market_median"):
            observations.append(f"{m['name']}: market median {m['market_median']:g} vs our base {m['our_base_price']:g} "
                                f"({m['observed_prices']} prices, {m['confidence']} confidence). Not used by the mock generator.")
    return {
        "name": f"{domain.name} strategy (mock draft)",
        "objective_text": description,
        "rules": rules,
        "constraints": constraints,
        "conflict_policy": {"mode": "stack", "combine": "multiplicative"},
        "optimizer_policy": {"objective": "max_expected_profit", "conversion_floor": 0.10},
        "assumptions": assumptions,
        "market_observations": observations,
        "warnings": warnings,
    }


# ---------- domain suggestion ----------

_CONCEPTS: list[tuple[str, list[dict[str, Any]]]] = [
    (r"\b(demand|busy|popular|peak)\b", [{"key": "demand_index", "label": "Demand index", "type": "number", "source": "inventory",
       "min": 0, "max": 100, "default": 50, "semantic": "demandPressure", "normalize": {"min": 0, "max": 100, "target_range": [0.2, 2.0]}}]),
    (r"\b(utili[sz]ation|capacity|availability|stock|inventory|occupied|full)\b", [{"key": "utilization", "label": "Utilization", "type": "number",
       "source": "inventory", "unit": "%", "min": 0, "max": 100, "default": 60, "semantic": "capacityUtilization", "normalize": {"min": 0, "max": 100}}]),
    (r"\b(competitor|rival|market rate)s?\b", [{"key": "competitor_price", "label": "Competitor price", "type": "number", "source": "inventory",
       "semantic": "competitorPrice"}]),
    (r"\b(advance|ahead|lead time|last[- ]minute|early|booking date|start date)\b", [
        {"key": "start_date", "label": "Start date", "type": "date", "source": "request", "customer_input": True, "required": True},
        {"key": "days_ahead", "label": "Days ahead", "type": "number", "source": "computed", "expression": "daysBetween(as_of, start_date)",
         "semantic": "urgency", "normalize": {"min": 0, "max": 30, "invert": True}}]),
    (r"\b(duration|days|nights|weeks|hours|length|term)\b", [{"key": "duration", "label": "Duration", "type": "number", "source": "request",
       "customer_input": True, "min": 1, "max": 365, "default": 1}]),
    (r"\b(quantity|units|people|seats|group|volume|bulk)\b", [{"key": "quantity", "label": "Quantity", "type": "number", "source": "request",
       "customer_input": True, "min": 1, "max": 1000, "default": 1}]),
    (r"\b(loyal|member|returning|repeat)\b", [{"key": "loyalty_score", "label": "Loyalty score", "type": "number", "source": "profile",
       "min": 0, "max": 100, "default": 0, "semantic": "customerLoyalty", "normalize": {"min": 0, "max": 100}}]),
    (r"\b(promo|promotion|sale|coupon)\b", [{"key": "promo_active", "label": "Promotion active", "type": "boolean", "source": "request",
       "default": False, "semantic": "promotionActive"}]),
]


def suggest_domain(description: str) -> dict[str, Any]:
    text = description.lower()
    attrs: list[dict[str, Any]] = []
    for pattern, concept in _CONCEPTS:
        if re.search(pattern, text):
            attrs.extend(concept)
    if not attrs:
        attrs = _CONCEPTS[0][1] + _CONCEPTS[1][1]
    words = re.findall(r"[a-z0-9]+", text)[:3] or ["new", "domain"]
    slug = "-".join(words)[:40].strip("-") or "new-domain"
    currency, prefix = ("INR", "₹") if re.search(r"₹|\binr\b|rupee", text) else ("EUR", "€") if re.search(r"€|\beur", text) else ("USD", "$")
    return {
        "id": slug,
        "name": " ".join(w.capitalize() for w in words),
        "description": f"Mock draft from description (review every field): {description[:400]}",
        "currency": currency,
        "output_unit": {"label": f"{currency} per unit", "prefix": prefix, "suffix": "", "decimals": 2},
        "product_noun": "Product",
        "pipeline": ["market", "customer"],
        "attributes": attrs,
        "products": [{"id": "STANDARD", "name": "Standard offering", "base_price": 100, "cost": 60}],
    }
