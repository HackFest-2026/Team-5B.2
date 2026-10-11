"""Map domain attributes onto reusable semantic signals.

Each domain decides which of its attributes stand for which signal (configuration, not code).
Raw values keep their domain units; normalised values are put on the configured target scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from app.schemas.domain import Attribute, DomainConfig


@dataclass
class SignalMap:
    raw: dict[str, float] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)  # normalised where configured, else raw
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _to_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def normalise(value: float, attr: Attribute) -> float:
    n = attr.normalize
    if n is None:
        return value
    x = (value - n.min) / (n.max - n.min)
    x = min(1.0, max(0.0, x))
    if n.invert:
        x = 1.0 - x
    lo, hi = n.target_range
    return lo + x * (hi - lo)


def map_signals(domain: DomainConfig, ctx: Mapping[str, Any]) -> SignalMap:
    out = SignalMap()
    out.raw["basePrice"] = out.values["basePrice"] = float(ctx["base_price"])
    out.raw["unitCost"] = out.values["unitCost"] = float(ctx["cost"])
    out.sources["basePrice"] = out.sources["unitCost"] = "product"

    for a in domain.attributes:
        if not a.semantic:
            continue
        v = _to_float(ctx.get(a.key))
        if v is None:
            if ctx.get(a.key) is not None:
                out.warnings.append(f'Signal {a.semantic}: "{a.key}" is not numeric')
            continue
        out.raw[a.semantic] = v
        out.values[a.semantic] = normalise(v, a)
        out.sources[a.semantic] = f"attribute:{a.key}"

    for sig, v in sorted(domain.defaults.items()):
        if sig not in out.values:
            out.raw[sig] = out.values[sig] = float(v)
            out.sources[sig] = "domain_default"
    return out
