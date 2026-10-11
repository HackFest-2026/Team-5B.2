"""Conservative reachability analysis for rule conditions (no false positives)."""

from __future__ import annotations

import math
from typing import Any, Optional

from app.schemas.domain import Attribute
from app.schemas.rules import AllCondition, AnyCondition, Condition, LeafCondition, NotCondition


def numeric(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False



def unreachable_reason(c: Condition, attrs: dict[str, Attribute]) -> Optional[str]:
    """A human reason if the condition can never be true, else None. Conservative (no false positives)."""
    if isinstance(c, AnyCondition):
        if not c.any:
            return "it has an empty OR group"
        reasons = [unreachable_reason(s, attrs) for s in c.any]
        return "; ".join(r for r in reasons if r) if all(reasons) else None
    if isinstance(c, NotCondition):
        if isinstance(c.not_, AllCondition) and not c.not_.all:
            return "it negates an always-true condition"
        return None
    leaves: list[LeafCondition] = []
    nested: list[Condition] = []

    def collect(x: Condition) -> None:
        if isinstance(x, LeafCondition):
            leaves.append(x)
        elif isinstance(x, AllCondition):
            for s in x.all:
                collect(s)
        else:
            nested.append(x)

    collect(c)
    for sub in nested:
        r = unreachable_reason(sub, attrs)
        if r:
            return r

    by_attr: dict[str, list[LeafCondition]] = {}
    for leaf in leaves:
        by_attr.setdefault(leaf.attr, []).append(leaf)

    for key, ls in by_attr.items():
        attr = attrs.get(key)
        label = attr.label if attr else key
        allowed: Optional[set[str]] = set(attr.allowed_values) if attr and attr.type == "enum" and attr.allowed_values else None
        if attr and attr.type == "boolean":
            allowed = {"true", "false"}

        def norm_v(v: Any) -> str:
            if isinstance(v, bool):
                return "true" if v else "false"
            return str(v)

        for leaf in ls:
            if leaf.op == "==":
                allowed = _intersect(allowed, [norm_v(leaf.value)])
            elif leaf.op == "in" and isinstance(leaf.value, list):
                allowed = _intersect(allowed, [norm_v(v) for v in leaf.value])
        if allowed is not None:
            for leaf in ls:
                if leaf.op == "!=":
                    allowed.discard(norm_v(leaf.value))
                elif leaf.op == "not_in" and isinstance(leaf.value, list):
                    for v in leaf.value:
                        allowed.discard(norm_v(v))
            if not allowed:
                return f"{label} cannot satisfy all its equality conditions at once"

        if attr is not None and attr.type != "number":
            continue
        lo, lo_inc = (attr.min if attr and attr.min is not None else -math.inf), True
        hi, hi_inc = (attr.max if attr and attr.max is not None else math.inf), True

        def tighten_lo(v: float, inc: bool) -> None:
            nonlocal lo, lo_inc
            if v > lo or (v == lo and not inc):
                lo, lo_inc = v, inc

        def tighten_hi(v: float, inc: bool) -> None:
            nonlocal hi, hi_inc
            if v < hi or (v == hi and not inc):
                hi, hi_inc = v, inc

        for leaf in ls:
            if leaf.op == "between" and isinstance(leaf.value, list):
                if not all(numeric(v) for v in leaf.value):
                    continue
                a, b = (float(v) for v in leaf.value)
                if a > b:
                    return f"{label} is between {a:g} and {b:g}, but the lower bound is above the upper bound"
                tighten_lo(a, True)
                tighten_hi(b, True)
                continue
            if isinstance(leaf.value, list) or not numeric(leaf.value):
                continue
            v = float(leaf.value)
            if leaf.op == ">":
                tighten_lo(v, False)
            elif leaf.op == ">=":
                tighten_lo(v, True)
            elif leaf.op == "<":
                tighten_hi(v, False)
            elif leaf.op == "<=":
                tighten_hi(v, True)
            elif leaf.op == "==":
                tighten_lo(v, True)
                tighten_hi(v, True)
        if lo > hi or (lo == hi and not (lo_inc and hi_inc)):
            return (f"{label} would need to be {'≥' if lo_inc else '>'} {lo:g} and "
                    f"{'≤' if hi_inc else '<'} {hi:g}, which is impossible")
    return None


def _intersect(s: Optional[set[str]], vals: list[str]) -> set[str]:
    return set(vals) if s is None else {v for v in vals if v in s}
