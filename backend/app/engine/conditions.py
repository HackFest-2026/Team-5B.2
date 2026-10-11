"""Evaluate rule conditions against a pricing context, explaining every result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.schemas.rules import AllCondition, AnyCondition, Condition, LeafCondition, NotCondition

from .expression import js_str


@dataclass(frozen=True)
class ConditionResult:
    ok: bool
    reason: str


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _as_number(v: Any) -> float | None:
    if _is_number(v):
        return float(v)
    if isinstance(v, str) and v.strip() != "":
        try:
            return float(v)
        except ValueError:
            return None
    return None


def cmp_values(a: Any, b: Any) -> float:
    """Numbers compare numerically (numeric strings too); ISO dates and other strings lexicographically."""
    if not isinstance(a, bool) and not isinstance(b, bool):
        na, nb = _as_number(a), _as_number(b)
        if na is not None and nb is not None:
            return na - nb
    sa, sb = js_str(a), js_str(b)
    return -1 if sa < sb else (1 if sa > sb else 0)


def values_equal(a: Any, b: Any) -> bool:
    if a == b and type(a) is type(b):
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return js_str(a) == js_str(b)
    return cmp_values(a, b) == 0 and len(js_str(a)) > 0


def _show(v: Any) -> str:
    if isinstance(v, list):
        return "[" + ", ".join(js_str(x) for x in v) + "]"
    return repr(v) if isinstance(v, str) else js_str(v)


def evaluate_leaf(c: LeafCondition, ctx: Mapping[str, Any]) -> ConditionResult:
    if ctx.get(c.attr) is None:
        return ConditionResult(False, f'"{c.attr}" has no value')
    actual, expected, op = ctx[c.attr], c.value, c.op
    if op == ">":
        ok = cmp_values(actual, expected) > 0
    elif op == ">=":
        ok = cmp_values(actual, expected) >= 0
    elif op == "<":
        ok = cmp_values(actual, expected) < 0
    elif op == "<=":
        ok = cmp_values(actual, expected) <= 0
    elif op == "==":
        ok = values_equal(actual, expected)
    elif op == "!=":
        ok = not values_equal(actual, expected)
    elif op in ("in", "not_in"):
        items = expected if isinstance(expected, list) else [expected]
        found = any(values_equal(actual, v) for v in items)
        ok = found if op == "in" else not found
    elif op == "between":
        lo, hi = expected if isinstance(expected, list) else (expected, expected)
        ok = cmp_values(actual, lo) >= 0 and cmp_values(actual, hi) <= 0
    else:  # pragma: no cover - schema rejects unknown operators
        return ConditionResult(False, f'unsupported operator "{op}"')
    return ConditionResult(ok, f"{c.attr} = {_show(actual)} {'satisfies' if ok else 'fails'} {op} {_show(expected)}")


def evaluate_condition(c: Condition, ctx: Mapping[str, Any]) -> ConditionResult:
    if isinstance(c, LeafCondition):
        return evaluate_leaf(c, ctx)
    if isinstance(c, AllCondition):
        for sub in c.all:
            r = evaluate_condition(sub, ctx)
            if not r.ok:
                return r
        return ConditionResult(True, "all conditions met" if c.all else "always applies")
    if isinstance(c, AnyCondition):
        reasons: list[str] = []
        for sub in c.any:
            r = evaluate_condition(sub, ctx)
            if r.ok:
                return r
            reasons.append(r.reason)
        return ConditionResult(False, f"none matched ({'; '.join(reasons)})" if c.any else "empty ANY group")
    if isinstance(c, NotCondition):
        r = evaluate_condition(c.not_, ctx)
        return ConditionResult(not r.ok, f"NOT failed: {r.reason}" if r.ok else f"NOT passed: {r.reason}")
    return ConditionResult(False, "malformed condition")
