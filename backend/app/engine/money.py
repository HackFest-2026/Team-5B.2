"""Decimal-safe price arithmetic. Every stored intermediate price is normalised to 6 dp (half-up)."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Context, Decimal, localcontext

PRICE_DP = 6
_CTX = Context(prec=40, rounding=ROUND_HALF_UP)


def D(value: float | int | str | Decimal) -> Decimal:
    """Build a Decimal from a float via its shortest repr, so 0.1 stays 0.1."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return Decimal(int(value))
    return Decimal(repr(value)) if isinstance(value, float) else Decimal(value)


def round_to(value: float | Decimal, decimals: int) -> float:
    with localcontext(_CTX):
        q = Decimal(1).scaleb(-decimals)
        return float(D(value).quantize(q, rounding=ROUND_HALF_UP))


def norm(value: float | Decimal) -> float:
    return round_to(value, PRICE_DP)


def round_to_step(value: float | Decimal, step: float | None) -> float:
    """Round to the nearest multiple of ``step`` (half-up)."""
    if not step or step <= 0:
        return float(D(value))
    with localcontext(_CTX):
        s = D(step)
        return norm((D(value) / s).quantize(Decimal(1), rounding=ROUND_HALF_UP) * s)


def is_multiple_of(value: float, step: float, tol: float = 1e-6) -> bool:
    with localcontext(_CTX):
        q = D(value) / D(step)
        return abs(q - q.to_integral_value(rounding=ROUND_HALF_UP)) <= Decimal(repr(tol))


def pct_of(base: float, pct: float) -> Decimal:
    """``base * pct / 100`` as a Decimal."""
    with localcontext(_CTX):
        return D(base) * D(pct) / Decimal(100)
