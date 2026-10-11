"""Build the pricing context: request + customer profile + inventory + computed attributes.

Invalid input is reported, never silently replaced: a non-numeric value for a number attribute
or an enum value outside its allowed list is an error, not a quiet fallback to the default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.schemas.common import BUILTIN_KEYS, Scalar, is_iso_date
from app.schemas.domain import Attribute, DomainConfig, Product
from app.schemas.pricing_parts import CustomerProfile

from .expression import ExpressionError, evaluate_expression

_TRUE = {"true", "1", "yes", "y"}
_FALSE = {"false", "0", "no", "n"}


class InvalidInputError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class EnrichResult:
    context: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def coerce(value: Scalar, attr: Attribute) -> tuple[Scalar, Optional[str]]:
    """Convert a raw value to the attribute's type. Returns (value, error)."""
    if value is None or value == "":
        return attr.default, None
    t = attr.type
    if t == "number":
        if isinstance(value, bool):
            return None, f'"{attr.key}" expects a number, got a boolean'
        try:
            n = float(value)
        except (TypeError, ValueError):
            return None, f'"{attr.key}" expects a number, got {value!r}'
        if n != n or n in (float("inf"), float("-inf")):
            return None, f'"{attr.key}" must be a finite number'
        return n, None
    if t == "boolean":
        if isinstance(value, bool):
            return value, None
        s = str(value).strip().lower()
        if s in _TRUE:
            return True, None
        if s in _FALSE:
            return False, None
        return None, f'"{attr.key}" expects true/false, got {value!r}'
    if t == "date":
        if not is_iso_date(value):
            return None, f'"{attr.key}" expects a date (YYYY-MM-DD), got {value!r}'
        return str(value)[:10], None
    if t == "enum":
        s = str(value)
        if attr.allowed_values and s not in attr.allowed_values:
            return None, f'"{attr.key}" must be one of {", ".join(attr.allowed_values)}; got {s!r}'
        return s, None
    return str(value), None


def enrich_context(
    domain: DomainConfig,
    product: Product,
    request_attributes: dict[str, Scalar],
    as_of: str,
    profile: Optional[CustomerProfile] = None,
) -> EnrichResult:
    res = EnrichResult(context={
        "as_of": as_of,
        "base_price": product.base_price,
        "cost": product.cost,
        "previous_price": product.previous_price,
        "product_id": product.id,
        "customer_id": profile.id if profile else None,
    })
    ctx = res.context
    known = {a.key for a in domain.attributes}
    for key in sorted(request_attributes):
        if key not in known:
            res.warnings.append(f'Ignored unknown attribute "{key}"')
        else:
            attr = domain.attribute(key)
            if attr is not None and attr.source == "computed":
                res.warnings.append(f'Ignored value for computed attribute "{key}"')

    for a in domain.attributes:
        if a.source == "computed":
            continue
        raw: Scalar = None
        if a.source == "request":
            raw = request_attributes.get(a.key)
        elif a.source == "profile":
            raw = profile.attributes.get(a.key) if profile and a.key in profile.attributes else request_attributes.get(a.key)
        elif a.source == "inventory":
            # An explicit request value (e.g. a simulation scenario) overrides the stored inventory value.
            # The API decides which callers may send inventory attributes.
            raw = request_attributes[a.key] if request_attributes.get(a.key) is not None else product.inventory.get(a.key)
        value, err = coerce(raw, a)
        if err:
            res.errors.append(err)
        elif value is None:
            if a.required:
                res.errors.append(f'"{a.key}" ({a.label}) is required')
            else:
                res.warnings.append(f'No value for "{a.label}" ({a.key})')
        elif a.type == "number" and ((a.min is not None and value < a.min) or (a.max is not None and value > a.max)):
            res.errors.append(f'"{a.key}" = {value:g} is outside the allowed range [{a.min}, {a.max}]')
        ctx[a.key] = value

    # Computed attributes may depend on each other in any declaration order.
    pending = [a for a in domain.attributes if a.source == "computed"]
    for _ in range(len(domain.attributes) + 1):
        if not pending:
            break
        nxt: list[Attribute] = []
        for a in pending:
            try:
                value, err = coerce(evaluate_expression(a.expression or "", ctx), a)
                if err:
                    raise ExpressionError(err)
                ctx[a.key] = value
            except ExpressionError:
                nxt.append(a)
        if len(nxt) == len(pending):
            break
        pending = nxt
    for a in pending:
        try:
            evaluate_expression(a.expression or "", ctx)
            msg = "result has the wrong type"
        except ExpressionError as e:
            msg = str(e)
        res.warnings.append(f'Could not compute "{a.label}": {msg}')
        ctx[a.key] = a.default
    return res


__all__ = ["BUILTIN_KEYS", "EnrichResult", "InvalidInputError", "coerce", "enrich_context"]
