"""Building blocks of a pricing decision that are shared by the engine and the optimizer contract."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from .common import Id, IsoDate, Key, Scalar, StrictModel


class PriceRequest(StrictModel):
    """A request to price one product. There is deliberately no price field: callers can't supply one."""

    domain_id: Id
    product_id: Id
    customer_id: Optional[Id] = None
    attributes: dict[Key, Scalar] = Field(default_factory=dict)
    as_of: IsoDate


class CustomerProfile(StrictModel):
    id: Id
    attributes: dict[Key, Scalar] = Field(default_factory=dict)


class AppliedRule(StrictModel):
    rule_id: str
    name: str
    stage: Optional[str] = None
    before: float
    after: float
    delta: float
    reason: str


class OverriddenRule(StrictModel):
    rule_id: str
    reason: str
    overridden_by: Optional[str] = None


class SkippedRule(StrictModel):
    rule_id: str
    reason: str


class Violation(StrictModel):
    constraint: str
    message: str


class ConstraintEval(StrictModel):
    constraint: str  # a configured ConstraintType, or the implicit "positive_price" / "cost_floor"
    value: Optional[float] = None
    kind: Literal["lower", "upper", "step"]
    bound: Optional[float] = None
    passed: bool
    message: str


class GeneratedCandidate(StrictModel):
    price: float
    reasons: list[str] = Field(min_length=1)


class CheckedCandidate(GeneratedCandidate):
    eligible: bool
    violations: list[Violation] = Field(default_factory=list)


class Bounds(StrictModel):
    lower: Optional[float] = None  # None = unbounded
    upper: Optional[float] = None
    lower_by: list[str] = Field(default_factory=list)
    upper_by: list[str] = Field(default_factory=list)
    conflict: bool = False


DecisionStatus = Literal["OK", "NO_VALID_PRICE"]
