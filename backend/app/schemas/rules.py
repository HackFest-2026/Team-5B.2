"""Rule contracts: conditions (a small boolean tree), actions and rules.

Rules are configuration, never code. The only expression support is the engine's safe
formula language, and formula actions are rejected in LLM-generated strategies.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import ConfigDict, Discriminator, Field, Tag, model_validator

from .common import Id, IsoDate, Key, Scalar, ScalarOrList, StrictModel

Op = Literal[">", ">=", "<", "<=", "==", "!=", "in", "not_in", "between"]
SUPPORTED_OPS: tuple[str, ...] = (">", ">=", "<", "<=", "==", "!=", "in", "not_in", "between")
LIST_OPS = ("in", "not_in")

ActionType = Literal["percent", "fixed", "set", "none", "multiply", "formula"]
SUPPORTED_ACTIONS: tuple[str, ...] = ("percent", "fixed", "set", "none", "multiply", "formula")

MAX_CONDITION_DEPTH = 4


class LeafCondition(StrictModel):
    attr: Key
    op: Op
    value: ScalarOrList = None

    @model_validator(mode="after")
    def _check_value_shape(self) -> "LeafCondition":
        v = self.value
        if self.op == "between":
            if not isinstance(v, list) or len(v) != 2:
                raise ValueError('"between" needs a list of exactly two values [low, high]')
        elif self.op in LIST_OPS:
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError(f'"{self.op}" needs a non-empty list of values')
        elif isinstance(v, list):
            raise ValueError(f'"{self.op}" compares against a single value, not a list')
        return self


class AllCondition(StrictModel):
    all: list["Condition"]


class AnyCondition(StrictModel):
    any: list["Condition"]


class NotCondition(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True, serialize_by_alias=True)

    not_: "Condition" = Field(alias="not")


def _condition_kind(v: Any) -> Optional[str]:
    if isinstance(v, dict):
        for key in ("attr", "all", "any", "not"):
            if key in v:
                return {"attr": "leaf", "all": "all", "any": "any", "not": "not"}[key]
        return None
    if isinstance(v, LeafCondition):
        return "leaf"
    if isinstance(v, AllCondition):
        return "all"
    if isinstance(v, AnyCondition):
        return "any"
    if isinstance(v, NotCondition):
        return "not"
    return None


Condition = Annotated[
    Union[
        Annotated[LeafCondition, Tag("leaf")],
        Annotated[AllCondition, Tag("all")],
        Annotated[AnyCondition, Tag("any")],
        Annotated[NotCondition, Tag("not")],
    ],
    Discriminator(
        _condition_kind,
        custom_error_type="invalid_condition",
        custom_error_message='A condition must have one of the keys "attr", "all", "any" or "not"',
    ),
]

AllCondition.model_rebuild()
AnyCondition.model_rebuild()
NotCondition.model_rebuild()

ALWAYS: AllCondition = AllCondition(all=[])


def condition_depth(c: Condition) -> int:
    if isinstance(c, LeafCondition):
        return 1
    if isinstance(c, AllCondition):
        return 1 + max((condition_depth(x) for x in c.all), default=0)
    if isinstance(c, AnyCondition):
        return 1 + max((condition_depth(x) for x in c.any), default=0)
    return 1 + condition_depth(c.not_)


class RuleAction(StrictModel):
    type: ActionType
    # percent: percentage points (+12 = +12 %, -10 = -10 %). fixed: signed amount. set: target price.
    value: Optional[float] = None
    expression: Optional[str] = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _check(self) -> "RuleAction":
        if self.type == "formula":
            if not self.expression or not self.expression.strip():
                raise ValueError('a "formula" action needs a non-empty expression')
        elif self.type == "none":
            pass
        else:
            if self.value is None:
                raise ValueError(f'a "{self.type}" action needs a numeric value')
            if self.expression is not None:
                raise ValueError(f'a "{self.type}" action must not have an expression')
        return self


class Rule(StrictModel):
    id: Id
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)
    condition: Condition = Field(default_factory=lambda: AllCondition(all=[]))
    action: RuleAction
    priority: int = Field(ge=0, le=100000)
    stage: Key
    enabled: bool = True
    effective_from: Optional[IsoDate] = None
    effective_to: Optional[IsoDate] = None
    customer_message: Optional[str] = Field(default=None, max_length=300)
    explanation_template: Optional[str] = Field(default=None, max_length=300)
    policy_ref: Optional[str] = Field(default=None, max_length=200)
    assumptions: list[str] = Field(default_factory=list)
    created_by: str = "system"
    created_at: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "Rule":
        if condition_depth(self.condition) > MAX_CONDITION_DEPTH:
            raise ValueError(f"condition is nested deeper than {MAX_CONDITION_DEPTH} levels")
        return self


def iter_leaves(c: Condition):
    if isinstance(c, LeafCondition):
        yield c
    elif isinstance(c, AllCondition):
        for x in c.all:
            yield from iter_leaves(x)
    elif isinstance(c, AnyCondition):
        for x in c.any:
            yield from iter_leaves(x)
    elif isinstance(c, NotCondition):
        yield from iter_leaves(c.not_)


__all__ = [
    "ALWAYS",
    "ActionType",
    "AllCondition",
    "AnyCondition",
    "Condition",
    "LeafCondition",
    "NotCondition",
    "Op",
    "Rule",
    "RuleAction",
    "SUPPORTED_ACTIONS",
    "SUPPORTED_OPS",
    "Scalar",
    "condition_depth",
    "iter_leaves",
]
