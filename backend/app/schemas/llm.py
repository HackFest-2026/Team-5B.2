"""Flat strategy schema used as the LLM's structured-output target.

It is deliberately simpler than StrategyConfig: no recursion (Gemini's response_schema handles
recursive schemas poorly), no formula actions, no code. The server converts it into a
StrategyConfig and runs the full validator before anything is stored.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import Field

from .common import Key, StrictModel
from .constraints import Constraint
from .rules import Op
from .strategy import ConflictPolicy, OptimizerPolicy

LlmValue = Union[float, str, bool, list[Union[float, str]]]


class LlmLeaf(StrictModel):
    attr: Key
    op: Op
    value: LlmValue
    negate: bool = False


class LlmCondition(StrictModel):
    match: Literal["all", "any"] = "all"
    conditions: list[LlmLeaf] = Field(default_factory=list, max_length=8)


class LlmAction(StrictModel):
    type: Literal["percent", "fixed", "set", "none", "multiply"]  # no formula
    value: Optional[float] = None


class LlmRule(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    condition: LlmCondition = Field(default_factory=LlmCondition)
    action: LlmAction
    priority: int = Field(ge=0, le=1000)
    stage: Key
    customer_message: Optional[str] = Field(default=None, max_length=300)


class LlmStrategyOut(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    objective_text: str = Field(default="", max_length=2000)
    rules: list[LlmRule] = Field(default_factory=list, max_length=30)
    constraints: list[Constraint] = Field(default_factory=list, max_length=12)
    conflict_policy: ConflictPolicy = Field(default_factory=ConflictPolicy)
    optimizer_policy: OptimizerPolicy = Field(default_factory=OptimizerPolicy)
    assumptions: list[str] = Field(default_factory=list)
    market_observations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
