"""Strategy configuration: the editable artefact the LLM drafts and the business approves."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from .common import Id, StrictModel
from .constraints import Constraint
from .domain import DomainConfig
from .rules import Rule

ConflictMode = Literal["stack", "first_match_per_stage", "highest_priority_wins"]
CombineMode = Literal["additive", "multiplicative"]
Objective = Literal["max_expected_profit", "max_expected_revenue"]
StrategySource = Literal["manual", "llm", "mock"]

DEFAULT_SCENARIOS = ["budget_squeeze", "competitor_undercut", "high_urgency", "perceived_value_drop"]


class ConflictPolicy(StrictModel):
    """How matching rules combine. Ordering is always: stage index, then priority (asc), then rule id.

    - stack: every matching rule applies in that order.
    - first_match_per_stage: per stage, only the first matching rule applies; the rest are overridden.
    - highest_priority_wins: only the single matching rule with the lowest priority number applies.
    """

    mode: ConflictMode = "stack"
    combine: CombineMode = "multiplicative"
    max_total_adjustment_pct: Optional[float] = Field(default=None, gt=0, le=1000)


class CandidatePolicy(StrictModel):
    count: int = Field(default=5, ge=1, le=15)
    offsets_pct: list[float] = Field(default_factory=lambda: [-10.0, -5.0, 0.0, 5.0, 10.0], min_length=1, max_length=15)
    include_base_price: bool = True
    include_bound_pins: bool = True
    show_alternatives: bool = True

    @model_validator(mode="after")
    def _check(self) -> "CandidatePolicy":
        if any(o <= -100 for o in self.offsets_pct):
            raise ValueError("candidate offsets must be above -100%")
        return self


class OptimizerPolicy(StrictModel):
    objective: Objective = "max_expected_profit"
    conversion_floor: float = Field(default=0.10, ge=0, le=1)


class ValidationPolicy(StrictModel):
    minimum_simulated_conversion: float = Field(default=0.05, ge=0, le=1)
    max_allowed_probability_gap: float = Field(default=0.15, ge=0, le=1)
    ranking_tolerance_rank_drop: int = Field(default=2, ge=0)
    ranking_tolerance_profit_gap: float = Field(default=0.12, ge=0, le=1)
    scenarios: list[str] = Field(default_factory=lambda: list(DEFAULT_SCENARIOS))
    personas: Optional[list[Id]] = None


class StrategyConfig(StrictModel):
    domain_id: Id
    name: str = Field(min_length=1, max_length=120)
    objective_text: str = Field(default="", max_length=4000)
    rules: list[Rule] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    conflict_policy: ConflictPolicy = Field(default_factory=ConflictPolicy)
    candidate_policy: CandidatePolicy = Field(default_factory=CandidatePolicy)
    optimizer_policy: OptimizerPolicy = Field(default_factory=OptimizerPolicy)
    validation_policy: ValidationPolicy = Field(default_factory=ValidationPolicy)
    source: StrategySource = "manual"

    @model_validator(mode="after")
    def _check(self) -> "StrategyConfig":
        ids = [r.id for r in self.rules]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate rule id(s): {', '.join(dup)}")
        return self

    def rounding_step(self) -> Optional[float]:
        steps = [c.value for c in self.constraints if c.type == "rounding"]
        return steps[0] if steps else None


class DomainBundle(StrictModel):
    """A domain plus its strategy, as stored in examples/domains/*.json."""

    domain: DomainConfig
    strategy: StrategyConfig

    @model_validator(mode="after")
    def _check(self) -> "DomainBundle":
        if self.strategy.domain_id != self.domain.id:
            raise ValueError(f'strategy.domain_id "{self.strategy.domain_id}" does not match domain.id "{self.domain.id}"')
        return self
