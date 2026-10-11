"""Persona cohort reactions and customer-agent shadow validation.

Everything here is simulated, advisory feedback. None of it can change a selected price.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from .common import StrictModel

Reaction = Literal["DELIGHTED", "ACCEPTS", "HESITANT", "REJECTS"]
ConcernDriver = Literal["price", "budget", "competitor", "value", "wait", "risk", "none"]


class PersonaReaction(StrictModel):
    persona_id: str
    name: str
    description: str = ""
    preferences: list[str] = Field(default_factory=list)
    price: float
    buy_share: float = Field(ge=0, le=1)
    defer_share: float = Field(ge=0, le=1)
    reject_share: float = Field(ge=0, le=1)
    reaction: Reaction
    main_concern: str
    concern_driver: ConcernDriver
    explanation: str
    suggestion: Optional[str] = None
    source: Literal["agent_sim", "agent_sim+llm_text"] = "agent_sim"
    simulated: Literal[True] = True


CheckKey = Literal[
    "A_behaviour", "B_heterogeneity", "C_scenarios", "D_ranking", "E_model_disagreement", "F_ml_monotonicity"
]


class ValidationCheck(StrictModel):
    key: CheckKey
    passed: Optional[bool] = None
    available: bool = True
    severity: Literal["ok", "review", "fail"]
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


ShadowStatus = Literal["PASS", "FAIL", "REVIEW_REQUIRED", "ERROR", "SKIPPED", "PENDING"]


class ShadowValidation(StrictModel):
    status: ShadowStatus
    simulated_conversion: Optional[float] = None
    ci: Optional[tuple[float, float]] = None
    model_disagreement_pp: Optional[float] = None
    checks: list[ValidationCheck] = Field(default_factory=list)
    scenarios: dict[str, Any] = Field(default_factory=dict)
    feedback: Optional[dict[str, Any]] = None
    personas: Optional[list[list[PersonaReaction]]] = None
    population_size: int = 0
    seed: int = 42
    module_version: str = ""
    error: Optional[str] = None
    reason: Optional[str] = None
    advisory_only: Literal[True] = True
