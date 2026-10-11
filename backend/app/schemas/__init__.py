"""Pydantic v2 contracts for Monetize360. Every pipeline stage talks through these types."""

from .common import BUILTIN_KEYS, FORMULA_KEYS, Scalar, StrictModel
from .constraints import SUPPORTED_CONSTRAINTS, Constraint, ConstraintType
from .domain import (
    Attribute,
    DomainConfig,
    MlOverrides,
    Normalize,
    OutputUnit,
    Persona,
    Product,
    SemanticSignal,
    TraitShift,
)
from .optimizer import (
    ML_FEATURE_COLUMNS,
    CandidateEvaluation,
    MLFeatures,
    ModelInfo,
    OptimizerRequest,
    OptimizerResponse,
)
from .persona import PersonaReaction, ShadowValidation, ValidationCheck
from .pricing import (
    Alternative,
    CustomerQuote,
    DecisionCandidate,
    Explanation,
    PricingDecision,
    Provenance,
)
from .pricing_parts import (
    AppliedRule,
    Bounds,
    CheckedCandidate,
    ConstraintEval,
    CustomerProfile,
    GeneratedCandidate,
    OverriddenRule,
    PriceRequest,
    SkippedRule,
    Violation,
)
from .rules import (
    SUPPORTED_ACTIONS,
    SUPPORTED_OPS,
    AllCondition,
    AnyCondition,
    Condition,
    LeafCondition,
    NotCondition,
    Rule,
    RuleAction,
)
from .strategy import (
    CandidatePolicy,
    ConflictPolicy,
    DomainBundle,
    OptimizerPolicy,
    StrategyConfig,
    ValidationPolicy,
)
from .validation import ValidationIssue, ValidationReport

__all__ = [name for name in dir() if not name.startswith("_")]
