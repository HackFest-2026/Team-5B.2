"""Domain configuration: what is being priced and which inputs exist.

Industry vocabulary lives only in domain configs (JSON). Attributes keep their domain names
and may map to a reusable *semantic signal* that downstream components understand.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator

from .common import BUILTIN_KEYS, FORMULA_KEYS, Id, Key, Scalar, StrictModel, is_iso_date

SemanticSignal = Literal[
    "basePrice", "unitCost", "competitorPrice", "perceivedValue",
    "customerSensitivity", "customerLoyalty", "urgency", "scarcity", "demandPressure",
    "competitorDensity", "promotionActive", "timeDecay",
    "capacityUtilization", "supplyAvailability", "risk",
]
# basePrice and unitCost always come from the product record.
PRODUCT_SIGNALS = ("basePrice", "unitCost")

AttributeType = Literal["number", "text", "boolean", "date", "enum"]
AttributeSource = Literal["request", "profile", "inventory", "computed"]


class Normalize(StrictModel):
    """Maps a raw value onto a signal scale: clamp((x-min)/(max-min)), optionally invert, then scale."""

    min: float
    max: float
    invert: bool = False
    target_range: tuple[float, float] = (0.0, 1.0)

    @model_validator(mode="after")
    def _check(self) -> "Normalize":
        if not self.min < self.max:
            raise ValueError("normalize.min must be below normalize.max")
        if not self.target_range[0] < self.target_range[1]:
            raise ValueError("normalize.target_range must be [low, high] with low < high")
        return self


class Attribute(StrictModel):
    key: Key
    label: str = Field(min_length=1, max_length=80)
    type: AttributeType
    source: AttributeSource
    semantic: Optional[SemanticSignal] = None
    unit: Optional[str] = Field(default=None, max_length=24)
    required: bool = False
    customer_input: bool = False
    allowed_values: Optional[list[str]] = None
    default: Scalar = None
    expression: Optional[str] = Field(default=None, max_length=500)
    normalize: Optional[Normalize] = None
    min: Optional[float] = None
    max: Optional[float] = None
    description: Optional[str] = Field(default=None, max_length=500)
    stage: Optional[Key] = None
    synonyms: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "Attribute":
        if self.key in BUILTIN_KEYS or self.key in FORMULA_KEYS:
            raise ValueError(f'"{self.key}" is a reserved key')
        if self.type == "enum" and not self.allowed_values:
            raise ValueError("enum attributes need allowed_values")
        if self.type != "enum" and self.allowed_values:
            raise ValueError("allowed_values is only valid for enum attributes")
        if self.source == "computed" and not (self.expression and self.expression.strip()):
            raise ValueError("computed attributes need an expression")
        if self.source != "computed" and self.expression:
            raise ValueError("only computed attributes may have an expression")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min must not exceed max")
        if self.semantic in PRODUCT_SIGNALS:
            raise ValueError(f"{self.semantic} is provided by the product record and cannot be mapped to an attribute")
        if self.normalize and self.type not in ("number", "boolean"):
            raise ValueError("normalize only applies to number or boolean attributes")
        if self.customer_input and self.source not in ("request",):
            raise ValueError("customer_input attributes must have source 'request'")
        self._check_default()
        if self.expression:
            from app.engine.expression import check_expression

            err = check_expression(self.expression)
            if err:
                raise ValueError(f"invalid expression: {err}")
        return self

    def _check_default(self) -> None:
        d = self.default
        if d is None:
            return
        ok = {
            "number": isinstance(d, (int, float)) and not isinstance(d, bool),
            "boolean": isinstance(d, bool),
            "text": isinstance(d, str),
            "enum": isinstance(d, str) and d in (self.allowed_values or []),
            "date": is_iso_date(d),
        }[self.type]
        if not ok:
            raise ValueError(f"default {d!r} does not match type {self.type}")


class OutputUnit(StrictModel):
    label: str = Field(min_length=1, max_length=40)
    prefix: str = ""
    suffix: str = ""
    decimals: int = Field(default=2, ge=0, le=6)


AgentTrait = Literal[
    "price_sensitivity", "value_orientation", "risk_aversion", "patience", "loyalty",
    "decision_noise", "budget_ratio", "need_intensity", "reference_price_bias",
]


class TraitShift(StrictModel):
    """Shift applied to simulated customer agents (customer_agent_sim trait scales, not ML 0..1 scales)."""

    trait: AgentTrait
    op: Literal["multiply", "set_range"]
    value: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None

    @model_validator(mode="after")
    def _check(self) -> "TraitShift":
        if self.op == "multiply":
            if self.value is None or self.value <= 0:
                raise ValueError("multiply needs a positive value")
        else:
            if self.low is None or self.high is None or self.low > self.high:
                raise ValueError("set_range needs low <= high")
        return self


class MlOverrides(StrictModel):
    customer_sensitivity: float = Field(ge=0, le=1)
    customer_loyalty_score: float = Field(ge=0, le=1)
    urgency_score: float = Field(ge=0, le=1)


class Persona(StrictModel):
    id: Id
    name: str = Field(min_length=1, max_length=60)
    description: str = Field(default="", max_length=500)
    preferences: list[str] = Field(default_factory=list)
    share: float = Field(default=1.0, ge=0)
    cohort: list[TraitShift] = Field(default_factory=list)
    ml_overrides: Optional[MlOverrides] = None


class Product(StrictModel):
    id: Id
    name: str = Field(min_length=1, max_length=120)
    base_price: float = Field(gt=0)
    cost: float = Field(ge=0)
    inventory: dict[Key, Scalar] = Field(default_factory=dict)
    previous_price: Optional[float] = Field(default=None, gt=0)


class DomainConfig(StrictModel):
    id: Id
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=2000)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    output_unit: OutputUnit
    product_noun: str = "Product"
    attributes: list[Attribute] = Field(default_factory=list)
    pipeline: list[Key] = Field(min_length=1)
    personas: list[Persona] = Field(default_factory=list)
    products: list[Product] = Field(default_factory=list)
    defaults: dict[SemanticSignal, float] = Field(default_factory=dict)
    allow_below_cost: bool = False

    @field_validator("pipeline")
    @classmethod
    def _unique_pipeline(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("pipeline stages must be unique")
        return v

    @model_validator(mode="after")
    def _check(self) -> "DomainConfig":
        _unique([a.key for a in self.attributes], "attribute key")
        _unique([p.id for p in self.products], "product id")
        _unique([p.id for p in self.personas], "persona id")
        sems = [a.semantic for a in self.attributes if a.semantic]
        dup = sorted({s for s in sems if sems.count(s) > 1})
        if dup:
            raise ValueError(f"semantic signal(s) mapped more than once: {', '.join(dup)}")
        for a in self.attributes:
            if a.stage and a.stage not in self.pipeline:
                raise ValueError(f'attribute "{a.key}" refers to stage "{a.stage}", which is not in the pipeline')
        return self

    def attribute(self, key: str) -> Optional[Attribute]:
        return next((a for a in self.attributes if a.key == key), None)

    def product(self, product_id: str) -> Optional[Product]:
        return next((p for p in self.products if p.id == product_id), None)


def _unique(values: list[str], what: str) -> None:
    dup = sorted({v for v in values if values.count(v) > 1})
    if dup:
        raise ValueError(f"duplicate {what}(s): {', '.join(dup)}")
