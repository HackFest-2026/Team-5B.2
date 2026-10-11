"""Hard pricing constraints.

Definitions (stated explicitly because they are easy to confuse):
- margin_floor (gross margin): (price - cost) / price * 100 >= value  ->  price >= cost / (1 - value/100)
- markup_floor:                (price - cost) / cost  * 100 >= value  ->  price >= cost * (1 + value/100)
- max_discount:  price >= base * (1 - value/100)
- max_increase:  price <= base * (1 + value/100)
- max_daily_change: |price - previous_price| / previous_price * 100 <= value (ignored without a previous price)
- floor / ceiling: absolute price bounds
- rounding: candidate prices are rounded to a multiple of this step

Always on (not configurable here): price > 0, and price >= unit cost unless the domain sets
``allow_below_cost``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from .common import StrictModel

ConstraintType = Literal[
    "floor", "ceiling", "margin_floor", "markup_floor", "max_discount", "max_increase", "max_daily_change", "rounding"
]
SUPPORTED_CONSTRAINTS: tuple[str, ...] = (
    "floor", "ceiling", "margin_floor", "markup_floor", "max_discount", "max_increase", "max_daily_change", "rounding",
)


class Constraint(StrictModel):
    type: ConstraintType
    value: float

    @model_validator(mode="after")
    def _check(self) -> "Constraint":
        t, v = self.type, self.value
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("constraint value must be a finite number")
        if t == "rounding" and v <= 0:
            raise ValueError("rounding step must be positive")
        if t in ("floor", "ceiling", "markup_floor", "max_increase", "max_daily_change") and v < 0:
            raise ValueError(f"{t} cannot be negative")
        if t == "margin_floor" and not (0 <= v < 100):
            raise ValueError("margin_floor is a gross margin % and must be in [0, 100)")
        if t == "max_discount" and not (0 <= v <= 100):
            raise ValueError("max_discount must be between 0 and 100")
        return self
