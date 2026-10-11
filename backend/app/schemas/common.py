"""Shared primitives for every Monetize360 contract.

All contract models forbid unknown keys, so a typo or an injected field (for example a
caller-supplied ``final_price``) is rejected instead of silently ignored.
"""

from __future__ import annotations

import re
from typing import Annotated, Union

from pydantic import BaseModel, ConfigDict, Field

Scalar = Union[bool, int, float, str, None]
ScalarOrList = Union[Scalar, list[Scalar]]

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,63}$"
KEY_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"

Id = Annotated[str, Field(pattern=ID_PATTERN)]
Key = Annotated[str, Field(pattern=KEY_PATTERN)]
IsoDate = Annotated[str, Field(pattern=DATE_PATTERN)]

_DATE_RE = re.compile(DATE_PATTERN)


def is_iso_date(value: object) -> bool:
    return isinstance(value, str) and bool(_DATE_RE.match(value[:10]))


class StrictModel(BaseModel):
    """Base for all contracts: unknown keys are errors, values are validated on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)


# Keys the engine always puts in the pricing context. Domain attributes may not reuse them.
BUILTIN_KEYS: tuple[str, ...] = ("as_of", "base_price", "cost", "previous_price", "product_id", "customer_id")
# Extra variables visible only inside formula actions.
FORMULA_KEYS: tuple[str, ...] = ("price", "base")
