"""Customer-agent shadow validation and persona cohorts (advisory only, never changes a price)."""

from .cohorts import default_personas
from .service import ValidationService, reactions_at

__all__ = ["ValidationService", "default_personas", "reactions_at"]
