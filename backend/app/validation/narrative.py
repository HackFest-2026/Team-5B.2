"""Optional: let an LLM rephrase persona concerns. Numbers and reaction categories never change.

Validated with Pydantic; on any failure the template text is kept.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

from app.llm.base import JsonLlm
from app.schemas.persona import ShadowValidation

log = logging.getLogger(__name__)


class _Line(BaseModel):
    persona_id: str
    price: float
    main_concern: str = Field(max_length=240)
    explanation: str = Field(max_length=400)
    suggestion: Optional[str] = Field(default=None, max_length=240)


class _Out(BaseModel):
    lines: list[_Line]


SYSTEM = ("You rewrite short, plain-English feedback for simulated customer personas. Keep every number exactly as given. "
          "Do not change the reaction category. These are synthetic agents, not real customers; never claim otherwise.")


def rewrite(validation: ShadowValidation, client: Optional[JsonLlm], price: Optional[float]) -> ShadowValidation:
    if client is None or not validation.personas or price is None:
        return validation
    rows = [r for row in validation.personas for r in row if abs(r.price - price) < 1e-6]
    payload = [{"persona_id": r.persona_id, "price": r.price, "reaction": r.reaction, "buy": r.buy_share,
                "defer": r.defer_share, "reject": r.reject_share, "driver": r.concern_driver,
                "main_concern": r.main_concern, "suggestion": r.suggestion} for r in rows]
    try:
        out = _Out.model_validate(client.generate_json(SYSTEM, json.dumps(payload), _Out))
    except Exception as e:  # noqa: BLE001 - optional enhancement; keep template text
        log.warning("persona narrative rewrite failed: %s", e)
        return validation
    by_key = {(line.persona_id, round(line.price, 6)): line for line in out.lines}
    for row in validation.personas:
        for r in row:
            line = by_key.get((r.persona_id, round(r.price, 6)))
            if line:
                r.main_concern, r.explanation, r.suggestion = line.main_concern, line.explanation, line.suggestion
                r.source = "agent_sim+llm_text"
    return validation
