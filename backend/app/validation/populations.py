"""Cached agent populations: the shared baseline population and one cohort per named persona.

Generating 1,000 agents costs ~40 ms; caching them keeps validation latency predictable.
Cohorts apply a persona's trait shifts to a seeded population in our code, so the simulator
module itself is never modified.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from app.schemas.domain import Persona, TraitShift

from .sim import cas

STATE_FIELDS = ("budget_ratio", "need_intensity", "reference_price_bias")
LIMITS = {
    "price_sensitivity": (0.1, 4.0), "value_orientation": (0.1, 3.0), "risk_aversion": (0.1, 3.0), "patience": (0.1, 3.0),
    "loyalty": (0.1, 2.5), "decision_noise": (0.05, 1.5), "budget_ratio": (0.3, 3.5), "need_intensity": (0.1, 2.5),
    "reference_price_bias": (-0.4, 0.4),
}


@lru_cache(maxsize=16)
def base_population(size: int, seed: int) -> tuple:
    return tuple(cas.PopulationGenerator().generate(size=size, seed=seed))


def _apply(agents: list, shifts: list[TraitShift], seed: int) -> list:
    rng = np.random.default_rng(seed)
    out = []
    draws = {s.trait: rng.uniform(s.low, s.high, size=len(agents)) for s in shifts if s.op == "set_range"}
    for i, a in enumerate(agents):
        traits = dict(price_sensitivity=a.traits.price_sensitivity, value_orientation=a.traits.value_orientation,
                      risk_aversion=a.traits.risk_aversion, patience=a.traits.patience, loyalty=a.traits.loyalty,
                      decision_noise=a.traits.decision_noise)
        state = dict(budget_ratio=a.state.budget_ratio, need_intensity=a.state.need_intensity,
                     reference_price_bias=a.state.reference_price_bias)
        for s in shifts:
            target = state if s.trait in STATE_FIELDS else traits
            value = target[s.trait] * s.value if s.op == "multiply" else float(draws[s.trait][i])
            lo, hi = LIMITS[s.trait]
            target[s.trait] = float(min(max(value, lo), hi))
        out.append(cas.CustomerAgent(a.agent_id, cas.AgentTraits(**traits), cas.AgentState(**state)))
    return out


@lru_cache(maxsize=64)
def _cohort_cached(persona_id: str, shifts_json: str, size: int, seed: int) -> tuple:
    shifts = [TraitShift.model_validate(s) for s in json.loads(shifts_json)]
    base = list(cas.PopulationGenerator().generate(size=size, seed=seed))
    return tuple(_apply(base, shifts, seed + 1))


def cohort(persona: Persona, size: int, seed: int) -> list:
    shifts_json = json.dumps([s.model_dump() for s in persona.cohort], sort_keys=True)
    return list(_cohort_cached(persona.id, shifts_json, size, seed))
