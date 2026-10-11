"""The PriceOptimizer boundary. Anything that predicts P(buy) for candidates implements this.

The pricing engine re-verifies whatever an optimizer returns, so an optimizer can never override
hard constraints. Keeping this boundary small is what lets the ML model and the simulator swap.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

from app.schemas.optimizer import CandidateEvaluation, ModelInfo, OptimizerRequest, OptimizerResponse


class PriceOptimizer(Protocol):
    info: ModelInfo

    def evaluate(self, req: OptimizerRequest) -> OptimizerResponse: ...


def select(evals: Sequence[CandidateEvaluation], objective: str, conversion_floor: float) -> Optional[float]:
    """Highest objective among eligible candidates at or above the conversion floor; ties → lower price."""
    metric = "expected_profit" if objective == "max_expected_profit" else "expected_revenue"
    best: Optional[CandidateEvaluation] = None
    for e in sorted(evals, key=lambda x: x.price):
        if not e.eligible or e.p_buy < conversion_floor:
            continue
        if best is None or getattr(e, metric) > getattr(best, metric) + 1e-12:
            best = e
    return best.price if best else None


def finish(evals: list[CandidateEvaluation], req: OptimizerRequest, model: ModelInfo, warnings: list[str]) -> OptimizerResponse:
    """Mark selection statuses consistently for every optimizer implementation."""
    chosen = select(evals, req.objective, req.conversion_floor)
    for e in evals:
        if not e.eligible:
            e.selection_status = "INELIGIBLE"
        elif e.p_buy < req.conversion_floor:
            e.selection_status = "BELOW_CONVERSION_FLOOR"
        elif chosen is not None and abs(e.price - chosen) < 1e-9:
            e.selection_status = "SELECTED"
        else:
            e.selection_status = "NOT_SELECTED"
    return OptimizerResponse(evaluations=evals, selected_price=chosen, model=model, warnings=warnings)
