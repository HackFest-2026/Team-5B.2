"""Shadow validation orchestration. Strictly observational: it never changes a price or status.

Any failure inside the simulator becomes status ERROR and the quote is unaffected.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from app.schemas.domain import Persona
from app.schemas.persona import ShadowValidation, ValidationCheck
from app.schemas.pricing import PricingDecision
from app.schemas.strategy import ValidationPolicy

from .adapter import ml_ladder_check, to_offer
from .cohorts import persona_grid
from .populations import base_population
from .sim import MODULE_VERSION, cas

log = logging.getLogger(__name__)
COHORT_SIZE = 200


def _checks(report, ladder_ok: bool, inversions: list[dict]) -> list[ValidationCheck]:
    c = report.consistency_results
    h = report.heterogeneity_results
    sc = report.scenario_results
    rk = report.ranking_results or {}
    d = report.discrepancy_results
    checks = [
        ValidationCheck(key="A_behaviour", passed=c["passed"], severity="ok" if c["passed"] else "fail",
                        summary="Simulated demand falls as price rises and responds to value." if c["passed"]
                        else "Simulated demand responded implausibly (price ladder or value test failed).",
                        details=c),
        ValidationCheck(key="B_heterogeneity", passed=h["passed"], severity="ok" if h["passed"] else "review",
                        summary=f"Agent purchase probabilities vary (variance {h['probability_variance']:.4f})." if h["passed"]
                        else (h.get("warning") or "Population variance too low."), details=h),
        ValidationCheck(key="C_scenarios", passed=sc["passed"] and not sc["vulnerabilities"],
                        severity="ok" if sc["passed"] and not sc["vulnerabilities"] else "review",
                        summary="Holds up under budget, competitor, urgency and value shocks." if sc["passed"] and not sc["vulnerabilities"]
                        else "; ".join(sc["vulnerabilities"]) or "A stress scenario collapsed conversion.",
                        details={"baseline_conversion": sc["baseline_conversion"], "vulnerabilities": sc["vulnerabilities"]}),
    ]
    if rk.get("available"):
        checks.append(ValidationCheck(
            key="D_ranking", passed=rk["passed"], severity="ok" if rk["passed"] else "review",
            summary=(f"Selected price ranks #{rk['sim_rank']} for simulated agents (best {rk['best_sim_price']:g}, "
                     f"{rk['ranking_metric_name']} gap {rk['metric_gap']:.0%})."), details=rk))
    else:
        checks.append(ValidationCheck(key="D_ranking", available=False, severity="ok",
                                      summary=rk.get("reason", "Ranking check unavailable."), details=rk))
    sev = "fail" if d["severe_discrepancy"] else "review" if d["gap_exceeded"] else "ok"
    checks.append(ValidationCheck(
        key="E_model_disagreement", passed=sev == "ok", severity=sev,  # type: ignore[arg-type]
        summary=(f"ML P(buy) {d['ml_probability']:.0%} vs simulated agents {d['simulated_conversion']:.0%} "
                 f"({d['gap_percentage_points']:.1f} pp). Two synthetic models disagreeing is a review signal, "
                 "not evidence about real customers."), details=d))
    checks.append(ValidationCheck(
        key="F_ml_monotonicity", passed=ladder_ok, severity="ok" if ladder_ok else "review",
        summary="Optimizer P(buy) falls as price rises." if ladder_ok
        else f"Optimizer P(buy) rises with price in {len(inversions)} place(s).", details={"inversions": inversions}))
    return checks


def _scenarios(sc: dict) -> dict:
    out = {}
    for name, data in sc.get("scenarios", {}).items():
        out[name] = {k: data.get(k) for k in ("applicable", "conversion_rate", "reason", "description") if k in data}
    return out


class ValidationService:
    def __init__(self, seed: int = 42, cohort_size: int = COHORT_SIZE):
        self.seed = seed
        self.cohort_size = cohort_size

    def run(self, decision: PricingDecision, policy: ValidationPolicy, personas: Sequence[Persona],
            population_size: int, include_personas: bool = True) -> ShadowValidation:
        if decision.status != "OK" or decision.final_price is None:
            return ShadowValidation(status="SKIPPED", reason="no valid price to validate", population_size=0, seed=self.seed,
                                    module_version=MODULE_VERSION)
        try:
            return self._run(decision, policy, personas, population_size, include_personas)
        except Exception as e:  # noqa: BLE001 - shadow mode: never let validation break pricing
            log.exception("shadow validation failed")
            return ShadowValidation(status="ERROR", error=f"{type(e).__name__}: {e}", population_size=population_size,
                                    seed=self.seed, module_version=MODULE_VERSION)

    def _run(self, decision, policy, personas, population_size, include_personas) -> ShadowValidation:
        offer, mapping_warnings = to_offer(decision)
        selected = next(c for c in decision.candidates if c.selection_status == "SELECTED")
        ranked = [c for c in decision.candidates if c.eligible and c.p_buy is not None]
        config = cas.ValidationConfig(
            population_size=population_size, seed=self.seed,
            minimum_simulated_conversion=policy.minimum_simulated_conversion,
            max_allowed_probability_gap=policy.max_allowed_probability_gap,
            ranking_tolerance_rank_drop=policy.ranking_tolerance_rank_drop,
            ranking_tolerance_profit_gap=policy.ranking_tolerance_profit_gap,
            scenarios=[s for s in policy.scenarios if s in cas.STANDARD_SCENARIOS],
        )
        report = cas.validate_offer(
            offer=offer, ml_probability=float(selected.p_buy),
            candidate_prices=[c.price for c in ranked] if len(ranked) > 1 else None,
            ml_probabilities={c.price: float(c.p_buy) for c in ranked} if len(ranked) > 1 else None,
            config=config, population=list(base_population(population_size, self.seed)),
        )
        feedback = cas.extract_feedback_diagnostics(report).to_dict()
        alt = feedback.get("alternative_candidate_price")
        if alt is not None:
            match = next((c for c in decision.candidates if abs(c.price - alt) < 1e-6), None)
            feedback["alternative_candidate_eligible"] = bool(match and match.eligible)
        feedback["advisory_only"] = True
        ladder_ok, inversions = ml_ladder_check(decision)
        status = report.status.value
        if not ladder_ok and status == "PASS":
            status = "REVIEW_REQUIRED"
        grid = None
        if include_personas and personas:
            prices = sorted(c.price for c in decision.candidates)
            grid = persona_grid(personas, offer, prices, self.cohort_size, self.seed, only=policy.personas)
        return ShadowValidation(
            status=status, simulated_conversion=round(report.simulated_conversion, 4),
            ci=(round(report.ci_lower, 4), round(report.ci_upper, 4)),
            model_disagreement_pp=round(report.probability_gap * 100, 2),
            checks=_checks(report, ladder_ok, inversions), scenarios=_scenarios(report.scenario_results),
            feedback=feedback, personas=grid, population_size=population_size, seed=self.seed,
            module_version=MODULE_VERSION, reason=report.reason + (" | " + "; ".join(mapping_warnings) if mapping_warnings else ""),
        )


def reactions_at(validation: ShadowValidation, price: Optional[float]):
    """Persona reactions at one price (used for the decision's persona_summary)."""
    if not validation.personas or price is None:
        return None
    return [r for row in validation.personas for r in row if abs(r.price - price) < 1e-6] or None
