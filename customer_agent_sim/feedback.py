"""
feedback.py -- Structured diagnostic interface and bounded feedback loop.

Provides actionable feedback signals for the Monetize360 pipeline to handle
validation failures safely without infinite loops or parameter overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from customer_agent_sim.models import Offer, ValidationConfig, ValidationReport, ValidationStatus


@dataclass
class FeedbackDiagnostic:
    """
    Structured feedback signal consumed by Monetize360 integration.
    """
    status: ValidationStatus
    conversion_threshold_failed: bool
    discrepancy_exceeded: bool
    scenario_robustness_failed: bool
    ranking_unstable: bool
    suggested_action: str
    alternative_candidate_price: Optional[float]
    retry_count: int
    reason: str
    recommended_steps: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "conversion_threshold_failed": self.conversion_threshold_failed,
            "discrepancy_exceeded": self.discrepancy_exceeded,
            "scenario_robustness_failed": self.scenario_robustness_failed,
            "ranking_unstable": self.ranking_unstable,
            "suggested_action": self.suggested_action,
            "alternative_candidate_price": self.alternative_candidate_price,
            "retry_count": self.retry_count,
            "reason": self.reason,
            "recommended_steps": self.recommended_steps,
        }


def extract_feedback_diagnostics(
    report: ValidationReport,
    retry_count: int = 0,
    minimum_simulated_conversion: Optional[float] = None,
) -> FeedbackDiagnostic:
    """
    Extract structured feedback flags and actionable recommendations from a ValidationReport.
    """
    min_conv = (
        minimum_simulated_conversion
        if minimum_simulated_conversion is not None
        else getattr(
            report,
            "minimum_simulated_conversion",
            report.discrepancy_results.get("minimum_simulated_conversion", 0.05),
        )
    )

    conversion_failed = (report.simulated_conversion < min_conv) or (report.ci_upper < min_conv)

    discrepancy_exceeded = bool(report.discrepancy_results.get("gap_exceeded", False))

    scenario_failed = bool(
        not report.scenario_results.get("passed", True)
        or len(report.scenario_results.get("vulnerabilities", [])) > 0
    )

    ranking_res = report.ranking_results or {}
    ranking_unstable = bool(ranking_res.get("available") and not ranking_res.get("passed"))

    alt_price = ranking_res.get("best_sim_price") if ranking_unstable else None

    # Suggested action determination
    if report.status == ValidationStatus.PASS:
        suggested_action = "PROCEED"
    elif report.status == ValidationStatus.FAIL:
        if alt_price is not None:
            suggested_action = "EVALUATE_ALTERNATIVE_CANDIDATE"
        else:
            suggested_action = "FALLBACK_TO_BASE_PRICE"
    else:  # REVIEW_REQUIRED
        if ranking_unstable and alt_price is not None:
            suggested_action = "EVALUATE_ALTERNATIVE_CANDIDATE"
        else:
            suggested_action = "HUMAN_PRICING_REVIEW"

    return FeedbackDiagnostic(
        status=report.status,
        conversion_threshold_failed=conversion_failed,
        discrepancy_exceeded=discrepancy_exceeded,
        scenario_robustness_failed=scenario_failed,
        ranking_unstable=ranking_unstable,
        suggested_action=suggested_action,
        alternative_candidate_price=alt_price,
        retry_count=retry_count,
        reason=report.reason,
        recommended_steps=report.recommendations,
    )


def bounded_validation_pipeline(
    validation_fn: Callable[[float], ValidationReport],
    candidate_prices: List[float],
    max_retries: int = 3,
) -> Tuple[ValidationReport, FeedbackDiagnostic]:
    """
    Safe, bounded candidate re-evaluation loop for the Monetize360 optimizer interface.

    Prevents unbounded looping by enforcing max_retries. If all retries are exhausted
    without achieving PASS, returns REVIEW_REQUIRED.

    Args:
        validation_fn: Callable taking a candidate price and returning a ValidationReport.
        candidate_prices: List of ordered candidate prices to attempt.
        max_retries: Maximum number of re-evaluations permitted.

    Returns:
        (final_report, feedback_diagnostic)
    """
    if max_retries <= 0:
        raise ValueError(f"max_retries must be strictly positive, got {max_retries}")

    attempts = 0
    last_report: Optional[ValidationReport] = None

    for price in candidate_prices:
        attempts += 1
        report = validation_fn(price)
        last_report = report

        if report.status == ValidationStatus.PASS:
            diagnostic = extract_feedback_diagnostics(report, retry_count=attempts - 1)
            return report, diagnostic

        if attempts >= max_retries:
            break

    # If retries exhausted and didn't pass, force REVIEW_REQUIRED if not FAIL
    assert last_report is not None
    if last_report.status == ValidationStatus.PASS:
        forced_status = ValidationStatus.PASS
    else:
        forced_status = ValidationStatus.REVIEW_REQUIRED if last_report.status != ValidationStatus.FAIL else ValidationStatus.FAIL

    last_report.status = forced_status
    last_report.reason += f" [Bounded feedback loop exhausted after {attempts} attempts]."

    diagnostic = extract_feedback_diagnostics(last_report, retry_count=attempts)
    diagnostic.suggested_action = "HUMAN_PRICING_REVIEW"
    return last_report, diagnostic
