"""
test_feedback_loop.py -- Tests for feedback diagnostic extraction and bounded loop termination.
"""

import pytest

from customer_agent_sim import (
    Offer,
    ValidationConfig,
    ValidationReport,
    ValidationStatus,
    bounded_validation_pipeline,
    extract_feedback_diagnostics,
    validate_offer,
)


def test_feedback_diagnostics_extraction():
    """Verify conversion of ValidationReport into actionable FeedbackDiagnostic."""
    offer = Offer(price=105.0, base_price=100.0, perceived_value=120.0)
    config = ValidationConfig(population_size=300, seed=42)
    report = validate_offer(offer=offer, ml_probability=0.70, config=config)

    diag = extract_feedback_diagnostics(report)
    assert diag.status == report.status
    assert diag.suggested_action in (
        "PROCEED",
        "EVALUATE_ALTERNATIVE_CANDIDATE",
        "FALLBACK_TO_BASE_PRICE",
        "HUMAN_PRICING_REVIEW",
    )
    assert isinstance(diag.to_dict(), dict)


def test_bounded_feedback_loop_termination():
    """Verify that bounded_validation_pipeline strictly halts when retry limit is reached."""
    # A validation function that always returns a failing report
    mock_offer = Offer(price=200.0, base_price=100.0, perceived_value=100.0)

    call_count = 0

    def mock_eval_fn(price: float) -> ValidationReport:
        nonlocal call_count
        call_count += 1
        return validate_offer(
            offer=Offer(price=price, base_price=100.0, perceived_value=100.0),
            ml_probability=0.80,
            config=ValidationConfig(population_size=100, seed=42),
        )

    candidates = [190.0, 180.0, 170.0, 160.0, 150.0]
    max_retries = 3

    final_report, diag = bounded_validation_pipeline(
        validation_fn=mock_eval_fn,
        candidate_prices=candidates,
        max_retries=max_retries,
    )

    # Must terminate after exactly max_retries attempts
    assert call_count == max_retries
    assert diag.retry_count == max_retries
    assert "exhausted" in final_report.reason.lower()
    assert diag.suggested_action == "HUMAN_PRICING_REVIEW"


def test_feedback_diagnostics_respects_custom_minimum_conversion_threshold():
    """
    Regression test: Verify that extract_feedback_diagnostics correctly flags
    conversion_threshold_failed when simulated conversion is e.g. 0.07 and the
    configured minimum_simulated_conversion is 0.10 (not falling back to 0.05).
    """
    # Offer that converts at ~7.5% (strictly between 5% and 10%)
    offer = Offer(price=122.0, base_price=100.0, perceived_value=110.0, urgency=0.17)
    # Set non-default minimum conversion threshold of 0.10 (10%)
    config = ValidationConfig(
        population_size=400,
        seed=42,
        minimum_simulated_conversion=0.10,
    )
    report = validate_offer(offer=offer, ml_probability=0.08, config=config)

    # Simulated conversion should be below 0.10, but strictly above 0.05
    assert 0.05 < report.simulated_conversion < 0.10

    # Diagnostic MUST flag conversion_threshold_failed as True because 0.075 < 0.10
    diag = extract_feedback_diagnostics(report)
    assert diag.conversion_threshold_failed is True

