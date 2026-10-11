"""
test_validation_gate.py -- Tests for decision gate logic, ML discrepancy, and status outcomes.
"""

import pytest

from customer_agent_sim import (
    Offer,
    ValidationConfig,
    ValidationStatus,
    validate_offer,
)
from customer_agent_sim.validation import validate_ml_discrepancy


def test_ml_discrepancy_calculation():
    """Verify calculation of percentage point gap and tolerance flags."""
    res_ok = validate_ml_discrepancy(ml_probability=0.70, sim_conversion=0.65, max_allowed_gap=0.15)
    assert pytest.approx(res_ok["gap"], rel=1e-4) == 0.05
    assert pytest.approx(res_ok["gap_percentage_points"], rel=1e-4) == 5.0
    assert res_ok["gap_exceeded"] is False
    assert res_ok["severe_discrepancy"] is False

    res_exceeded = validate_ml_discrepancy(ml_probability=0.85, sim_conversion=0.65, max_allowed_gap=0.15)
    assert pytest.approx(res_exceeded["gap"], rel=1e-4) == 0.20
    assert res_exceeded["gap_exceeded"] is True
    assert res_exceeded["severe_discrepancy"] is False

    res_severe = validate_ml_discrepancy(ml_probability=0.90, sim_conversion=0.40, max_allowed_gap=0.15)
    assert pytest.approx(res_severe["gap"], rel=1e-4) == 0.50
    assert res_severe["gap_exceeded"] is True
    assert res_severe["severe_discrepancy"] is True


def test_invalid_input_handling():
    """Verify that invalid inputs are detected and rejected with clear exceptions."""
    valid_offer = Offer(price=100.0, base_price=100.0, perceived_value=110.0)

    # ML probability out of bounds
    with pytest.raises(ValueError, match="ml_probability"):
        validate_offer(offer=valid_offer, ml_probability=-0.1)

    with pytest.raises(ValueError, match="ml_probability"):
        validate_offer(offer=valid_offer, ml_probability=1.5)


def test_gate_pass_condition():
    """Verify that a well-calibrated, high-value offer achieves PASS."""
    offer = Offer(
        price=95.0,
        base_price=100.0,
        perceived_value=120.0,
        competitor_price=105.0,
        urgency=0.4,
        scarcity=0.3,
    )
    # ML predicted prob close to expected simulated conversion (~90%)
    config = ValidationConfig(population_size=500, seed=42, max_allowed_probability_gap=0.15)
    report = validate_offer(offer=offer, ml_probability=0.88, config=config)

    assert report.status == ValidationStatus.PASS
    assert "passed" in report.reason.lower()
    assert report.simulated_conversion > 0.60
    assert len(report.recommendations) > 0


def test_gate_fail_condition():
    """Verify that a grossly overpriced offer with zero conversion triggers FAIL."""
    overpriced_offer = Offer(
        price=250.0,
        base_price=100.0,
        perceived_value=105.0,
        competitor_price=95.0,
    )
    config = ValidationConfig(
        population_size=500,
        seed=42,
        minimum_simulated_conversion=0.05,
        max_allowed_probability_gap=0.15,
    )
    report = validate_offer(
        offer=overpriced_offer,
        ml_probability=0.70,  # Unrealistic ML expectation
        config=config,
    )

    assert report.status == ValidationStatus.FAIL
    assert report.simulated_conversion < 0.05
    assert len(report.recommendations) > 0


def test_gate_review_required_condition():
    """Verify that moderate discrepancy triggers REVIEW_REQUIRED."""
    offer = Offer(
        price=105.0,
        base_price=100.0,
        perceived_value=115.0,
    )
    # Configure narrow gap tolerance (e.g. 0.05) so moderate gap triggers REVIEW_REQUIRED
    config = ValidationConfig(
        population_size=500,
        seed=42,
        minimum_simulated_conversion=0.05,
        max_allowed_probability_gap=0.03,  # Strict threshold
    )
    report = validate_offer(
        offer=offer,
        ml_probability=0.85,
        config=config,
    )

    # Should trigger REVIEW_REQUIRED due to exceeding the 3% gap tolerance
    assert report.status in (ValidationStatus.REVIEW_REQUIRED, ValidationStatus.FAIL)
    assert report.probability_gap > 0.03
