"""
test_reproducibility.py -- Tests for deterministic reproducibility and statistical uncertainty estimation.
"""

import pytest

from customer_agent_sim import Offer, ValidationConfig, validate_offer
from customer_agent_sim.metrics import wilson_score_interval


def test_wilson_score_interval_properties():
    """Verify statistical properties of Wilson score interval."""
    # Zero successes
    lower0, upper0 = wilson_score_interval(0, 100, 0.95)
    assert lower0 == 0.0
    assert 0.0 < upper0 < 0.10

    # All successes
    lower100, upper100 = wilson_score_interval(100, 100, 0.95)
    assert 0.90 < lower100 < 1.0
    assert upper100 == 1.0

    # 50% success
    low, high = wilson_score_interval(50, 100, 0.95)
    assert low < 0.50 < high
    assert pytest.approx(0.50 - low, rel=1e-3) == high - 0.50


def test_end_to_end_validation_reproducibility():
    """Verify that validate_offer produces 100% bit-exact identical reports on repeated runs with fixed seed."""
    offer = Offer(
        price=105.0,
        base_price=100.0,
        perceived_value=115.0,
        competitor_price=102.0,
        urgency=0.3,
        scarcity=0.2,
    )
    candidates = [90.0, 100.0, 105.0, 115.0]
    config = ValidationConfig(population_size=400, seed=42)

    report1 = validate_offer(offer=offer, ml_probability=0.68, candidate_prices=candidates, config=config)
    report2 = validate_offer(offer=offer, ml_probability=0.68, candidate_prices=candidates, config=config)

    assert report1.status == report2.status
    assert report1.simulated_conversion == report2.simulated_conversion
    assert report1.probability_gap == report2.probability_gap
    assert report1.ci_lower == report2.ci_lower
    assert report1.ci_upper == report2.ci_upper
    assert report1.n_purchases == report2.n_purchases
    assert report1.ranking_results == report2.ranking_results
    assert report1.reason == report2.reason
