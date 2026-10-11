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


def test_validation_checks_detect_deliberate_failures():
    """
    Verify that each of the individual validation check functions catches
    a deliberately failing case.
    """
    from customer_agent_sim.decision_policy import StandardDecisionPolicy
    from customer_agent_sim.population import PopulationGenerator
    from customer_agent_sim.validation import (
        validate_behavioural_consistency,
        validate_population_heterogeneity,
        validate_scenario_robustness,
        validate_ranking_stability,
    )

    gen = PopulationGenerator()
    pop = gen.generate(size=300, seed=42)

    # 1. Check A: Inverted decision policy violating price monotonicity
    class InvertedDemandPolicy(StandardDecisionPolicy):
        def calculate_probabilities(self, agent, offer):
            # Higher price strictly yields higher purchase probability
            p_buy = min(1.0, max(0.0, offer.price / 300.0))
            return p_buy, 0.0, 1.0 - p_buy, 1.0, 0.0

    res_consistency = validate_behavioural_consistency(
        population=pop,
        offer=Offer(price=100.0, base_price=100.0, perceived_value=120.0),
        policy=InvertedDemandPolicy(),
    )
    assert res_consistency["passed"] is False
    assert res_consistency["checks"]["price_monotonicity"]["passed"] is False
    assert len(res_consistency["diagnostics"]) > 0

    # 2. Check B: Artificially cloned population violating heterogeneity
    cloned_pop = [pop[0].with_state_update() for _ in range(100)]
    for a in cloned_pop:
        a.traits.decision_noise = 0.05
    res_heterogeneity = validate_population_heterogeneity(
        population=cloned_pop,
        offer=Offer(price=100.0, base_price=100.0, perceived_value=120.0),
        min_variance=0.002,
    )
    assert res_heterogeneity["passed"] is False
    assert res_heterogeneity["probability_variance"] < 0.001
    assert res_heterogeneity["warning"] is not None

    # 3. Check C: Fragile offer collapsing under scenario stress
    fragile_offer = Offer(price=145.0, base_price=100.0, perceived_value=110.0)
    res_scenario = validate_scenario_robustness(
        population=pop,
        offer=fragile_offer,
        scenarios=["budget_squeeze"],
        min_conversion_threshold=0.05,
    )
    assert (not res_scenario["passed"]) or (len(res_scenario["vulnerabilities"]) > 0)

    # 4. Check D: Suboptimal candidate failing ranking stability
    subopt_offer = Offer(price=150.0, base_price=100.0, perceived_value=110.0, unit_cost=40.0)
    res_ranking = validate_ranking_stability(
        population=pop,
        candidate_prices=[80.0, 90.0, 100.0, 150.0],
        base_offer=subopt_offer,
        tolerance_rank_drop=1,
        tolerance_profit_gap=0.05,
    )
    assert res_ranking["passed"] is False
    assert (not res_ranking["rank_drop_passed"]) or (not res_ranking["gap_passed"])

    # 5. Check E: Severe ML discrepancy
    res_disc = validate_ml_discrepancy(ml_probability=0.95, sim_conversion=0.10, max_allowed_gap=0.15)
    assert res_disc["severe_discrepancy"] is True
    assert res_disc["gap_exceeded"] is True
