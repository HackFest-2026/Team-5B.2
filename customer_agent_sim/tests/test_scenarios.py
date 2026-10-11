"""
test_scenarios.py -- Tests for scenario definitions, common population preservation, and stress testing.
"""

import pytest

from customer_agent_sim.models import Offer
from customer_agent_sim.population import PopulationGenerator
from customer_agent_sim.scenarios import (
    STANDARD_SCENARIOS,
    apply_scenario,
    evaluate_scenarios,
)


def test_scenario_population_reuse():
    """Verify that applying a scenario updates dynamic state while strictly preserving agent traits."""
    gen = PopulationGenerator()
    population = gen.generate(size=50, seed=42)
    offer = Offer(price=100.0, base_price=100.0, perceived_value=110.0)

    mod_pop, mod_offer = apply_scenario(population, offer, "budget_squeeze")

    assert len(mod_pop) == len(population)
    for orig, mod in zip(population, mod_pop):
        # IDs and stable traits must be identical
        assert orig.agent_id == mod.agent_id
        assert orig.traits.price_sensitivity == mod.traits.price_sensitivity
        assert orig.traits.value_orientation == mod.traits.value_orientation
        assert orig.traits.risk_aversion == mod.traits.risk_aversion

        # Budget ratio must reflect the 25% squeeze
        assert pytest.approx(mod.state.budget_ratio, rel=1e-4) == orig.state.budget_ratio * 0.75


def test_scenario_evaluation_directional_consistency():
    """Verify that market scenarios move conversion rates in expected economic directions."""
    gen = PopulationGenerator()
    population = gen.generate(size=500, seed=42)
    offer = Offer(
        price=105.0,
        base_price=100.0,
        perceived_value=115.0,
        competitor_price=105.0,
        urgency=0.3,
        scarcity=0.2,
    )

    scenarios = ["budget_squeeze", "high_urgency", "competitor_undercut", "perceived_value_drop"]
    results = evaluate_scenarios(population, offer, scenarios, seed=42)

    base_conv = results["baseline"]["conversion_rate"]
    squeeze_conv = results["budget_squeeze"]["conversion_rate"]
    urgent_conv = results["high_urgency"]["conversion_rate"]
    comp_undercut_conv = results["competitor_undercut"]["conversion_rate"]
    val_drop_conv = results["perceived_value_drop"]["conversion_rate"]

    # Budget squeeze should decrease purchase propensity
    assert squeeze_conv < base_conv

    # High urgency should increase purchase propensity
    assert urgent_conv > base_conv

    # Competitor discount should reduce our purchase propensity
    assert comp_undercut_conv < base_conv

    # Perceived value drop should reduce purchase propensity
    assert val_drop_conv < base_conv


def test_invalid_scenario_name_raises():
    """Verify that unknown scenario names raise KeyError."""
    gen = PopulationGenerator()
    pop = gen.generate(size=10, seed=42)
    offer = Offer(price=100.0, base_price=100.0, perceived_value=110.0)

    with pytest.raises(KeyError):
        apply_scenario(pop, offer, "non_existent_scenario_123")


def test_offer_without_competitor_price_marks_competitor_scenario_inapplicable():
    """
    Verify that when an offer has no competitor_price (None):
      1. apply_scenario does NOT invent a competitor price.
      2. competitor-specific scenarios are marked applicable=False.
      3. evaluate_scenario_robustness does not treat inapplicable scenarios as vulnerabilities.
    """
    from customer_agent_sim.validation import validate_scenario_robustness

    gen = PopulationGenerator()
    pop = gen.generate(size=100, seed=42)
    no_comp_offer = Offer(
        price=100.0,
        base_price=100.0,
        perceived_value=110.0,
        competitor_price=None,  # No competitor in market
    )

    # 1. apply_scenario keeps competitor_price as None
    _, mod_offer = apply_scenario(pop, no_comp_offer, "competitor_undercut")
    assert mod_offer.competitor_price is None

    # 2. evaluate_scenarios marks competitor_undercut as inapplicable
    scen_results = evaluate_scenarios(pop, no_comp_offer, ["competitor_undercut", "budget_squeeze"], seed=42)
    assert scen_results["competitor_undercut"]["applicable"] is False
    assert "requires competitor_price" in scen_results["competitor_undercut"]["reason"]
    assert scen_results["budget_squeeze"]["applicable"] is True

    # 3. validate_scenario_robustness does not create false vulnerabilities from inapplicable scenarios
    rob_check = validate_scenario_robustness(
        pop,
        no_comp_offer,
        scenarios=["competitor_undercut", "budget_squeeze"],
        min_conversion_threshold=0.01,
        seed=42,
    )
    for vuln in rob_check["vulnerabilities"]:
        assert "competitor_undercut" not in vuln

