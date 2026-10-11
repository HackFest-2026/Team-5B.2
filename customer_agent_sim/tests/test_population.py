"""
test_population.py -- Tests for population generation, reproducibility, and heterogeneity.
"""

import numpy as np
import pytest

from customer_agent_sim.models import Offer
from customer_agent_sim.population import PopulationGenerator, split_population_by_trait
from customer_agent_sim.runner import simulate_offer
from customer_agent_sim.validation import validate_population_heterogeneity


def test_reproducible_population_generation():
    """Verify that generation with identical seeds yields identical agents."""
    gen = PopulationGenerator()
    pop_1 = gen.generate(size=50, seed=123)
    pop_2 = gen.generate(size=50, seed=123)

    assert len(pop_1) == len(pop_2)
    for a1, a2 in zip(pop_1, pop_2):
        assert a1.agent_id == a2.agent_id
        assert a1.traits.price_sensitivity == a2.traits.price_sensitivity
        assert a1.traits.value_orientation == a2.traits.value_orientation
        assert a1.state.budget_ratio == a2.state.budget_ratio


def test_different_seeds_produce_different_populations():
    """Verify that different seeds produce distinct agent parameters."""
    gen = PopulationGenerator()
    pop_a = gen.generate(size=30, seed=100)
    pop_b = gen.generate(size=30, seed=200)

    sens_a = [a.traits.price_sensitivity for a in pop_a]
    sens_b = [a.traits.price_sensitivity for a in pop_b]
    assert sens_a != sens_b


def test_population_meaningful_variation():
    """Verify that population traits possess substantive statistical spread."""
    gen = PopulationGenerator()
    pop = gen.generate(size=200, seed=42)

    ps_vals = [a.traits.price_sensitivity for a in pop]
    vo_vals = [a.traits.value_orientation for a in pop]
    br_vals = [a.state.budget_ratio for a in pop]

    # Verify standard deviations are substantial
    assert np.std(ps_vals) > 0.15
    assert np.std(vo_vals) > 0.15
    assert np.std(br_vals) > 0.15


def test_subgroup_behavioral_differences():
    """Verify that agents with different traits exhibit meaningfully different purchase rates."""
    gen = PopulationGenerator()
    pop = gen.generate(size=500, seed=42)
    offer = Offer(price=120.0, base_price=100.0, perceived_value=110.0, urgency=0.25)

    low_sens, high_sens = split_population_by_trait(pop, "price_sensitivity", 0.50)
    m_low = simulate_offer(low_sens, offer, seed=42)
    m_high = simulate_offer(high_sens, offer, seed=42)

    # Low sensitivity customers should purchase at a higher rate than high sensitivity customers
    assert m_low.conversion_rate > m_high.conversion_rate

    # Budget split should also show significant behavioral difference
    low_b, high_b = split_population_by_trait(pop, "budget_ratio", 0.50)
    m_low_b = simulate_offer(low_b, offer, seed=42)
    m_high_b = simulate_offer(high_b, offer, seed=42)
    assert m_high_b.conversion_rate > m_low_b.conversion_rate


def test_detection_of_accidentally_identical_population():
    """Verify that heterogeneity check catches an artificially cloned population."""
    gen = PopulationGenerator()
    pop = gen.generate(size=100, seed=42)

    # Force all agents to identical cloned traits and states
    cloned_agent = pop[0]
    cloned_pop = [cloned_agent.with_state_update() for _ in range(100)]
    # Overwrite decision noise to near zero
    for a in cloned_pop:
        a.traits.decision_noise = 0.05

    offer = Offer(price=100.0, base_price=100.0, perceived_value=110.0)
    check = validate_population_heterogeneity(cloned_pop, offer, min_variance=0.01)

    # Heterogeneity test must flag this near-zero variance
    assert check["probability_variance"] < 0.001
    assert check["passed"] is False
    assert check["warning"] is not None
