"""
test_agent_behaviour.py -- Tests for agent traits, state validation, and behavioral decision policies.
"""

import pytest
import numpy as np

from customer_agent_sim.agent import CustomerAgent
from customer_agent_sim.decision_policy import StandardDecisionPolicy
from customer_agent_sim.models import (
    AgentDecision,
    AgentState,
    AgentTraits,
    DecisionOutcome,
    Offer,
)


def test_trait_and_state_validation():
    """Verify input validation for agent traits and states."""
    # Valid trait instantiation
    traits = AgentTraits(price_sensitivity=1.2, value_orientation=0.9, risk_aversion=1.1)
    assert traits.price_sensitivity == 1.2

    # Negative trait must raise ValueError
    with pytest.raises(ValueError, match="price_sensitivity"):
        AgentTraits(price_sensitivity=-0.5)

    with pytest.raises(ValueError, match="decision_noise"):
        AgentTraits(decision_noise=0.0)

    # Valid state instantiation
    state = AgentState(budget_ratio=1.5, need_intensity=1.2, reference_price_bias=0.1)
    assert state.budget_ratio == 1.5

    # Non-positive budget ratio must raise ValueError
    with pytest.raises(ValueError, match="budget_ratio"):
        AgentState(budget_ratio=0.0)

    # Out of range reference bias
    with pytest.raises(ValueError, match="reference_price_bias"):
        AgentState(reference_price_bias=3.5)


def test_offer_validation():
    """Verify input validation for domain-neutral Offer schema."""
    valid_offer = Offer(price=100.0, base_price=100.0, perceived_value=110.0)
    assert valid_offer.price == 100.0

    # Negative or zero price
    with pytest.raises(ValueError, match="price"):
        Offer(price=-10.0, base_price=100.0, perceived_value=100.0)

    with pytest.raises(ValueError, match="base_price"):
        Offer(price=100.0, base_price=0.0, perceived_value=100.0)

    # Out of bounds urgency / scarcity
    with pytest.raises(ValueError, match="urgency"):
        Offer(price=100.0, base_price=100.0, perceived_value=100.0, urgency=1.5)

    with pytest.raises(ValueError, match="scarcity"):
        Offer(price=100.0, base_price=100.0, perceived_value=100.0, scarcity=-0.2)


def test_probabilistic_decision_outputs():
    """Verify choice probabilities and decision outcomes conform to valid ranges and sums."""
    agent = CustomerAgent(
        agent_id="test_01",
        traits=AgentTraits(price_sensitivity=1.2, decision_noise=0.3),
        state=AgentState(budget_ratio=1.3, need_intensity=1.0),
    )
    offer = Offer(price=95.0, base_price=100.0, perceived_value=110.0)
    policy = StandardDecisionPolicy()
    rng = np.random.default_rng(42)

    decision = agent.evaluate(offer, policy=policy, rng=rng)
    assert isinstance(decision, AgentDecision)
    assert decision.outcome in (DecisionOutcome.BUY, DecisionOutcome.DEFER, DecisionOutcome.REJECT)

    # Probabilities must be within [0, 1] and sum to 1.0
    assert 0.0 <= decision.p_buy <= 1.0
    assert 0.0 <= decision.p_defer <= 1.0
    assert 0.0 <= decision.p_reject <= 1.0
    total_p = decision.p_buy + decision.p_defer + decision.p_reject
    assert abs(total_p - 1.0) < 1e-6


def test_behavioural_consistency_price_increase():
    """Verify that increasing candidate price reduces purchase probability ceteris paribus."""
    agent = CustomerAgent(
        agent_id="test_price_sensitive",
        traits=AgentTraits(price_sensitivity=2.0, value_orientation=1.0, decision_noise=0.2),
        state=AgentState(budget_ratio=1.5, need_intensity=1.0),
    )
    policy = StandardDecisionPolicy()

    low_price_offer = Offer(price=80.0, base_price=100.0, perceived_value=110.0)
    high_price_offer = Offer(price=140.0, base_price=100.0, perceived_value=110.0)

    p_low_buy, _, _, _, _ = policy.calculate_probabilities(agent, low_price_offer)
    p_high_buy, _, _, _, _ = policy.calculate_probabilities(agent, high_price_offer)

    assert p_low_buy > p_high_buy


def test_behavioural_consistency_perceived_value():
    """Verify that higher perceived value increases purchase probability."""
    agent = CustomerAgent(
        agent_id="test_value_seeker",
        traits=AgentTraits(price_sensitivity=1.0, value_orientation=2.0, decision_noise=0.2),
        state=AgentState(budget_ratio=1.5, need_intensity=1.0),
    )
    policy = StandardDecisionPolicy()

    low_val_offer = Offer(price=100.0, base_price=100.0, perceived_value=85.0)
    high_val_offer = Offer(price=100.0, base_price=100.0, perceived_value=135.0)

    p_low_buy, _, _, _, _ = policy.calculate_probabilities(agent, low_val_offer)
    p_high_buy, _, _, _, _ = policy.calculate_probabilities(agent, high_val_offer)

    assert p_high_buy > p_low_buy


def test_budget_constraint_penalty():
    """Verify that pricing exceeding customer budget applies steep penalty."""
    agent = CustomerAgent(
        agent_id="test_tight_budget",
        traits=AgentTraits(price_sensitivity=1.0, risk_aversion=1.5, decision_noise=0.2),
        state=AgentState(budget_ratio=0.90),  # Budget is $90
    )
    policy = StandardDecisionPolicy()

    affordable_offer = Offer(price=85.0, base_price=100.0, perceived_value=110.0)
    unaffordable_offer = Offer(price=120.0, base_price=100.0, perceived_value=110.0)

    p_aff_buy, _, _, u_aff_buy, _ = policy.calculate_probabilities(agent, affordable_offer)
    p_unaff_buy, _, _, u_unaff_buy, _ = policy.calculate_probabilities(agent, unaffordable_offer)

    assert u_aff_buy > u_unaff_buy
    assert p_aff_buy > p_unaff_buy
