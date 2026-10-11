"""
agent.py -- Customer agent entity representing individual heterogeneous consumers.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional
import numpy as np

from customer_agent_sim.decision_policy import BaseDecisionPolicy, StandardDecisionPolicy
from customer_agent_sim.models import AgentDecision, AgentState, AgentTraits, Offer


class CustomerAgent:
    """
    Simulated customer agent characterized by persistent traits and dynamic state.

    Attributes:
        agent_id: Unique identifier for the agent.
        traits: Stable behavioral preferences and characteristics.
        state: Dynamic financial and situational conditions.
    """

    def __init__(
        self,
        agent_id: str,
        traits: AgentTraits,
        state: AgentState,
    ) -> None:
        self.agent_id = agent_id
        self.traits = traits
        self.state = state

    def evaluate(
        self,
        offer: Offer,
        policy: Optional[BaseDecisionPolicy] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> AgentDecision:
        """
        Evaluate an offer under a decision policy and return the resulting decision.
        """
        if policy is None:
            policy = StandardDecisionPolicy()
        if rng is None:
            rng = np.random.default_rng()
        return policy.evaluate(self, offer, rng)

    def with_state_update(
        self,
        budget_ratio: Optional[float] = None,
        need_intensity: Optional[float] = None,
        reference_price_bias: Optional[float] = None,
        memory: Optional[Any] = None,
    ) -> "CustomerAgent":
        """
        Create a clone of this agent with updated dynamic state attributes.
        Preserves stable traits exactly, ensuring controlled ceteris-paribus scenario testing.
        """
        new_state = AgentState(
            budget_ratio=budget_ratio if budget_ratio is not None else self.state.budget_ratio,
            need_intensity=need_intensity if need_intensity is not None else self.state.need_intensity,
            reference_price_bias=reference_price_bias if reference_price_bias is not None else self.state.reference_price_bias,
            memory=memory if memory is not None else copy.deepcopy(self.state.memory),
        )
        return CustomerAgent(
            agent_id=self.agent_id,
            traits=copy.deepcopy(self.traits),
            state=new_state,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent specification to dictionary."""
        return {
            "agent_id": self.agent_id,
            "traits": {
                "price_sensitivity": self.traits.price_sensitivity,
                "value_orientation": self.traits.value_orientation,
                "risk_aversion": self.traits.risk_aversion,
                "patience": self.traits.patience,
                "loyalty": self.traits.loyalty,
                "decision_noise": self.traits.decision_noise,
            },
            "state": {
                "budget_ratio": self.state.budget_ratio,
                "need_intensity": self.state.need_intensity,
                "reference_price_bias": self.state.reference_price_bias,
            },
        }
