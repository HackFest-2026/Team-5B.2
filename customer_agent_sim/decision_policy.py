"""
decision_policy.py -- Interpretable, probabilistic behavioral decision policies.

Provides:
  - BaseDecisionPolicy: Abstract interface for agent decision engines.
  - StandardDecisionPolicy: Transparent behavioral model incorporating:
      * Reference-price comparison & loss aversion
      * Perceived value surplus
      * Hard/soft budget constraints
      * Competitor price contrast & loyalty shielding
      * Urgency and scarcity dynamics
      * Risk aversion and patience trade-offs
      * Multinomial logit choice probabilities (BUY, DEFER, REJECT)

All behavioural coefficients are explicit simulation assumptions, documented
and parameterizable rather than hardcoded empirical claims.
"""

from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np

from customer_agent_sim.models import AgentDecision, DecisionOutcome, Offer

if TYPE_CHECKING:
    from customer_agent_sim.agent import CustomerAgent


class BaseDecisionPolicy(abc.ABC):
    """Abstract interface for customer agent decision policies."""

    @abc.abstractmethod
    def evaluate(
        self,
        agent: "CustomerAgent",
        offer: Offer,
        rng: np.random.Generator,
    ) -> AgentDecision:
        """Evaluate an offer from an agent's perspective and sample a decision."""
        pass

    @abc.abstractmethod
    def calculate_probabilities(
        self,
        agent: "CustomerAgent",
        offer: Offer,
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate analytical choice probabilities and latent utilities.

        Returns:
            (p_buy, p_defer, p_reject, utility_buy, utility_defer)
        """
        pass


class StandardDecisionPolicy(BaseDecisionPolicy):
    """
    Standard interpretable behavioral choice policy.

    Mathematical Formulation:
    -------------------------
    1. Effective Anchor Price:
       R_eff = base_price * (1.0 + state.reference_price_bias)

    2. Normalized Price Ratio:
       r = price / max(R_eff, 0.01)

    3. Perceived Value Surplus:
       surplus = (perceived_value - price) / max(R_eff, 0.01)
       U_surplus = traits.value_orientation * surplus * surplus_weight

    4. Price Disutility (with loss aversion for r > 1):
       If r <= 1.0:
           U_price = - traits.price_sensitivity * (r - 1.0) * price_sensitivity_weight
       Else (price exceeds anchor):
           U_price = - traits.price_sensitivity * (r - 1.0) * price_sensitivity_weight * loss_aversion_factor

    5. Budget Constraint Pressure:
       B = base_price * state.budget_ratio
       If price > B:
           excess_ratio = (price - B) / max(B, 0.01)
           U_budget = - (1.0 + traits.risk_aversion) * excess_ratio * budget_penalty_weight
       Else:
           U_budget = 0.0

    6. Competitor Benchmark:
       If competitor_price is present:
           comp_diff = (competitor_price - price) / max(R_eff, 0.01)
           If comp_diff >= 0 (our price is cheaper):
               U_comp = (1.0 + 0.5 * traits.loyalty) * comp_diff * competitor_weight
           Else (competitor is cheaper):
               shield = max(0.2, 2.0 - traits.loyalty)
               U_comp = - shield * abs(comp_diff) * competitor_weight
       Else:
           U_comp = 0.0

    7. Urgency, Scarcity & Need:
       U_urgency = (offer.urgency * state.need_intensity * 1.5 + offer.scarcity * 1.0) * urgency_weight

    8. Risk Disutility on Price Premium:
       If r > 1.0:
           U_risk = - 0.5 * traits.risk_aversion * (r - 1.0)
       Else:
           U_risk = 0.0

    9. Total Latent Buy Utility:
       U_buy = base_intercept + U_surplus + U_price + U_budget + U_comp + U_urgency + U_risk

    10. Latent Defer Utility:
        U_defer = defer_intercept + traits.patience * patience_weight
                  - (offer.urgency * 2.2 + offer.scarcity * 1.8)
                  - 0.5 * state.need_intensity
                  - 0.6 * abs(r - 1.0)

    11. Multinomial Logit Softmax with Decision Noise (tau):
        tau = max(0.05, traits.decision_noise)
        Denom = 1.0 + exp(U_buy / tau) + exp(U_defer / tau)
        P(BUY) = exp(U_buy / tau) / Denom
        P(DEFER) = exp(U_defer / tau) / Denom
        P(REJECT) = 1.0 / Denom  (where baseline U_reject = 0)
    """

    def __init__(
        self,
        base_intercept: float = 0.20,
        defer_intercept: float = -0.50,
        surplus_weight: float = 2.20,
        price_sensitivity_weight: float = 2.50,
        loss_aversion_factor: float = 1.60,
        budget_penalty_weight: float = 6.00,
        competitor_weight: float = 1.80,
        urgency_weight: float = 1.40,
        patience_weight: float = 1.20,
    ) -> None:
        self.base_intercept = base_intercept
        self.defer_intercept = defer_intercept
        self.surplus_weight = surplus_weight
        self.price_sensitivity_weight = price_sensitivity_weight
        self.loss_aversion_factor = loss_aversion_factor
        self.budget_penalty_weight = budget_penalty_weight
        self.competitor_weight = competitor_weight
        self.urgency_weight = urgency_weight
        self.patience_weight = patience_weight

    def calculate_probabilities(
        self,
        agent: "CustomerAgent",
        offer: Offer,
    ) -> Tuple[float, float, float, float, float]:
        traits = agent.traits
        state = agent.state

        # 1. Effective Anchor Price
        r_eff = max(0.01, offer.base_price * (1.0 + state.reference_price_bias))

        # 2. Price Ratio
        r = offer.price / r_eff

        # 3. Perceived Surplus Utility
        norm_surplus = (offer.perceived_value - offer.price) / r_eff
        u_surplus = traits.value_orientation * norm_surplus * self.surplus_weight

        # 4. Price Disutility with Loss Aversion
        if r <= 1.0:
            u_price = -traits.price_sensitivity * (r - 1.0) * self.price_sensitivity_weight
        else:
            u_price = -traits.price_sensitivity * (r - 1.0) * self.price_sensitivity_weight * self.loss_aversion_factor

        # 5. Budget Pressure
        budget = offer.base_price * state.budget_ratio
        if offer.price > budget:
            excess_ratio = (offer.price - budget) / max(0.01, budget)
            u_budget = -(1.0 + traits.risk_aversion) * excess_ratio * self.budget_penalty_weight
        else:
            u_budget = 0.0

        # 6. Competitor Benchmark
        if offer.competitor_price is not None:
            comp_diff = (offer.competitor_price - offer.price) / r_eff
            if comp_diff >= 0.0:
                # Our price is lower than competitor
                u_comp = (1.0 + 0.5 * traits.loyalty) * comp_diff * self.competitor_weight
            else:
                # Competitor is lower
                shield = max(0.2, 2.0 - traits.loyalty)
                u_comp = -shield * abs(comp_diff) * self.competitor_weight
        else:
            u_comp = 0.0

        # 7. Urgency, Scarcity & Need
        u_urgency = (offer.urgency * state.need_intensity * 1.5 + offer.scarcity * 1.0) * self.urgency_weight

        # 8. Risk Disutility on Premium
        if r > 1.0:
            u_risk = -0.5 * traits.risk_aversion * (r - 1.0)
        else:
            u_risk = 0.0

        # 9. Total Latent Buy Utility
        u_buy = self.base_intercept + u_surplus + u_price + u_budget + u_comp + u_urgency + u_risk

        # 10. Latent Defer Utility
        u_defer = (
            self.defer_intercept
            + traits.patience * self.patience_weight
            - (offer.urgency * 2.2 + offer.scarcity * 1.8)
            - 0.5 * state.need_intensity
            - 0.6 * abs(r - 1.0)
        )

        # 11. Softmax Multinomial Probabilities with temperature tau
        tau = max(0.05, traits.decision_noise)
        scaled_u_buy = np.clip(u_buy / tau, -50.0, 50.0)
        scaled_u_defer = np.clip(u_defer / tau, -50.0, 50.0)

        # Baseline U_reject = 0.0
        exp_buy = math.exp(scaled_u_buy)
        exp_defer = math.exp(scaled_u_defer)
        denom = 1.0 + exp_buy + exp_defer

        p_buy = exp_buy / denom
        p_defer = exp_defer / denom
        p_reject = 1.0 / denom

        # Ensure probabilities sum to 1.0 and stay non-negative
        total = p_buy + p_defer + p_reject
        p_buy /= total
        p_defer /= total
        p_reject /= total

        return p_buy, p_defer, p_reject, u_buy, u_defer

    def evaluate(
        self,
        agent: "CustomerAgent",
        offer: Offer,
        rng: np.random.Generator,
    ) -> AgentDecision:
        p_buy, p_defer, p_reject, u_buy, u_defer = self.calculate_probabilities(agent, offer)

        # Stochastic choice using common random numbers / generator
        outcome_idx = rng.choice(3, p=[p_buy, p_defer, p_reject])
        outcomes = (DecisionOutcome.BUY, DecisionOutcome.DEFER, DecisionOutcome.REJECT)
        selected_outcome = outcomes[outcome_idx]

        return AgentDecision(
            agent_id=agent.agent_id,
            outcome=selected_outcome,
            p_buy=p_buy,
            p_defer=p_defer,
            p_reject=p_reject,
            utility_buy=u_buy,
            utility_defer=u_defer,
        )
