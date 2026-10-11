"""
population.py -- Generator and analyzer for heterogeneous customer agent populations.

Generates diverse customer populations from configurable probability distributions
using explicit random generators and seeds for full reproducibility.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from customer_agent_sim.agent import CustomerAgent
from customer_agent_sim.models import AgentState, AgentTraits


class PopulationGenerator:
    """
    Factory for instantiating heterogeneous populations of CustomerAgents.

    Generates non-identical agents using continuous probability distributions:
      - price_sensitivity: Lognormal(mu=0.0, sigma=0.35) -> mean ~1.06, positive-skewed
      - value_orientation: Normal(1.0, 0.30) clipped to [0.1, 3.0]
      - risk_aversion: Beta-derived or Normal(1.0, 0.35) clipped to [0.1, 3.0]
      - patience: Normal(1.0, 0.35) clipped to [0.1, 3.0]
      - loyalty: Beta(2.0, 2.0) scaled to [0.1, 2.5]
      - decision_noise: Truncated Normal(0.40, 0.15) clipped to [0.05, 1.5]
      - budget_ratio: Lognormal(mu=0.15, sigma=0.30) -> mean ~1.22 clipped to [0.3, 3.5]
      - need_intensity: Normal(1.0, 0.30) clipped to [0.1, 2.5]
      - reference_price_bias: Normal(0.0, 0.12) clipped to [-0.4, 0.4]
    """

    def __init__(
        self,
        trait_distribution_params: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.params = trait_distribution_params or {}

    def generate(
        self,
        size: int = 1000,
        seed: Optional[int] = 42,
    ) -> List[CustomerAgent]:
        """
        Generate a heterogeneous population of CustomerAgents.

        Args:
            size: Number of agents to generate (must be >= 1).
            seed: Explicit seed for deterministic reproducibility.

        Returns:
            List of CustomerAgent instances.
        """
        if size <= 0:
            raise ValueError(f"Population size must be at least 1, got {size}")

        rng = np.random.default_rng(seed)

        # 1. Generate stable traits
        ps_mu = self.params.get("price_sensitivity", {}).get("mean", 0.0)
        ps_sigma = self.params.get("price_sensitivity", {}).get("sigma", 0.35)
        price_sensitivities = rng.lognormal(mean=ps_mu, sigma=ps_sigma, size=size)
        price_sensitivities = np.clip(price_sensitivities, 0.1, 4.0)

        vo_mean = self.params.get("value_orientation", {}).get("mean", 1.0)
        vo_std = self.params.get("value_orientation", {}).get("std", 0.30)
        value_orientations = np.clip(rng.normal(vo_mean, vo_std, size=size), 0.1, 3.0)

        ra_mean = self.params.get("risk_aversion", {}).get("mean", 1.0)
        ra_std = self.params.get("risk_aversion", {}).get("std", 0.35)
        risk_aversions = np.clip(rng.normal(ra_mean, ra_std, size=size), 0.1, 3.0)

        pat_mean = self.params.get("patience", {}).get("mean", 1.0)
        pat_std = self.params.get("patience", {}).get("std", 0.35)
        patiences = np.clip(rng.normal(pat_mean, pat_std, size=size), 0.1, 3.0)

        loy_mean = self.params.get("loyalty", {}).get("mean", 1.0)
        loy_std = self.params.get("loyalty", {}).get("std", 0.35)
        loyalties = np.clip(rng.normal(loy_mean, loy_std, size=size), 0.1, 2.5)

        noise_mean = self.params.get("decision_noise", {}).get("mean", 0.40)
        noise_std = self.params.get("decision_noise", {}).get("std", 0.15)
        decision_noises = np.clip(rng.normal(noise_mean, noise_std, size=size), 0.05, 1.5)

        # 2. Generate dynamic states
        br_mean = self.params.get("budget_ratio", {}).get("mean", 0.15)
        br_sigma = self.params.get("budget_ratio", {}).get("sigma", 0.30)
        budget_ratios = np.clip(rng.lognormal(mean=br_mean, sigma=br_sigma, size=size), 0.3, 3.5)

        ni_mean = self.params.get("need_intensity", {}).get("mean", 1.0)
        ni_std = self.params.get("need_intensity", {}).get("std", 0.30)
        need_intensities = np.clip(rng.normal(ni_mean, ni_std, size=size), 0.1, 2.5)

        bias_std = self.params.get("reference_price_bias", {}).get("std", 0.12)
        ref_biases = np.clip(rng.normal(0.0, bias_std, size=size), -0.4, 0.4)

        population: List[CustomerAgent] = []
        for i in range(size):
            traits = AgentTraits(
                price_sensitivity=float(price_sensitivities[i]),
                value_orientation=float(value_orientations[i]),
                risk_aversion=float(risk_aversions[i]),
                patience=float(patiences[i]),
                loyalty=float(loyalties[i]),
                decision_noise=float(decision_noises[i]),
            )
            state = AgentState(
                budget_ratio=float(budget_ratios[i]),
                need_intensity=float(need_intensities[i]),
                reference_price_bias=float(ref_biases[i]),
            )
            population.append(CustomerAgent(agent_id=f"agent_{i:05d}", traits=traits, state=state))

        return population


def split_population_by_trait(
    population: List[CustomerAgent],
    trait_name: str,
    quantile: float = 0.50,
) -> Tuple[List[CustomerAgent], List[CustomerAgent]]:
    """
    Split population into low and high subgroups based on a trait or state quantile.
    """
    vals = [getattr(a.traits, trait_name, getattr(a.state, trait_name, None)) for a in population]
    if any(v is None for v in vals):
        raise ValueError(f"Invalid attribute name '{trait_name}' for splitting")

    cutoff = float(np.quantile(vals, quantile))
    low_group = [a for a, v in zip(population, vals) if v <= cutoff]
    high_group = [a for a, v in zip(population, vals) if v > cutoff]
    return low_group, high_group
