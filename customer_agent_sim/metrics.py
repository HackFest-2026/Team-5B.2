"""
metrics.py -- Statistical metrics, uncertainty estimation, and evaluation routines.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate the Wilson score confidence interval for a binomial proportion.

    Handles small samples and boundary proportions (p ~ 0 or 1) accurately without
    overshooting the [0, 1] interval.

    Args:
        successes: Number of simulated successes (purchases).
        total: Total number of trials (population size).
        confidence: Desired confidence level (default 0.95).

    Returns:
        (lower_bound, upper_bound)
    """
    if total <= 0:
        return 0.0, 0.0

    # Two-sided standard normal quantiles
    z_table = {
        0.90: 1.644853,
        0.95: 1.959964,
        0.98: 2.326348,
        0.99: 2.575829,
    }
    z = z_table.get(round(confidence, 2), 1.959964)

    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    spread = (z / denom) * math.sqrt(
        (p_hat * (1.0 - p_hat) / total) + (z**2) / (4 * (total**2))
    )

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return lower, upper


def calculate_variance(values: List[float]) -> float:
    """Calculate sample variance of a float list."""
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(np.var(arr, ddof=1))


def spearman_rank_correlation(ranks_a: List[int], ranks_b: List[int]) -> float:
    """
    Calculate Spearman's rank correlation coefficient between two equal-length rank sequences.
    """
    n = len(ranks_a)
    if n != len(ranks_b) or n < 2:
        return 1.0

    d_sq = sum((ra - rb) ** 2 for ra, rb in zip(ranks_a, ranks_b))
    rho = 1.0 - (6.0 * d_sq) / (n * (n**2 - 1))
    return float(np.clip(rho, -1.0, 1.0))
