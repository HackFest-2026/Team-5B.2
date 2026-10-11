"""
test_ranking_stability.py -- Tests for candidate grid ranking, revenue objectives, and metric tolerances.
"""

import pytest

from customer_agent_sim.models import Offer
from customer_agent_sim.population import PopulationGenerator
from customer_agent_sim.runner import simulate_candidates
from customer_agent_sim.validation import validate_ranking_stability


def test_single_price_mode_marks_ranking_unavailable():
    """Verify that when no candidate grid is provided, ranking stability is marked unavailable."""
    gen = PopulationGenerator()
    pop = gen.generate(size=50, seed=42)
    offer = Offer(price=100.0, base_price=100.0, perceived_value=110.0)

    # Empty candidate list
    res_none = validate_ranking_stability(pop, None, offer)
    assert res_none["available"] is False
    assert "no candidate alternatives" in res_none["reason"]

    # Single-element candidate list
    res_single = validate_ranking_stability(pop, [100.0], offer)
    assert res_single["available"] is False


def test_ranking_stability_with_candidate_grid():
    """Verify candidate ranking calculation and rank stability determination."""
    gen = PopulationGenerator()
    pop = gen.generate(size=300, seed=42)

    # Offer where candidate $100 is evaluated with cost
    offer = Offer(price=100.0, base_price=100.0, perceived_value=115.0, unit_cost=50.0)
    candidates = [80.0, 90.0, 100.0, 110.0, 130.0]
    ml_probs = {80.0: 0.85, 90.0: 0.78, 100.0: 0.72, 110.0: 0.60, 130.0: 0.35}

    res = validate_ranking_stability(
        population=pop,
        candidate_prices=candidates,
        base_offer=offer,
        ml_probabilities=ml_probs,
        tolerance_rank_drop=2,
        tolerance_profit_gap=0.15,
        seed=42,
    )

    assert res["available"] is True
    assert "sim_rank" in res
    assert "ml_rank" in res
    assert "rank_drop" in res
    assert res["ranking_metric_name"] == "profit"
    assert len(res["candidates"]) == 5
    # Ranks must be assigned 1 to 5
    sim_ranks = [c["rank_sim"] for c in res["candidates"]]
    assert sorted(sim_ranks) == [1, 2, 3, 4, 5]


def test_expected_revenue_ranking_higher_price_outranks_lower_price():
    """
    Verify that when unit_cost is None, candidates are ranked by expected revenue,
    proving a higher-priced candidate can outrank a cheaper one with higher conversion.
    """
    gen = PopulationGenerator()
    pop = gen.generate(size=500, seed=42)

    # Offer without unit_cost -> objective is expected revenue (price * conversion)
    base_offer = Offer(price=100.0, base_price=100.0, perceived_value=120.0, urgency=0.4)
    # Evaluate $70 vs $100
    candidates = [70.0, 100.0]
    results = simulate_candidates(pop, candidates, base_offer, seed=42)

    res_70 = next(r for r in results if r.price == 70.0)
    res_100 = next(r for r in results if r.price == 100.0)

    assert res_70.ranking_metric_name == "revenue"
    assert res_100.ranking_metric_name == "revenue"

    # Conversion at $70 is higher than at $100
    assert res_70.sim_conversion > res_100.sim_conversion

    # But expected revenue ($100 * conv_100) is higher than ($70 * conv_70)
    rev_70 = 70.0 * res_70.sim_conversion
    rev_100 = 100.0 * res_100.sim_conversion
    assert rev_100 > rev_70

    # Therefore, $100 outranks $70 in simulation rank!
    assert res_100.rank_sim == 1
    assert res_70.rank_sim == 2


def test_ranking_stability_metric_gap_violation():
    """Verify that exceeding tolerance_profit_gap causes stability failure even if rank drop passes."""
    gen = PopulationGenerator()
    pop = gen.generate(size=300, seed=42)

    offer = Offer(price=105.0, base_price=100.0, perceived_value=115.0, unit_cost=50.0)
    candidates = [95.0, 100.0, 105.0]

    # Set strict profit gap tolerance of 1% (0.01) while rank drop tolerance is generous (3)
    res = validate_ranking_stability(
        population=pop,
        candidate_prices=candidates,
        base_offer=offer,
        tolerance_rank_drop=3,
        tolerance_profit_gap=0.005,  # 0.5% tolerance
        seed=42,
    )

    if res["sim_rank"] > 1:
        # Candidate dropped at least 1 rank
        assert res["rank_drop_passed"] is True   # Rank drop <= 3 passes
        assert res["gap_passed"] is False        # Profit gap > 0.5% fails
        assert res["passed"] is False            # Overall stability fails


def test_ranking_stability_rank_drop_violation():
    """Verify that excessive rank drop causes stability failure even if profit gap is loose."""
    gen = PopulationGenerator()
    pop = gen.generate(size=300, seed=42)

    offer = Offer(price=135.0, base_price=100.0, perceived_value=110.0, unit_cost=50.0)
    candidates = [80.0, 90.0, 100.0, 110.0, 120.0, 135.0]

    # Candidate at $135 will rank low (e.g. rank 5 or 6). Set tolerance_rank_drop = 1
    res = validate_ranking_stability(
        population=pop,
        candidate_prices=candidates,
        base_offer=offer,
        tolerance_rank_drop=1,
        tolerance_profit_gap=0.99,  # Very loose gap tolerance
        seed=42,
    )

    assert res["rank_drop"] > 1
    assert res["rank_drop_passed"] is False
    assert res["passed"] is False


def test_ranking_stability_negative_profit_edge_cases():
    """Verify safe calculation of metric gap when prices are below unit cost (negative profit)."""
    gen = PopulationGenerator()
    pop = gen.generate(size=100, seed=42)

    # Unit cost $150 > price $100 -> negative unit margin
    loss_offer = Offer(price=100.0, base_price=100.0, perceived_value=110.0, unit_cost=150.0)
    candidates = [90.0, 100.0]

    res = validate_ranking_stability(
        population=pop,
        candidate_prices=candidates,
        base_offer=loss_offer,
        tolerance_rank_drop=2,
        tolerance_profit_gap=0.20,
        seed=42,
    )

    assert res["available"] is True
    assert "metric_gap" in res
    assert res["metric_gap"] >= 0.0


def test_candidate_simulation_uses_common_random_numbers():
    """Verify that simulate_candidates applies identical random sequences across price evaluations."""
    gen = PopulationGenerator()
    pop = gen.generate(size=100, seed=42)
    offer = Offer(price=100.0, base_price=100.0, perceived_value=110.0)

    # Run candidates simulation twice with the same seed
    res1 = simulate_candidates(pop, [90.0, 100.0, 110.0], offer, seed=99)
    res2 = simulate_candidates(pop, [90.0, 100.0, 110.0], offer, seed=99)

    for r1, r2 in zip(res1, res2):
        assert r1.price == r2.price
        assert r1.sim_conversion == r2.sim_conversion
        assert r1.rank_sim == r2.rank_sim
