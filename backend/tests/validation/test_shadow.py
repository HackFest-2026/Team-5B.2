"""Customer-agent validation unit tests: adapter mapping, cohorts, utility mirror, Check F, determinism."""

from __future__ import annotations

import pytest

from app.engine import finalize, prepare
from app.optimizer import SimulatedPriceOptimizer
from app.schemas import CandidateEvaluation, ModelInfo, OptimizerResponse
from app.schemas.strategy import ValidationPolicy
from app.validation import ValidationService, default_personas
from app.validation.adapter import ml_ladder_check, to_offer
from app.validation.cohorts import evaluate_cohort, utility_terms
from app.validation.populations import base_population
from app.validation.sim import cas
from tests.conftest import load_bundle, make_strategy, req


def decision_for(name="hospitality", product_idx=1, attrs=None, optimizer=None, **req_over):
    b = load_bundle(name)
    product = b.domain.products[product_idx]
    r = req(attrs if attrs is not None else {"check_in": "2026-10-24"}, domain_id=b.domain.id, product_id=product.id,
            as_of="2026-10-11", **req_over)
    p = prepare(b.domain, product, b.strategy, r, "v1")
    return b, finalize(p, (optimizer or SimulatedPriceOptimizer()).evaluate(p.optimizer_request()))


def test_offer_mapping_keeps_missing_competitor_none(domain, product) -> None:
    s = make_strategy(rules=[], constraints=[])
    p = prepare(domain, product, s, req(), "v1")
    d = finalize(p, SimulatedPriceOptimizer().evaluate(p.optimizer_request()))
    offer, warnings = to_offer(d)
    assert offer.competitor_price is None and offer.perceived_value == 100 and warnings
    assert 0 <= offer.urgency <= 1 and offer.scarcity == pytest.approx(0.95)


def test_utility_mirror_matches_module() -> None:
    policy = cas.StandardDecisionPolicy()
    offer = cas.Offer(price=115, base_price=100, perceived_value=110, competitor_price=105, urgency=0.4, scarcity=0.3, unit_cost=50)
    for agent in list(base_population(50, 3))[:50]:
        terms = utility_terms(policy, agent, offer)
        u_buy = policy.calculate_probabilities(agent, offer)[3]
        assert sum(terms.values()) == pytest.approx(u_buy, abs=1e-9)


def test_cohorts_move_in_the_expected_direction() -> None:
    personas = {p.id: p for p in default_personas()}
    offer = cas.Offer(price=140, base_price=100, perceived_value=100, competitor_price=110, urgency=0.2, scarcity=0.2, unit_cost=50)
    budget = evaluate_cohort(personas["budget_conscious"], offer, [140], 200, 42)[0]
    premium = evaluate_cohort(personas["premium_seeker"], offer, [140], 200, 42)[0]
    assert budget.buy_share < premium.buy_share
    assert budget.concern_driver in ("budget", "price", "competitor", "wait")
    assert abs(budget.buy_share + budget.defer_share + budget.reject_share - 1) < 1e-3


def test_validation_is_deterministic_and_skips_no_valid_price() -> None:
    b, d = decision_for()
    svc = ValidationService()
    a = svc.run(d, b.strategy.validation_policy, default_personas(), 200)
    c = svc.run(d, b.strategy.validation_policy, default_personas(), 200)
    assert a.model_dump() == c.model_dump()
    skipped = svc.run(d.model_copy(update={"status": "NO_VALID_PRICE", "final_price": None}), ValidationPolicy(), [], 200)
    assert skipped.status == "SKIPPED"


def test_personas_never_change_the_price() -> None:
    b, d = decision_for()
    before = d.model_dump()
    v = ValidationService().run(d, b.strategy.validation_policy, default_personas(), 200)
    assert d.model_dump() == before  # the decision object is untouched
    everyone_rejects = all(r.reaction == "REJECTS" for row in (v.personas or []) for r in row)
    assert not everyone_rejects or d.final_price == before["final_price"]


def test_check_f_flags_rising_probabilities(domain, product) -> None:
    s = make_strategy(rules=[], constraints=[])
    p = prepare(domain, product, s, req(), "v1")

    class Inverted:
        def evaluate(self, r):
            evals = [CandidateEvaluation(price=c.price, p_buy=min(0.9, c.price / 150), expected_revenue=0, expected_profit=0,
                                         eligible=c.eligible, selection_status="NOT_SELECTED") for c in r.candidates]
            return OptimizerResponse(evaluations=evals, selected_price=None, model=ModelInfo(name="inv", version="0", kind="simulated"))

    d = finalize(p, Inverted().evaluate(p.optimizer_request()))
    ok, inversions = ml_ladder_check(d)
    assert not ok and inversions
    v = ValidationService().run(d, ValidationPolicy(), [], 200, include_personas=False)
    assert v.status in ("REVIEW_REQUIRED", "FAIL")
    assert next(c for c in v.checks if c.key == "F_ml_monotonicity").severity == "review"


def test_alternative_suggestion_is_marked_with_eligibility() -> None:
    b, d = decision_for()
    v = ValidationService().run(d, b.strategy.validation_policy, [], 200, include_personas=False)
    assert v.feedback["advisory_only"] is True
    if v.feedback.get("alternative_candidate_price") is not None:
        assert "alternative_candidate_eligible" in v.feedback
