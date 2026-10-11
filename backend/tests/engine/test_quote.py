"""prepare → optimizer → finalize, end to end."""

from __future__ import annotations

import json

import pytest

from app.engine import InvalidInputError, Prepared, finalize, prepare, to_customer_quote
from app.schemas import CandidateEvaluation, ModelInfo, OptimizerResponse
from tests.conftest import FakeOptimizer, load_bundle, make_strategy, req

V1 = "v1"


def quote(domain, product, strategy, attrs=None, optimizer=None, version=V1, **req_over):
    p = prepare(domain, product, strategy, req(attrs, **req_over), version)
    resp = (optimizer or FakeOptimizer()).evaluate(p.optimizer_request()) if p.candidates else None
    return p, finalize(p, resp, created_at="2026-01-01T00:00:00Z")


class TestEndToEnd:
    def test_rules_candidates_selection_and_explanation(self, domain, product, strategy) -> None:
        p, d = quote(domain, product, strategy, {"qty": 12, "express": True})
        # stock 5 → scarce +20% (demand stage) → 120; bulk −10% → 108; express +15 → 123
        assert p.anchor_price == 123
        assert [a.rule_id for a in d.applied_rules] == ["scarce", "bulk", "express"]
        assert [s.rule_id for s in d.skipped_rules] == ["gold"]
        assert len(d.candidates) == 5
        assert [c.price for c in d.candidates] == [100, 110.5, 117, 123, 129]
        assert all(c.eligible for c in d.candidates)
        best = max(d.candidates, key=lambda c: c.expected_profit)
        assert d.status == "OK" and d.final_price == best.price
        assert [c.selection_status for c in d.candidates].count("SELECTED") == 1
        assert d.constraints_evaluated and all(c.passed for c in d.constraints_evaluated)
        assert any(line.startswith("Selected") for line in d.explanation.business)
        assert d.provenance.optimizer.name == "fake-linear" and d.provenance.simulated
        assert d.decision_id.startswith("D-")

    def test_picks_highest_expected_profit_not_highest_probability(self, domain, product) -> None:
        s = make_strategy(rules=[], constraints=[])
        _, d = quote(domain, product, s)
        assert [c.price for c in d.candidates] == [90, 95, 100, 105, 110]
        sel = next(c for c in d.candidates if c.selection_status == "SELECTED")
        assert sel.price == 100  # profit 36 beats 90's 35, though 90 has the highest P(buy)
        assert max(d.candidates, key=lambda c: c.p_buy).price == 90
        for c in d.candidates:
            assert c.expected_profit == pytest.approx(c.p_buy * (c.price - 40))
            assert c.expected_revenue == pytest.approx(c.p_buy * c.price)

    def test_revenue_objective(self, domain, product) -> None:
        s = make_strategy(rules=[], constraints=[], optimizer_policy={"objective": "max_expected_revenue"})
        _, d = quote(domain, product, s)
        assert d.final_price == 90 and d.objective == "max_expected_revenue"

    def test_ties_go_to_the_lower_price(self, domain, product) -> None:
        s = make_strategy(rules=[], constraints=[])
        flat = FakeOptimizer(p_buy=lambda price, base: 20 / (price - 40))  # equal profit everywhere
        _, d = quote(domain, product, s, optimizer=flat)
        assert d.final_price == 90

    def test_overridden_rules_are_reported(self, domain, product) -> None:
        s = make_strategy(conflict_policy={"mode": "first_match_per_stage", "combine": "multiplicative"})
        _, d = quote(domain, product, s, {"qty": 12, "express": True})
        assert [a.rule_id for a in d.applied_rules] == ["scarce", "bulk"]
        assert [(o.rule_id, o.overridden_by) for o in d.overridden_rules] == [("express", "bulk")]
        assert any("overridden" in line for line in d.explanation.business)

    def test_formula_errors_are_skipped_with_reason(self, domain, product) -> None:
        bad = {"id": "bad", "name": "bad", "priority": 1, "stage": "adjust", "action": {"type": "formula", "expression": "price / 0"}}
        _, d = quote(domain, product, make_strategy(rules=[bad]))
        assert any(s.rule_id == "bad" and "formula error" in s.reason for s in d.skipped_rules)
        assert d.anchor_price == 100


class TestDeterminism:
    def test_identical_input_gives_identical_decision(self, domain, product, strategy) -> None:
        _, a = quote(domain, product, strategy, {"qty": 12, "express": True})
        _, b = quote(domain, product, strategy, {"express": True, "qty": 12})
        assert a.model_dump() == b.model_dump()

    def test_decision_id_changes_with_inputs_and_version(self, domain, product, strategy) -> None:
        ids = {
            quote(domain, product, strategy, {"qty": 12})[1].decision_id,
            quote(domain, product, strategy, {"qty": 13})[1].decision_id,
            quote(domain, product, strategy, {"qty": 12}, version="v2")[1].decision_id,
        }
        assert len(ids) == 3

    def test_recompute_from_stored_prepared_is_identical(self, domain, product, strategy) -> None:
        p, d = quote(domain, product, strategy, {"qty": 3})
        resp = FakeOptimizer().evaluate(p.optimizer_request())
        stored_p = json.loads(p.model_dump_json(by_alias=True))
        stored_r = json.loads(resp.model_dump_json())
        again = finalize(Prepared.model_validate(stored_p), OptimizerResponse.model_validate(stored_r), created_at=d.created_at)
        assert again.model_dump() == d.model_dump()

    def test_uses_the_given_strategy_not_live_config(self, domain, product) -> None:
        s = make_strategy(rules=[r for r in make_strategy().model_dump()["rules"] if r["id"] != "scarce"])
        assert prepare(domain, product, s, req(), V1).anchor_price == 100


class TestNoValidPrice:
    def test_conflicting_constraints(self, domain, product) -> None:
        s = make_strategy(constraints=[{"type": "floor", "value": 200}, {"type": "ceiling", "value": 150}])
        p, d = quote(domain, product, s)
        assert p.status == "NO_VALID_PRICE" and "conflict" in p.status_reason
        assert d.status == "NO_VALID_PRICE" and d.final_price is None
        assert d.candidates and all(c.selection_status == "INELIGIBLE" for c in d.candidates)
        assert d.explanation.customer == ["This option isn't available for the details provided."]
        assert finalize(p, None).status == "NO_VALID_PRICE"  # works without calling an optimizer

    def test_every_candidate_ineligible(self, domain, product) -> None:
        s = make_strategy(
            constraints=[{"type": "floor", "value": 100}, {"type": "ceiling", "value": 110}],
            candidate_policy={"offsets_pct": [-30, 30], "include_base_price": False, "include_bound_pins": False},
        )
        p, d = quote(domain, product, s, {"qty": 12, "express": True})
        assert not p.bounds.conflict
        assert d.status == "NO_VALID_PRICE" and d.final_price is None
        assert all(c.violations for c in d.candidates)

    def test_all_below_cost_is_refused(self, domain, product) -> None:
        cheap = {"id": "dump", "name": "dump", "priority": 1, "stage": "adjust", "action": {"type": "set", "value": 20}}
        s = make_strategy(rules=[cheap], constraints=[], candidate_policy={"include_base_price": False, "include_bound_pins": False})
        _, d = quote(domain, product, s)
        assert d.status == "NO_VALID_PRICE"
        assert all(any(v.constraint == "cost_floor" for v in c.violations) for c in d.candidates)
        # With bound pins on, the engine offers the lowest valid price instead of inventing one.
        pinned = make_strategy(rules=[cheap], constraints=[], candidate_policy={"include_base_price": False})
        _, d = quote(domain, product, pinned)
        assert d.status == "OK" and d.final_price == 40
        assert any("pinned to lower bound (cost_floor)" in r for c in d.candidates for r in c.reasons)

    def test_conversion_floor(self, domain, product) -> None:
        s = make_strategy(rules=[], constraints=[], optimizer_policy={"conversion_floor": 0.2})
        _, d = quote(domain, product, s, optimizer=FakeOptimizer(p_buy=lambda price, base: 0.05))
        assert d.status == "NO_VALID_PRICE" and "conversion floor" in d.status_reason
        assert {c.selection_status for c in d.candidates} == {"BELOW_CONVERSION_FLOOR"}


class TestReverification:
    def test_engine_overrides_an_ineligible_optimizer_pick(self, domain, product) -> None:
        s = make_strategy(rules=[], constraints=[{"type": "max_increase", "value": 5}])

        class Rogue:
            def evaluate(self, r):
                evals = [CandidateEvaluation(price=c.price, p_buy=0.9, expected_revenue=0.9 * c.price,
                                             expected_profit=0.9 * (c.price - 40), eligible=True, selection_status="NOT_SELECTED")
                         for c in r.candidates]
                return OptimizerResponse(evaluations=evals, selected_price=max(c.price for c in r.candidates),
                                         model=ModelInfo(name="rogue", version="0", kind="simulated"))

        _, d = quote(domain, product, s, optimizer=Rogue())
        assert d.final_price == 105  # 110 breaks max_increase 5%
        assert next(c for c in d.candidates if c.price == 110).selection_status == "INELIGIBLE"
        assert any(w.startswith("OPTIMIZER_SELECTION_OVERRIDDEN") for w in d.warnings)
        assert any(w.startswith("OPTIMIZER_ELIGIBILITY_MISMATCH") for w in d.warnings)

    def test_unknown_candidates_from_optimizer_are_ignored(self, domain, product) -> None:
        s = make_strategy(rules=[], constraints=[])
        p = prepare(domain, product, s, req(), V1)
        resp = FakeOptimizer().evaluate(p.optimizer_request())
        resp.evaluations.append(CandidateEvaluation(price=1.0, p_buy=1, expected_revenue=1, expected_profit=1, eligible=True,
                                                    selection_status="NOT_SELECTED"))
        resp.selected_price = 1.0
        d = finalize(p, resp)
        assert d.final_price == 100
        assert any(w.startswith("OPTIMIZER_UNKNOWN_CANDIDATE") for w in d.warnings)

    def test_optimizer_response_required_when_candidates_are_eligible(self, domain, product, strategy) -> None:
        with pytest.raises(ValueError):
            finalize(prepare(domain, product, strategy, req(), V1), None)


class TestInputs:
    def test_bad_input_raises(self, domain, product, strategy) -> None:
        with pytest.raises(InvalidInputError) as e:
            prepare(domain, product, strategy, req({"qty": "many"}), V1)
        assert "qty" in str(e.value)
        with pytest.raises(InvalidInputError):
            prepare(domain, product, strategy, req(domain_id="other"), V1)


class TestCustomerQuote:
    FORBIDDEN = {"cost", "unit_cost", "p_buy", "expected_profit", "expected_revenue", "rule_id", "applied_rules",
                 "violations", "shadow_validation", "margin", "context_snapshot", "signals"}

    def _keys(self, obj) -> set[str]:
        if isinstance(obj, dict):
            return set(obj) | set().union(*(self._keys(v) for v in obj.values())) if obj else set()
        if isinstance(obj, list):
            return set().union(*(self._keys(v) for v in obj)) if obj else set()
        return set()

    def test_customer_view_is_safe(self, domain, product, strategy) -> None:
        _, d = quote(domain, product, strategy, {"qty": 12, "express": True})
        q = to_customer_quote(d)
        assert not self._keys(q.model_dump()) & self.FORBIDDEN
        assert q.final_price == d.final_price
        eligible = {c.price for c in d.candidates if c.eligible}
        assert q.alternatives and all(a.price in eligible and a.price != d.final_price for a in q.alternatives)
        assert to_customer_quote(d, show_alternatives=False).alternatives is None


@pytest.mark.parametrize("name, attrs", [
    ("hospitality", {"check_in": "2026-10-24", "nights": 2, "channel": "Direct"}),
    ("gpu-compute", {"commitment": "reserved_1m", "hours": 200}),
    ("edtech", {"session_date": "2026-10-12", "session_hour": 18}),
    ("car-rental", {"pickup_date": "2026-10-20", "rental_days": 7, "driver_age": 23}),
    ("ecommerce", {"quantity": 3, "channel": "app"}),
    ("banking", {"credit_score": 800, "amount": 2500000, "tenure_months": 60}),
])
def test_same_engine_prices_every_domain(name: str, attrs: dict) -> None:
    b = load_bundle(name)
    for product in b.domain.products:
        r = req(attrs, domain_id=b.domain.id, product_id=product.id, as_of="2026-10-11")
        p = prepare(b.domain, product, b.strategy, r, f"{name}-v1")
        d = finalize(p, FakeOptimizer().evaluate(p.optimizer_request()))
        assert d.status == "OK", (product.id, d.status_reason)
        assert d.currency == b.domain.currency
        assert d.final_price is not None and all(c.passed for c in d.constraints_evaluated)
        assert d.explanation.customer
