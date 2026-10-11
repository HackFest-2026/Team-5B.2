"""Hard constraints (no clamping) and candidate generation."""

from __future__ import annotations

import pytest

from app.engine import PriceFacts, check_candidates, check_price, compute_bounds, generate_candidates, gross_margin_pct, markup_pct
from app.engine.constraints import evaluate_constraints
from app.schemas import Bounds, CandidatePolicy, Constraint

FACTS = PriceFacts(base=100, cost=60, previous_price=100)


def C(t: str, v: float) -> Constraint:
    return Constraint(type=t, value=v)  # type: ignore[arg-type]


class TestBounds:
    @pytest.mark.parametrize("cons, lower, upper", [
        ([C("floor", 50)], 60, None),             # implicit cost floor (60) beats a floor of 50
        ([C("floor", 70)], 70, None),
        ([C("ceiling", 120)], 60, 120),
        ([C("margin_floor", 25)], 80, None),      # 60 / 0.75
        ([C("markup_floor", 25)], 75, None),      # 60 * 1.25
        ([C("max_discount", 20)], 80, None),
        ([C("max_increase", 15)], 60, 115),
        ([C("max_daily_change", 10)], 90, 110),
    ])
    def test_each_constraint(self, cons, lower, upper) -> None:
        b = compute_bounds(cons, FACTS)
        assert b.lower == lower and b.upper == upper and not b.conflict

    def test_margin_vs_markup_are_different(self) -> None:
        f = PriceFacts(base=200, cost=100)
        assert compute_bounds([C("margin_floor", 20)], f).lower == 125
        assert compute_bounds([C("markup_floor", 20)], f).lower == 120
        assert gross_margin_pct(125, 100) == pytest.approx(20)
        assert markup_pct(120, 100) == pytest.approx(20)

    def test_conflicts_are_detected(self) -> None:
        b = compute_bounds([C("floor", 120), C("ceiling", 110)], FACTS)
        assert b.conflict and b.lower_by == ["floor"] and b.upper_by == ["ceiling"]
        assert compute_bounds([C("max_increase", 0), C("margin_floor", 50)], FACTS).conflict

    def test_allow_below_cost_removes_cost_floor(self) -> None:
        assert compute_bounds([], PriceFacts(100, 60, allow_below_cost=True)).lower == 0


class TestChecks:
    def test_violations_are_reported_not_clamped(self) -> None:
        v = check_price(130, [C("ceiling", 120), C("rounding", 5)], FACTS)
        assert [x.constraint for x in v] == ["ceiling"]
        v = check_price(81.5, [C("rounding", 5)], FACTS)
        assert [x.constraint for x in v] == ["rounding"]
        assert {x.constraint for x in check_price(-1, [], FACTS)} == {"positive_price", "cost_floor"}
        assert check_price(0, [], PriceFacts(1, 0, allow_below_cost=True))[0].constraint == "positive_price"

    def test_constraint_report(self) -> None:
        rows = evaluate_constraints(90, [C("margin_floor", 25), C("rounding", 5)], FACTS)
        assert all(r.passed for r in rows)
        assert {r.constraint for r in rows} == {"positive_price", "cost_floor", "margin_floor", "rounding"}


class TestCandidates:
    def gen(self, anchor: float, bounds: Bounds = Bounds(lower=60), step=None, decimals=2, **policy):
        return generate_candidates(anchor, 100, bounds, CandidatePolicy(**policy), step, decimals)

    def test_default_five_with_reasons(self) -> None:
        c = self.gen(100)
        assert [x.price for x in c] == [90, 95, 100, 105, 110]
        mid = next(x for x in c if x.price == 100)
        assert mid.reasons == ["rule-adjusted anchor", "reference (base) price"]  # deduped and merged

    def test_dedupe_after_rounding_merges_reasons(self) -> None:
        c = self.gen(101, step=10, offsets_pct=[-1, 0, 1, 20])
        prices = [x.price for x in c]
        assert len(prices) == len(set(prices))
        merged = next(x for x in c if x.price == 100)
        assert "rule-adjusted anchor" in merged.reasons and "anchor -1%" in merged.reasons and "reference (base) price" in merged.reasons

    def test_count_trims_farthest_offsets_first(self) -> None:
        c = self.gen(120, count=3)
        # anchor, base, then the nearest offset survive.
        assert sorted(x.price for x in c) == [100, 114, 120]

    def test_bound_pins_only_when_anchor_out_of_bounds(self) -> None:
        inside = self.gen(100, Bounds(lower=60, upper=130, lower_by=["cost_floor"], upper_by=["ceiling"]))
        assert not any("pinned" in r for x in inside for r in x.reasons)
        above = self.gen(150, Bounds(lower=60, upper=118.3, upper_by=["ceiling"]), step=5)
        pin = next(x for x in above if any("pinned" in r for r in x.reasons))
        assert pin.price == 115  # rounded down so it stays inside the bound
        below = self.gen(50, Bounds(lower=81.2, lower_by=["margin_floor"]), step=5)
        assert next(x for x in below if any("pinned" in r for r in x.reasons)).price == 85
        conflict = self.gen(150, Bounds(lower=120, upper=110, conflict=True))
        assert not any("pinned" in r for x in conflict for r in x.reasons)

    def test_currency_decimals(self) -> None:
        c = self.gen(33.333, Bounds(), decimals=0, offsets_pct=[0])
        assert [x.price for x in c] == [33, 100]
        pinned = self.gen(33.333, Bounds(lower=60, lower_by=["cost_floor"]), decimals=0, offsets_pct=[0])
        assert [x.price for x in pinned] == [33, 60, 100]

    def test_eligibility_keeps_rejected_candidates_visible(self) -> None:
        cons = [C("max_increase", 5)]
        checked = check_candidates(self.gen(100), cons, FACTS)
        assert [x.price for x in checked] == [90, 95, 100, 105, 110]
        bad = [x for x in checked if not x.eligible]
        assert [x.price for x in bad] == [110]
        assert bad[0].violations[0].constraint == "max_increase"
