"""Rule ordering, selection, conflict resolution and action arithmetic."""

from __future__ import annotations

from app.engine import apply_rules, cap_total_adjustment, order_rules, resolve_conflicts, select_rules
from app.engine.pipeline import CAP_RULE_ID
from tests.conftest import rule

PIPELINE = ["base", "demand", "adjust"]
ALWAYS: dict = {"all": []}


class TestSelection:
    def test_order_is_stage_then_priority_then_id(self) -> None:
        rules = [
            rule("b", ALWAYS, {"type": "fixed", "value": 1}, stage="adjust", priority=5, created_at="2026-01-01"),
            rule("a", ALWAYS, {"type": "fixed", "value": 1}, stage="adjust", priority=5, created_at="2026-01-02"),
            rule("c", ALWAYS, {"type": "fixed", "value": 1}, stage="demand", priority=99),
        ]
        assert [r.id for r in select_rules(rules, {}, "2026-01-01", PIPELINE).fired] == ["c", "a", "b"]
        # Input order never matters.
        assert [r.id for r in order_rules(list(reversed(rules)), PIPELINE)] == ["c", "a", "b"]

    def test_skip_reasons(self) -> None:
        rules = [
            rule("off", ALWAYS, {"type": "fixed", "value": 1}, enabled=False),
            rule("future", ALWAYS, {"type": "fixed", "value": 1}, effective_from="2027-01-01"),
            rule("past", ALWAYS, {"type": "fixed", "value": 1}, effective_to="2025-12-31"),
            rule("nomatch", {"attr": "qty", "op": ">", "value": 5}, {"type": "fixed", "value": 1}),
            rule("badstage", ALWAYS, {"type": "fixed", "value": 1}, stage="nowhere"),
        ]
        sel = select_rules(rules, {"qty": 1}, "2026-06-01", PIPELINE)
        assert sel.fired == []
        reasons = {s.rule_id: s.reason for s in sel.skipped}
        assert "disabled" in reasons["off"]
        assert "not effective" in reasons["future"]
        assert "expired" in reasons["past"]
        assert "fails" in reasons["nomatch"]
        assert "not in the pipeline" in reasons["badstage"]


class TestConflicts:
    rules = [
        rule("d1", ALWAYS, {"type": "percent", "value": 10}, stage="demand", priority=20),
        rule("d0", ALWAYS, {"type": "percent", "value": 5}, stage="demand", priority=10),
        rule("a1", ALWAYS, {"type": "percent", "value": -3}, stage="adjust", priority=1),
    ]

    def fired(self):
        return select_rules(self.rules, {}, "2026-01-01", PIPELINE).fired

    def test_stack_applies_all(self) -> None:
        applied, overridden = resolve_conflicts(self.fired(), "stack", PIPELINE)
        assert [r.id for r in applied] == ["d0", "d1", "a1"] and overridden == []

    def test_first_match_per_stage(self) -> None:
        applied, overridden = resolve_conflicts(self.fired(), "first_match_per_stage", PIPELINE)
        assert [r.id for r in applied] == ["d0", "a1"]
        assert [(o.rule_id, o.overridden_by) for o in overridden] == [("d1", "d0")]
        assert "first matching rule" in overridden[0].reason

    def test_highest_priority_wins(self) -> None:
        applied, overridden = resolve_conflicts(self.fired(), "highest_priority_wins", PIPELINE)
        assert [r.id for r in applied] == ["a1"]  # priority 1 is the highest priority overall
        assert sorted(o.rule_id for o in overridden) == ["d0", "d1"]
        assert all(o.overridden_by == "a1" for o in overridden)

    def test_ties_resolved_by_rule_id(self) -> None:
        tied = [rule("zeta", ALWAYS, {"type": "percent", "value": 1}, priority=5),
                rule("alpha", ALWAYS, {"type": "percent", "value": 2}, priority=5)]
        fired = select_rules(tied, {}, "2026-01-01", PIPELINE).fired
        applied, _ = resolve_conflicts(fired, "highest_priority_wins", PIPELINE)
        assert applied[0].id == "alpha"


class TestActions:
    plus10 = rule("p", ALWAYS, {"type": "percent", "value": 10})
    minus10 = rule("m", ALWAYS, {"type": "percent", "value": -10})

    def test_multiplicative_compounds(self) -> None:
        assert apply_rules(100, [self.plus10, self.minus10], "multiplicative", {}).anchor == 99

    def test_additive_sums_against_base(self) -> None:
        assert apply_rules(100, [self.plus10, self.minus10], "additive", {}).anchor == 100

    def test_decimal_safe(self) -> None:
        assert apply_rules(0.1, [rule("f", ALWAYS, {"type": "fixed", "value": 0.2})], "additive", {}).anchor == 0.3

    def test_set_multiply_formula_none_logged(self) -> None:
        rules = [
            rule("s", ALWAYS, {"type": "set", "value": 50}),
            rule("x", ALWAYS, {"type": "multiply", "value": 1.5}),
            rule("f", ALWAYS, {"type": "formula", "expression": "price + qty * 2"}),
            rule("n", ALWAYS, {"type": "none"}),
        ]
        r = apply_rules(100, rules, "multiplicative", {"qty": 3})
        assert r.anchor == 81
        assert [(b.before, b.after, b.delta) for b in r.breakdown] == [(100, 50, -50), (50, 75, 25), (75, 81, 6), (81, 81, 0)]
        assert r.breakdown[3].reason == "no adjustment"

    def test_formula_errors_leave_price_unchanged(self) -> None:
        r = apply_rules(100, [rule("bad", ALWAYS, {"type": "formula", "expression": "price / 0"})], "multiplicative", {})
        assert r.anchor == 100 and r.errors[0].rule_id == "bad" and r.breakdown == []

    def test_total_adjustment_cap_is_logged(self) -> None:
        big = rule("big", ALWAYS, {"type": "percent", "value": 40})
        r = cap_total_adjustment(100, apply_rules(100, [big], "multiplicative", {}), 25)
        assert r.anchor == 125
        assert r.breakdown[-1].rule_id == CAP_RULE_ID and r.breakdown[-1].before == 140
        small = cap_total_adjustment(100, apply_rules(100, [self.plus10], "multiplicative", {}), 25)
        assert small.anchor == 110 and small.breakdown[-1].rule_id == "p"
