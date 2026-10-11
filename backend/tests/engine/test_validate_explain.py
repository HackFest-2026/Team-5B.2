"""Strategy validation, plain-English read-back and version diffs."""

from __future__ import annotations

from app.engine import condition_to_english, constraint_to_english, diff_strategies, rule_to_english, validate_strategy, validate_strategy_payload
from app.schemas import Constraint
from tests.conftest import BASE_RULES, make_strategy, rule


def v(domain, **over):
    return validate_strategy(domain, make_strategy(**over))


def raw_rule(id, condition=None, action=None, **over):
    return {"id": id, "name": id, "priority": over.pop("priority", 10), "stage": over.pop("stage", "adjust"),
            "condition": condition or {"all": []}, "action": action or {"type": "fixed", "value": 1}, **over}


class TestValidation:
    def test_clean_strategy_passes(self, domain) -> None:
        r = v(domain, constraints=[{"type": "rounding", "value": 0.5}, {"type": "max_increase", "value": 30}, {"type": "max_discount", "value": 30}])
        assert r.ok and r.counts["warning"] == 0, [i.message for i in r.issues]

    def test_unknown_attributes_bad_stage_and_big_discount(self, domain) -> None:
        r = v(domain, rules=[
            raw_rule("u", {"attr": "ghost", "op": "==", "value": 1}, {"type": "percent", "value": 5}),
            raw_rule("s", stage="nope", action={"type": "percent", "value": 5}),
            raw_rule("d", action={"type": "percent", "value": -120}),
            raw_rule("f", action={"type": "formula", "expression": "price * ghostly"}),
        ])
        codes = [i.code for i in r.issues]
        assert "schema" in codes and "discount_over_100" in codes
        assert codes.count("unknown_attribute") == 2
        assert not r.ok

    def test_formula_not_allowed_in_generated_strategies(self, domain) -> None:
        f = raw_rule("f", action={"type": "formula", "expression": "price * 0.9"})
        assert v(domain, rules=[f]).ok
        assert any(i.code == "formula_not_allowed" for i in v(domain, rules=[f], source="llm").issues)
        assert not validate_strategy(domain, make_strategy(rules=[f]), allow_formula=False).ok

    def test_floor_above_ceiling_and_bounds_conflict(self, domain) -> None:
        r = v(domain, constraints=[{"type": "floor", "value": 200}, {"type": "ceiling", "value": 150}])
        codes = {i.code for i in r.issues}
        assert {"floor_above_ceiling", "bounds_conflict"} <= codes
        r = v(domain, constraints=[{"type": "max_increase", "value": 0}, {"type": "margin_floor", "value": 70}])
        assert any(i.code == "bounds_conflict" for i in r.issues)

    def test_multiple_rounding_steps_is_an_error(self, domain) -> None:
        r = v(domain, constraints=[{"type": "rounding", "value": 1}, {"type": "rounding", "value": 5}])
        assert not r.ok

    def test_unreachable_conditions(self, domain) -> None:
        r = v(domain, rules=[
            raw_rule("x", {"all": [{"attr": "qty", "op": ">", "value": 10}, {"attr": "qty", "op": "<", "value": 5}]}),
            raw_rule("y", {"all": [{"attr": "colour", "op": "==", "value": "red"}, {"attr": "colour", "op": "==", "value": "blue"}]}),
            raw_rule("z", {"attr": "qty", "op": "between", "value": [9, 3]}),
            raw_rule("w", {"all": [{"attr": "colour", "op": "in", "value": ["red"]}, {"attr": "colour", "op": "not_in", "value": ["red"]}]}),
        ])
        assert [i.rule_ids[0] for i in r.issues if i.code == "unreachable"] == ["x", "y", "z", "w"]

    def test_type_checks(self, domain) -> None:
        r = v(domain, rules=[
            raw_rule("n", {"attr": "qty", "op": ">", "value": "lots"}),
            raw_rule("b", {"attr": "express", "op": ">", "value": True}),
            raw_rule("e", {"attr": "colour", "op": "==", "value": "purple"}),
        ])
        by_rule = {i.rule_ids[0]: i for i in r.issues if i.rule_ids}
        assert by_rule["n"].code == "invalid_value" and by_rule["n"].severity == "error"
        assert by_rule["b"].code == "unsupported_operator"
        assert by_rule["e"].severity == "warning"

    def test_priority_ties_only_when_order_changes_the_price(self, domain) -> None:
        pct = raw_rule("pct", action={"type": "percent", "value": -10}, priority=1)
        fix = raw_rule("fix", action={"type": "fixed", "value": -10}, priority=1)
        pct2 = raw_rule("pct2", action={"type": "percent", "value": 5}, priority=1)
        tie = next(i for i in v(domain, rules=[pct, fix]).issues if i.code == "priority_tie")
        assert {tie.details["order_a"]["price"], tie.details["order_b"]["price"]} == {80, 81}
        assert "rule id" in tie.message
        assert not any(i.code == "priority_tie" for i in v(domain, rules=[pct, pct2]).issues)  # percents commute
        # In first-match mode, which rule wins matters even for commuting percents.
        r = v(domain, rules=[pct, pct2], conflict_policy={"mode": "first_match_per_stage"})
        assert any(i.code == "priority_tie" for i in r.issues)

    def test_ood_risk_warning(self, domain) -> None:
        r = v(domain, constraints=[])
        assert any(i.code == "ood_risk" for i in r.issues) and r.ok

    def test_payload_validation_returns_paths(self, domain) -> None:
        raw = make_strategy().model_dump(mode="json", by_alias=True)
        raw["rules"][0]["condition"] = {"attr": "qty", "op": "regex", "value": 1}
        s, r = validate_strategy_payload(domain, raw)
        assert s is None and not r.ok
        assert r.issues[0].code == "unsupported_operator" and r.issues[0].path == "rules.0.condition.op"
        raw = make_strategy().model_dump(mode="json", by_alias=True)
        raw["final_price"] = 1
        assert not validate_strategy_payload(domain, raw)[1].ok


class TestReadBack:
    def test_rules_and_conditions(self, domain) -> None:
        bulk = rule(**BASE_RULES[0])
        assert rule_to_english(bulk, domain.attributes, domain.output_unit) == "If Quantity is at least 10, decrease the price by 10%."
        assert condition_to_english(rule("a").condition, []) == "always"
        c = rule("x", {"any": [{"attr": "qty", "op": "<", "value": 2}, {"attr": "colour", "op": "not_in", "value": ["red", "blue"]}]}).condition
        assert condition_to_english(c, domain.attributes) == "Quantity is less than 2 or Colour is not one of red, blue"

    def test_constraints_state_margin_vs_markup(self, domain) -> None:
        assert "(price - cost) / price" in constraint_to_english(Constraint(type="margin_floor", value=20), domain.output_unit)
        assert "(price - cost) / cost" in constraint_to_english(Constraint(type="markup_floor", value=20), domain.output_unit)


class TestDiff:
    def test_added_removed_changed_and_settings(self) -> None:
        a = make_strategy()
        rules = a.model_dump(mode="json", by_alias=True)["rules"]
        rules[0]["action"]["value"] = -15
        rules = [r for r in rules if r["id"] != "gold"] + [raw_rule("new")]
        b = make_strategy(rules=rules, conflict_policy={"mode": "first_match_per_stage"})
        d = diff_strategies(a, b)
        assert [r.id for r in d.added] == ["new"] and [r.id for r in d.removed] == ["gold"]
        assert d.changed[0].before.id == "bulk" and d.changed[0].fields == ["action"]
        assert [s.field for s in d.settings] == ["conflict_policy"]
        assert d.has_changes and not diff_strategies(a, a).has_changes
