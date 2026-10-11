"""Expression language, conditions, context enrichment and semantic signals."""

from __future__ import annotations

import pytest

from app.engine import (
    ExpressionError,
    InvalidInputError,
    check_expression,
    coerce,
    enrich_context,
    evaluate_condition,
    evaluate_expression,
    expression_variables,
    map_signals,
)
from app.engine.hashing import canonical_json, content_hash
from app.schemas import Attribute, CustomerProfile, Rule
from app.schemas.rules import LeafCondition


def cond(raw: dict):
    return Rule.model_validate({"id": "r", "name": "r", "priority": 1, "stage": "s", "condition": raw, "action": {"type": "none"}}).condition


class TestExpression:
    def test_arithmetic_precedence_and_ternary(self) -> None:
        assert evaluate_expression("1 + 2 * 3", {}) == 7
        assert evaluate_expression("(1 + 2) * 3", {}) == 9
        assert evaluate_expression("price > 100 ? price * 0.9 : price", {"price": 200}) == 180
        assert evaluate_expression("-x + max(1, 5, 3)", {"x": 2}) == 3
        assert evaluate_expression("round(2.345, 2)", {}) == 2.35

    def test_date_helpers(self) -> None:
        assert evaluate_expression("daysBetween('2026-01-01', '2026-01-31')", {}) == 30
        assert evaluate_expression("isWeekend('2026-01-03')", {}) is True  # Saturday
        assert evaluate_expression("dayOfWeek('2026-01-05')", {}) == 1  # Monday (0 = Sunday)
        assert evaluate_expression("month('2026-03-09')", {}) == 3

    def test_no_global_or_attribute_access(self) -> None:
        with pytest.raises(ExpressionError, match="Unknown variable"):
            evaluate_expression("constructor", {})
        with pytest.raises(ExpressionError, match="Unknown function"):
            evaluate_expression("alert(1)", {})
        for evil in ("__import__('os')", "x.__class__", "().__class__", "open('f')", "exec('1')", "a[0]", "lambda: 1"):
            with pytest.raises(ExpressionError):
                evaluate_expression(evil, {"x": 1, "a": 1})
        assert check_expression("__import__('os').system('dir')") is not None
        assert check_expression("eval('1')") is not None

    def test_dotted_names_are_plain_keys(self) -> None:
        # Dots are part of a variable name; there is no attribute lookup.
        assert evaluate_expression("a.b + 1", {"a.b": 1}) == 2
        with pytest.raises(ExpressionError, match="Unknown variable"):
            evaluate_expression("a.b", {"a": {"b": 1}})

    def test_errors_and_introspection(self) -> None:
        assert check_expression("1 +") is not None
        assert check_expression("a * (b + 2)") is None
        assert sorted(expression_variables("a * max(b, 2) + c")) == ["a", "b", "c"]
        with pytest.raises(ExpressionError, match="Division by zero"):
            evaluate_expression("1 / 0", {})
        with pytest.raises(ExpressionError, match="longer than"):
            evaluate_expression("1+" * 300 + "1", {})
        with pytest.raises(ExpressionError, match="nested"):
            evaluate_expression("(" * 60 + "1" + ")" * 60, {})


class TestConditions:
    ctx = {"qty": 12, "colour": "blue", "express": True, "start": "2026-02-01"}

    @pytest.mark.parametrize("raw, expected", [
        ({"attr": "qty", "op": ">", "value": 10}, True),
        ({"attr": "qty", "op": "<=", "value": 11}, False),
        ({"attr": "colour", "op": "in", "value": ["red", "blue"]}, True),
        ({"attr": "colour", "op": "not_in", "value": ["red", "blue"]}, False),
        ({"attr": "colour", "op": "not_in", "value": ["red"]}, True),
        ({"attr": "colour", "op": "!=", "value": "blue"}, False),
        ({"attr": "qty", "op": "between", "value": [10, 12]}, True),
        ({"attr": "express", "op": "==", "value": "true"}, True),
        ({"attr": "start", "op": ">=", "value": "2026-01-15"}, True),
        ({"attr": "qty", "op": "==", "value": "12"}, True),
    ])
    def test_operators(self, raw: dict, expected: bool) -> None:
        assert evaluate_condition(LeafCondition.model_validate(raw), self.ctx).ok is expected

    def test_nested_logic_and_reasons(self) -> None:
        c = cond({"all": [{"attr": "qty", "op": ">", "value": 5},
                          {"any": [{"attr": "colour", "op": "==", "value": "red"}, {"not": {"attr": "express", "op": "==", "value": False}}]}]})
        assert evaluate_condition(c, self.ctx).ok
        missing = evaluate_condition(LeafCondition(attr="nope", op="==", value=1), self.ctx)
        assert not missing.ok and "no value" in missing.reason
        assert evaluate_condition(cond({"all": []}), {}).reason == "always applies"


class TestContext:
    def test_merges_sources_and_resolves_computed_in_any_order(self, domain, product) -> None:
        reversed_domain = domain.model_copy(update={"attributes": list(reversed(domain.attributes))})
        res = enrich_context(reversed_domain, product, {"qty": "7", "express": "yes", "start": "2026-01-29"}, "2026-01-01",
                             CustomerProfile(id="C1", attributes={"tier": "Gold"}))
        ctx = res.context
        assert ctx["qty"] == 7 and ctx["express"] is True and ctx["stock"] == 5 and ctx["tier"] == "Gold"
        assert ctx["lead"] == 28 and ctx["lead_weeks"] == 4
        assert ctx["colour"] == "red"  # default
        assert ctx["base_price"] == 100 and ctx["cost"] == 40
        assert res.errors == []

    def test_invalid_values_are_errors_not_silent_defaults(self, domain, product) -> None:
        res = enrich_context(domain, product, {"qty": "lots", "colour": "purple", "express": "maybe", "start": "tomorrow"}, "2026-01-01")
        joined = " | ".join(res.errors)
        assert "qty" in joined and "colour" in joined and "express" in joined and "start" in joined

    def test_range_and_required(self, domain, product) -> None:
        res = enrich_context(domain, product, {"qty": 500}, "2026-01-01")
        assert any("outside the allowed range" in e for e in res.errors)
        req_attr = Attribute(key="must", label="Must", type="number", source="request", required=True)
        d2 = domain.model_copy(update={"attributes": [*domain.attributes, req_attr]})
        assert any("required" in e for e in enrich_context(d2, product, {}, "2026-01-01").errors)

    def test_request_overrides_inventory_and_unknown_keys_warn(self, domain, product) -> None:
        res = enrich_context(domain, product, {"stock": 80, "bogus": 1}, "2026-01-01")
        assert res.context["stock"] == 80
        assert any("bogus" in w for w in res.warnings)

    def test_coerce(self) -> None:
        num = Attribute(key="x", label="x", type="number", source="request", default=9)
        assert coerce("3.5", num) == (3.5, None)
        assert coerce("", num) == (9, None)
        assert coerce(True, num)[1] is not None
        boolean = Attribute(key="b", label="b", type="boolean", source="request")
        assert coerce("false", boolean) == (False, None)


class TestSignals:
    def test_mapping_normalisation_and_defaults(self, domain, product) -> None:
        d = domain.model_copy(update={"defaults": {"customerSensitivity": 0.7, "scarcity": 0.1}})
        ctx = enrich_context(d, product, {"rival": 120}, "2026-01-01").context
        s = map_signals(d, ctx)
        assert s.values["basePrice"] == 100 and s.values["unitCost"] == 40
        assert s.raw["scarcity"] == 5 and s.values["scarcity"] == pytest.approx(0.95)  # inverted 0..100
        assert s.sources["scarcity"] == "attribute:stock"  # attribute beats domain default
        assert s.values["competitorPrice"] == 120
        assert s.values["customerSensitivity"] == 0.7 and s.sources["customerSensitivity"] == "domain_default"


def test_canonical_hash_ignores_key_order() -> None:
    assert canonical_json({"b": 1, "a": {"d": 2, "c": 3}}) == canonical_json({"a": {"c": 3, "d": 2}, "b": 1})
    assert content_hash({"x": 1}) != content_hash({"x": 2})


def test_invalid_input_error_lists_all_errors() -> None:
    e = InvalidInputError(["a", "b"])
    assert e.errors == ["a", "b"] and str(e) == "a; b"
