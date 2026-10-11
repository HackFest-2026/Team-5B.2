"""Phase 1: contracts parse valid configs and reject invalid ones with path-based errors."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.engine import validate_bundle_payload
from app.schemas import (
    Constraint,
    CustomerQuote,
    DomainConfig,
    LeafCondition,
    ML_FEATURE_COLUMNS,
    MLFeatures,
    OptimizerResponse,
    PersonaReaction,
    PriceRequest,
    PricingDecision,
    Rule,
    ShadowValidation,
    StrategyConfig,
)
from app.schemas.llm import LlmStrategyOut
from tests.conftest import EXAMPLES, WIDGET_DOMAIN, load_json

VALID = ["hospitality", "gpu-compute", "edtech", "ecommerce", "banking", "car-rental"]

# file -> (expected issue code, substring expected in the issue path or message)
INVALID = {
    "unsupported_operator": ("unsupported_operator", "rules.0.condition.op"),
    "code_action": ("schema", "rules.0.action.type"),
    "missing_action_value": ("schema", "rules.1.action"),
    "wrong_priority_type": ("schema", "rules.0.priority"),
    "caller_final_price": ("schema", "strategy.final_price"),
    "enum_without_values": ("schema", "allowed_values"),
    "duplicate_semantic_signal": ("schema", "capacityUtilization"),
    "margin_floor_100": ("schema", "strategy.constraints.0"),
    "between_needs_two": ("schema", "rules.0.condition"),
    "unknown_attribute": ("unknown_attribute", "moon_phase"),
    "stage_not_in_pipeline": ("schema", "strategy.rules.0.stage"),
    "floor_above_ceiling": ("floor_above_ceiling", "floor"),
    "bounds_conflict": ("bounds_conflict", "No price can satisfy"),
    "discount_over_100": ("discount_over_100", "120%"),
    "mismatched_domain_id": ("schema", "does not match"),
}


@pytest.mark.parametrize("name", VALID)
def test_example_bundles_are_valid(name: str) -> None:
    bundle, report = validate_bundle_payload(load_json(EXAMPLES / f"{name}.json"))
    assert bundle is not None
    assert report.ok, [i.message for i in report.issues]
    assert report.counts["error"] == 0
    assert report.counts["warning"] == 0, [i.message for i in report.issues]


def test_every_invalid_file_is_covered() -> None:
    files = {p.stem for p in (EXAMPLES / "invalid").glob("*.json")}
    assert files == set(INVALID)


@pytest.mark.parametrize("name", sorted(INVALID))
def test_invalid_bundles_are_rejected_with_actionable_errors(name: str) -> None:
    code, needle = INVALID[name]
    _, report = validate_bundle_payload(load_json(EXAMPLES / "invalid" / f"{name}.json"))
    assert not report.ok
    matching = [i for i in report.errors() if i.code == code]
    assert matching, f"expected a {code} error, got {[(i.code, i.message) for i in report.issues]}"
    assert any(needle in (i.path or "") or needle in i.message for i in matching)


def test_condition_tree_round_trips_with_aliases() -> None:
    raw = {"all": [{"attr": "qty", "op": ">", "value": 1}, {"not": {"any": [{"attr": "colour", "op": "not_in", "value": ["red"]}]}}]}
    r = Rule.model_validate({"id": "r", "name": "r", "priority": 1, "stage": "adjust", "condition": raw, "action": {"type": "none"}})
    assert r.model_dump(mode="json")["condition"] == raw


def test_condition_shape_errors() -> None:
    with pytest.raises(ValidationError, match="between"):
        LeafCondition(attr="x", op="between", value=[1])
    with pytest.raises(ValidationError, match="non-empty list"):
        LeafCondition(attr="x", op="in", value=[])
    with pytest.raises(ValidationError, match="single value"):
        LeafCondition(attr="x", op=">", value=[1, 2])
    with pytest.raises(ValidationError, match='one of the keys'):
        Rule.model_validate({"id": "r", "name": "r", "priority": 1, "stage": "s", "condition": {"weird": 1}, "action": {"type": "none"}})


def test_condition_depth_is_limited() -> None:
    deep: dict = {"attr": "x", "op": ">", "value": 1}
    for _ in range(4):
        deep = {"all": [deep]}
    with pytest.raises(ValidationError, match="nested deeper"):
        Rule.model_validate({"id": "r", "name": "r", "priority": 1, "stage": "s", "condition": deep, "action": {"type": "none"}})


def test_action_rules() -> None:
    base = {"id": "r", "name": "r", "priority": 1, "stage": "s"}
    with pytest.raises(ValidationError, match="needs a numeric value"):
        Rule.model_validate({**base, "action": {"type": "fixed"}})
    with pytest.raises(ValidationError, match="non-empty expression"):
        Rule.model_validate({**base, "action": {"type": "formula"}})
    assert Rule.model_validate({**base, "action": {"type": "none"}}).action.value is None


def test_constraint_ranges() -> None:
    with pytest.raises(ValidationError):
        Constraint(type="rounding", value=0)
    with pytest.raises(ValidationError):
        Constraint(type="max_discount", value=101)
    with pytest.raises(ValidationError):
        Constraint(type="unknown", value=1)  # type: ignore[arg-type]
    assert Constraint(type="markup_floor", value=150).value == 150  # markup may exceed 100 %


def test_domain_rules() -> None:
    bad = {**WIDGET_DOMAIN, "attributes": [*WIDGET_DOMAIN["attributes"], {"key": "cost", "label": "x", "type": "number", "source": "request"}]}
    with pytest.raises(ValidationError, match="reserved"):
        DomainConfig.model_validate(bad)
    bad = {**WIDGET_DOMAIN, "attributes": [{"key": "c", "label": "c", "type": "number", "source": "computed", "expression": "evil(1)"}]}
    with pytest.raises(ValidationError, match="Unknown function"):
        DomainConfig.model_validate(bad)
    bad = {**WIDGET_DOMAIN, "attributes": [{"key": "p", "label": "p", "type": "number", "source": "request", "semantic": "unitCost"}]}
    with pytest.raises(ValidationError, match="product record"):
        DomainConfig.model_validate(bad)
    bad = {**WIDGET_DOMAIN, "pipeline": ["a", "a"]}
    with pytest.raises(ValidationError, match="unique"):
        DomainConfig.model_validate(bad)


def test_price_request_has_no_price_field() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        PriceRequest.model_validate({"domain_id": "d", "product_id": "p", "as_of": "2026-01-01", "final_price": 1})


def test_strategy_rejects_duplicate_rule_ids() -> None:
    r = {"id": "x", "name": "x", "priority": 1, "stage": "s", "action": {"type": "none"}}
    with pytest.raises(ValidationError, match="duplicate rule id"):
        StrategyConfig.model_validate({"domain_id": "d", "name": "n", "rules": [r, r]})


def test_ml_feature_contract_matches_column_list() -> None:
    assert tuple(MLFeatures.model_fields) == ML_FEATURE_COLUMNS
    assert len(ML_FEATURE_COLUMNS) == 14


def test_llm_schema_excludes_formula_and_recursion() -> None:
    schema = LlmStrategyOut.model_json_schema()
    text = str(schema)
    assert "formula" not in text
    with pytest.raises(ValidationError):
        LlmStrategyOut.model_validate({"name": "n", "rules": [{"name": "r", "priority": 1, "stage": "s", "action": {"type": "formula"}}]})


def test_other_contracts_construct() -> None:
    OptimizerResponse.model_validate({"evaluations": [], "model": {"name": "m", "version": "1", "kind": "simulated"}})
    PersonaReaction.model_validate({
        "persona_id": "p", "name": "P", "price": 10, "buy_share": 0.5, "defer_share": 0.2, "reject_share": 0.3,
        "reaction": "ACCEPTS", "main_concern": "price", "concern_driver": "price", "explanation": "e",
    })
    ShadowValidation.model_validate({"status": "SKIPPED", "reason": "no valid price"})
    with pytest.raises(ValidationError):
        ShadowValidation.model_validate({"status": "PASS", "advisory_only": False})
    fields = set(CustomerQuote.model_fields)
    assert not fields & {"cost", "unit_cost", "p_buy", "applied_rules", "shadow_validation", "candidates"}
    assert {"shadow_validation", "candidates", "provenance"} <= set(PricingDecision.model_fields)
