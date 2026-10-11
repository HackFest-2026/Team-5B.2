"""Strategy validation: schema errors and semantic checks, reported in one ValidationReport.

Kept separate from runtime evaluation. The API, the LLM pipeline and the approval flow all go
through ``validate_strategy_payload`` so raw, untrusted JSON is checked the same way everywhere.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import ValidationError

from app.schemas.common import BUILTIN_KEYS, FORMULA_KEYS
from app.schemas.domain import Attribute, DomainConfig
from app.schemas.rules import Rule, iter_leaves
from app.schemas.strategy import DomainBundle, StrategyConfig
from app.schemas.validation import IssueCode, Severity, ValidationIssue, ValidationReport

from .constraints import PriceFacts, compute_bounds
from .expression import check_expression, expression_variables
from .reachability import numeric as _numeric
from .reachability import unreachable_reason
from .ties import order_sensitive_ties, sample_context_for

ML_RATIO_RANGE = (0.6, 1.5)  # price/base range the ML model was trained on


class _Collector:
    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    def add(self, severity: Severity, code: IssueCode, message: str, *, path: Optional[str] = None,
            rule_ids: Optional[list[str]] = None, details: Optional[dict[str, Any]] = None) -> None:
        self.issues.append(ValidationIssue(
            id=f"V{len(self.issues) + 1}", severity=severity, code=code, message=message,
            path=path, rule_ids=rule_ids or [], details=details,
        ))

    def report(self) -> ValidationReport:
        counts = {"error": 0, "warning": 0, "info": 0}
        for i in self.issues:
            counts[i.severity] += 1
        return ValidationReport(ok=counts["error"] == 0, issues=self.issues, counts=counts)


# ---------- schema (raw payload) ----------


def _loc(loc: tuple) -> str:
    # Drop pydantic's condition-union tags so paths read like the JSON
    # (the tag "all" is immediately followed by the field "all", so collapse the repeat).
    out: list[str] = []
    for p in loc:
        s = str(p)
        if s == "leaf" or (s in ("all", "any", "not") and out and out[-1] == s):
            continue
        out.append(s)
    return ".".join(out)


def schema_issues(err: ValidationError, prefix: str = "") -> list[tuple[IssueCode, str, str]]:
    out: list[tuple[IssueCode, str, str]] = []
    for e in err.errors(include_url=False):
        path = _loc(e["loc"])
        path = f"{prefix}{path}" if path else prefix.rstrip(".")
        code: IssueCode = "schema"
        if path.endswith(".op") and e["type"] == "literal_error":
            code = "unsupported_operator"
        msg = e["msg"]
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        out.append((code, path, f"{path}: {msg}" if path else msg))
    return out


def validate_strategy_payload(
    domain: DomainConfig, raw: Any, allow_formula: Optional[bool] = None
) -> tuple[Optional[StrategyConfig], ValidationReport]:
    try:
        strategy = StrategyConfig.model_validate(raw)
    except ValidationError as e:
        col = _Collector()
        for code, path, msg in schema_issues(e):
            col.add("error", code, msg, path=path)
        return None, col.report()
    return strategy, validate_strategy(domain, strategy, allow_formula)


def validate_bundle_payload(raw: Any) -> tuple[Optional[DomainBundle], ValidationReport]:
    try:
        bundle = DomainBundle.model_validate(raw)
    except ValidationError as e:
        col = _Collector()
        for code, path, msg in schema_issues(e):
            col.add("error", code, msg, path=path)
        return None, col.report()
    report = validate_strategy(bundle.domain, bundle.strategy)
    for issue in report.issues:  # semantic paths are relative to the strategy
        issue.path = f"strategy.{issue.path}" if issue.path else "strategy"
    return bundle, report


# ---------- semantic ----------


def validate_strategy(domain: DomainConfig, strategy: StrategyConfig, allow_formula: Optional[bool] = None) -> ValidationReport:
    if allow_formula is None:
        allow_formula = strategy.source == "manual"
    col = _Collector()
    attrs = {a.key: a for a in domain.attributes}
    known = set(attrs) | set(BUILTIN_KEYS)

    if strategy.domain_id != domain.id:
        col.add("error", "schema", f'strategy.domain_id "{strategy.domain_id}" does not match domain "{domain.id}"', path="domain_id")

    for i, r in enumerate(strategy.rules):
        _check_rule(col, r, i, domain, attrs, known, allow_formula)

    _check_constraints(col, domain, strategy)
    _check_policies(col, strategy)
    for issue in order_sensitive_ties(domain, strategy):
        col.add("warning", "priority_tie", issue["message"], rule_ids=issue["rule_ids"], details=issue["details"])
    return col.report()


def _check_rule(col: _Collector, r: Rule, i: int, domain: DomainConfig, attrs: dict[str, Attribute],
                known: set[str], allow_formula: bool) -> None:
    path = f"rules.{i}"
    name = r.name or r.id
    if r.stage not in domain.pipeline:
        col.add("error", "schema", f'"{name}" uses stage "{r.stage}", which is not in the pipeline ({" → ".join(domain.pipeline)}).',
                path=f"{path}.stage", rule_ids=[r.id])
    if r.effective_from and r.effective_to and r.effective_from > r.effective_to:
        col.add("error", "unreachable", f'"{name}" is effective from {r.effective_from} to {r.effective_to}, an empty window.',
                path=path, rule_ids=[r.id])

    a = r.action
    if a.type == "formula":
        if not allow_formula:
            col.add("error", "formula_not_allowed", f'"{name}" uses a formula action, which is not allowed in generated strategies.',
                    path=f"{path}.action", rule_ids=[r.id])
        else:
            err = check_expression(a.expression or "")
            if err:
                col.add("error", "formula", f'"{name}" has an invalid formula: {err}.', path=f"{path}.action.expression", rule_ids=[r.id])
            else:
                unknown = [v for v in expression_variables(a.expression or "") if v not in known and v not in FORMULA_KEYS]
                if unknown:
                    col.add("error", "unknown_attribute", f'"{name}" formula references unknown attribute(s): {", ".join(unknown)}.',
                            path=f"{path}.action.expression", rule_ids=[r.id])
    elif a.value is not None:
        if a.type == "percent" and a.value <= -100:
            col.add("error", "discount_over_100", f'"{name}" discounts {abs(a.value):g}%, which would make the price zero or negative.',
                    path=f"{path}.action.value", rule_ids=[r.id])
        if a.type == "set" and a.value <= 0:
            col.add("error", "invalid_value", f'"{name}" sets a non-positive target price.', path=f"{path}.action.value", rule_ids=[r.id])
        if a.type == "multiply" and a.value <= 0:
            col.add("error", "invalid_value", f'"{name}" multiplies by {a.value:g}, which yields a non-positive price.',
                    path=f"{path}.action.value", rule_ids=[r.id])

    for leaf in iter_leaves(r.condition):
        if leaf.attr not in known:
            col.add("error", "unknown_attribute", f'"{name}" refers to unknown attribute "{leaf.attr}".', path=f"{path}.condition", rule_ids=[r.id])
            continue
        attr = attrs.get(leaf.attr)
        if attr is None:
            continue
        values = leaf.value if isinstance(leaf.value, list) else [leaf.value]
        if attr.type == "enum" and attr.allowed_values:
            bad = [str(v) for v in values if str(v) not in attr.allowed_values]
            if bad:
                col.add("warning", "invalid_value",
                        f'"{name}": {attr.label} has no value(s) {", ".join(repr(b) for b in bad)} (allowed: {", ".join(attr.allowed_values)}).',
                        path=f"{path}.condition", rule_ids=[r.id])
        if attr.type == "number" and leaf.op in ("<", "<=", ">", ">=", "==", "!=", "between"):
            if any(not _numeric(v) for v in values):
                col.add("error", "invalid_value", f'"{name}": {attr.label} is numeric but compared to a non-number.',
                        path=f"{path}.condition", rule_ids=[r.id])
        if attr.type == "boolean" and leaf.op not in ("==", "!="):
            col.add("error", "unsupported_operator", f'"{name}": {attr.label} is a yes/no value and only supports == and !=.',
                    path=f"{path}.condition", rule_ids=[r.id])

    why = unreachable_reason(r.condition, attrs)
    if why:
        col.add("warning", "unreachable", f'"{name}" can never fire: {why}.', path=f"{path}.condition", rule_ids=[r.id])


def _check_constraints(col: _Collector, domain: DomainConfig, strategy: StrategyConfig) -> None:
    cons = strategy.constraints
    floors = [c.value for c in cons if c.type == "floor"]
    ceils = [c.value for c in cons if c.type == "ceiling"]
    if floors and ceils and max(floors) > min(ceils):
        col.add("error", "floor_above_ceiling", f"Price floor ({max(floors):g}) is above the ceiling ({min(ceils):g}).", path="constraints")
    types = [c.type for c in cons]
    for t in sorted(set(types)):
        if types.count(t) > 1:
            if t == "rounding":
                col.add("error", "invalid_constraint", "There is more than one rounding step; keep exactly one.", path="constraints")
            else:
                col.add("info", "invalid_constraint", f'There are multiple "{t}" constraints; the strictest one applies.', path="constraints")

    low_side: list[str] = []
    high_side: list[str] = []
    for p in domain.products:
        b = compute_bounds(cons, PriceFacts(p.base_price, p.cost, p.previous_price, domain.allow_below_cost))
        if b.conflict:
            lo = "∞" if b.lower is None else f"{b.lower:g}"
            up = "none" if b.upper is None else f"{b.upper:g}"
            col.add("error", "bounds_conflict",
                    f'No price can satisfy the constraints for "{p.name}": lower bound {lo} ({", ".join(b.lower_by)}) '
                    f'vs upper bound {up} ({", ".join(b.upper_by)}).',
                    path="constraints", details={"product_id": p.id})
            continue
        if b.upper is None or b.upper > p.base_price * ML_RATIO_RANGE[1]:
            high_side.append(p.id)
        if (b.lower or 0) < p.base_price * ML_RATIO_RANGE[0]:
            low_side.append(p.id)
    if high_side:
        col.add("warning", "ood_risk",
                f"Constraints allow prices above {ML_RATIO_RANGE[1]:g}× base for {', '.join(high_side)}; "
                "purchase predictions are outside the model's training range there. Consider max_increase or a ceiling.",
                path="constraints")
    if low_side:
        col.add("warning", "ood_risk",
                f"Constraints allow prices below {ML_RATIO_RANGE[0]:g}× base for {', '.join(low_side)}; "
                "purchase predictions are outside the model's training range there. Consider max_discount or a floor.",
                path="constraints")


def _check_policies(col: _Collector, strategy: StrategyConfig) -> None:
    if strategy.optimizer_policy.conversion_floor >= 0.9:
        col.add("warning", "policy", f"Conversion floor {strategy.optimizer_policy.conversion_floor:.0%} will reject almost every candidate.",
                path="optimizer_policy.conversion_floor")
    distinct = {o for o in strategy.candidate_policy.offsets_pct if o != 0}
    if len(distinct) + 1 > strategy.candidate_policy.count:
        col.add("info", "policy",
                f"{len(distinct) + 1} offset candidates exceed the candidate count {strategy.candidate_policy.count}; the farthest offsets are dropped.",
                path="candidate_policy")
    if strategy.conflict_policy.mode != "stack" and any(r.action.type == "set" for r in strategy.rules):
        pass  # set actions are legitimate in every mode


__all__ = ["sample_context_for", "unreachable_reason", "validate_bundle_payload", "validate_strategy", "validate_strategy_payload"]
