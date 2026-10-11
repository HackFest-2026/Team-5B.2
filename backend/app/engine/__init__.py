"""The universal pricing engine: pure, deterministic, domain-agnostic.

No database, no clock, no randomness, no industry names. Domain behaviour comes only from
validated configuration (DomainConfig + StrategyConfig).

Typical flow::

    prepared = prepare(domain, product, strategy, request, version_id)
    response = optimizer.evaluate(prepared.optimizer_request())   # any PriceOptimizer
    decision = finalize(prepared, response, created_at=...)
"""

from .candidates import check_candidates, generate_candidates
from .conditions import evaluate_condition
from .conflicts import resolve_conflicts
from .constraints import PriceFacts, check_price, compute_bounds, evaluate_constraints, gross_margin_pct, markup_pct
from .context import InvalidInputError, coerce, enrich_context
from .customer import to_customer_quote
from .diff import diff_strategies
from .explain import condition_to_english, constraint_to_english, rule_to_english
from .expression import ExpressionError, check_expression, evaluate_expression, expression_variables
from .finalize import decision_id_for, finalize
from .hashing import canonical_json, content_hash
from .pipeline import apply_rules, cap_total_adjustment
from .prepare import Prepared, prepare
from .rules import order_rules, select_rules
from .signals import map_signals
from .validate import validate_bundle_payload, validate_strategy, validate_strategy_payload

__all__ = [name for name in dir() if not name.startswith("_")]
