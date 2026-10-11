"""Phase 3: feature contract, ML adapter, simulator, selection and engine re-verification."""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

from app.config import Settings
from app.engine import finalize, prepare
from app.optimizer import SimulatedPriceOptimizer, build_features, build_optimizer
from app.optimizer.factory import OptimizerStatus
from app.schemas import CheckedCandidate, ML_FEATURE_COLUMNS, OptimizerRequest, OptimizerResponse
from tests.conftest import ROOT, load_bundle, make_strategy, req

ARTIFACTS = ROOT / "ml_updated" / "artifacts"
HAS_ML = (ARTIFACTS / "model_gbt.pkl").is_file()
needs_ml = pytest.mark.skipif(not HAS_ML, reason="ml_updated artifacts not present")


def cand(price: float, eligible: bool = True) -> CheckedCandidate:
    return CheckedCandidate(price=price, reasons=["test"], eligible=eligible)


def readme_request(**over) -> OptimizerRequest:
    data = dict(
        domain_id="d", strategy_version_id="v", base_price=100, unit_cost=40, competitor_price=95,
        signals={"customerSensitivity": 0.65, "customerLoyalty": 0.30},
        candidates=[cand(p) for p in (80, 90, 100, 110, 120)],
    )
    data.update(over)
    return OptimizerRequest(**data)


@pytest.fixture(scope="module")
def ml():
    from app.optimizer.ml import MLPriceOptimizer

    return MLPriceOptimizer(ARTIFACTS, "gbt")


def _load_reference_features():
    spec = importlib.util.spec_from_file_location("ml_updated_features", ROOT / "ml_updated" / "features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class TestFeatures:
    def test_columns_and_rank_over_all_candidates(self) -> None:
        r = readme_request(candidates=[cand(110), cand(80, eligible=False), cand(100)])
        b = build_features(r)
        assert b.prices == [80, 100, 110]
        assert [row.price_rank_normalized for row in b.rows] == [0.0, 0.5, 1.0]
        assert tuple(type(b.rows[0]).model_fields) == ML_FEATURE_COLUMNS
        assert len(b.matrix()[0]) == 14

    def test_defaults_and_ood_are_reported(self) -> None:
        r = readme_request(competitor_price=None, signals={}, candidates=[cand(40), cand(100)])
        b = build_features(r)
        joined = " ".join(b.warnings)
        assert "DEFAULT_SIGNALS:" in joined and "competitorPrice" in joined and "DEFAULT_PERCEIVED_VALUE" in joined
        assert "OOD_FEATURE:price_ratio" in joined  # 0.4 x base is outside 0.6-1.5

    @needs_ml
    def test_matches_ml_updated_reference_implementation(self) -> None:
        ref = _load_reference_features()
        rng = random.Random(7)
        for _ in range(25):
            prices = sorted(round(rng.uniform(60, 140), 2) for _ in range(5))
            sig = {"customerSensitivity": rng.random(), "customerLoyalty": rng.random(), "urgency": rng.random(),
                   "scarcity": rng.random(), "demandPressure": rng.uniform(0.2, 2), "competitorDensity": rng.random(),
                   "promotionActive": rng.choice([0.0, 1.0]), "timeDecay": rng.random()}
            r = readme_request(signals=sig, competitor_price=rng.uniform(70, 130), candidates=[cand(p) for p in prices])
            ours = build_features(r)
            for i, (p, row) in enumerate(zip(ours.prices, ours.rows)):
                theirs = ref.build_candidate_features(
                    p, r.base_price, r.unit_cost, r.competitor_price, i / 4, sig["customerSensitivity"], sig["customerLoyalty"],
                    sig["urgency"], sig["scarcity"], sig["demandPressure"], sig["competitorDensity"],
                    int(sig["promotionActive"]), sig["timeDecay"], None,
                )
                assert row.model_dump() == pytest.approx(theirs)


class TestSimulated:
    def test_schema_profit_math_and_determinism(self) -> None:
        opt = SimulatedPriceOptimizer()
        a, b = opt.evaluate(readme_request()), opt.evaluate(readme_request())
        assert a.model_dump() == b.model_dump()
        OptimizerResponse.model_validate(a.model_dump())
        assert a.model.kind == "simulated" and a.model.training_data == "n/a"
        for e in a.evaluations:
            assert e.expected_profit == pytest.approx(e.p_buy * (e.price - 40), abs=1e-5)
        probs = [e.p_buy for e in a.evaluations]
        assert probs == sorted(probs, reverse=True)  # cheaper → more likely to buy

    def test_selects_highest_profit_not_highest_probability(self) -> None:
        r = SimulatedPriceOptimizer().evaluate(readme_request())
        best = max(r.evaluations, key=lambda e: e.expected_profit)
        assert r.selected_price == best.price
        assert max(r.evaluations, key=lambda e: e.p_buy).price == 80 != r.selected_price

    def test_conversion_floor_and_ineligible(self) -> None:
        r = SimulatedPriceOptimizer().evaluate(readme_request(conversion_floor=0.99))
        assert r.selected_price is None
        assert {e.selection_status for e in r.evaluations} == {"BELOW_CONVERSION_FLOOR"}
        cands = [cand(p, eligible=p < 100) for p in (80, 90, 100, 110, 120)]
        r = SimulatedPriceOptimizer().evaluate(readme_request(candidates=cands))
        assert r.selected_price in (80, 90)
        assert [e.selection_status for e in r.evaluations if e.price >= 100] == ["INELIGIBLE"] * 3


@needs_ml
class TestML:
    def test_loads_with_matching_version_and_hash(self, ml) -> None:
        assert ml.info.kind == "ml" and ml.info.training_data == "synthetic"
        assert "sklearn-1.6.1" in ml.info.version and len(ml.info.artifact_sha256) == 64
        assert list(ml.model.n_features_in_ for _ in [0]) == [14]

    def test_readme_demo_case_picks_90(self, ml) -> None:
        r = ml.evaluate(readme_request())
        assert r.selected_price == 90  # the README's worked example: $90 has the top E[profit]
        probs = [e.p_buy for e in r.evaluations]
        assert probs == sorted(probs, reverse=True)  # monotone in price
        assert all(e.features is not None for e in r.evaluations)
        assert r.model.name == "ml_updated-gbt"

    def test_deterministic(self, ml) -> None:
        assert ml.evaluate(readme_request()).model_dump() == ml.evaluate(readme_request()).model_dump()

    def test_hash_pin_is_enforced(self) -> None:
        from app.optimizer.ml import ModelLoadError, MLPriceOptimizer

        with pytest.raises(ModelLoadError, match="hash mismatch"):
            MLPriceOptimizer(ARTIFACTS, "gbt", expected_sha256="0" * 64)

    def test_lr_baseline(self) -> None:
        from app.optimizer.ml import MLPriceOptimizer

        r = MLPriceOptimizer(ARTIFACTS, "lr").evaluate(readme_request())
        assert r.model.name == "ml_updated-lr" and r.selected_price is not None

    def test_artifacts_untouched(self, ml) -> None:
        before = {p.name: p.stat().st_mtime for p in ARTIFACTS.iterdir()}
        ml.evaluate(readme_request())
        assert {p.name: p.stat().st_mtime for p in ARTIFACTS.iterdir()} == before

    def test_full_engine_with_ml_on_every_domain(self, ml) -> None:
        for name, attrs in [("hospitality", {"check_in": "2026-10-24"}), ("gpu-compute", {}), ("edtech", {"session_date": "2026-10-13"})]:
            b = load_bundle(name)
            for product in b.domain.products:
                p = prepare(b.domain, product, b.strategy, req(attrs, domain_id=b.domain.id, product_id=product.id, as_of="2026-10-11"), "v1")
                d = finalize(p, ml.evaluate(p.optimizer_request()))
                assert d.status == "OK", (name, product.id, d.status_reason)
                assert d.provenance.optimizer.kind == "ml" and d.provenance.simulated  # synthetic training data
                assert not any(w.startswith("OPTIMIZER_SELECTION_OVERRIDDEN") for w in d.warnings)


class TestFactory:
    def test_auto_falls_back_to_simulated(self, tmp_path: Path) -> None:
        status = OptimizerStatus()
        opt = build_optimizer(Settings(optimizer_provider="auto", ml_artifacts_dir=str(tmp_path)), status)
        assert opt.info.kind == "simulated" and "not found" in status.fallback_reason

    def test_ml_mode_is_strict(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            build_optimizer(Settings(optimizer_provider="ml", ml_artifacts_dir=str(tmp_path)))

    @needs_ml
    def test_auto_uses_ml_when_present(self) -> None:
        assert build_optimizer(Settings(optimizer_provider="auto", ml_artifacts_dir=str(ARTIFACTS))).info.kind == "ml"

    def test_http_optimizer_validates_response(self) -> None:
        import httpx

        from app.optimizer.http import HttpPriceOptimizer

        good = SimulatedPriceOptimizer().evaluate(readme_request()).model_dump(mode="json")
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=good))
        opt = HttpPriceOptimizer("http://model.local/evaluate", client=httpx.Client(transport=transport))
        assert opt.evaluate(readme_request()).selected_price == good["selected_price"]
        bad = httpx.MockTransport(lambda request: httpx.Response(200, json={"evaluations": "nope"}))
        with pytest.raises(Exception):
            HttpPriceOptimizer("http://x", client=httpx.Client(transport=bad)).evaluate(readme_request())


def test_engine_rejects_ineligible_pick_from_a_real_optimizer(domain, product) -> None:
    s = make_strategy(rules=[], constraints=[{"type": "max_increase", "value": 5}])
    p = prepare(domain, product, s, req(), "v1")
    resp = SimulatedPriceOptimizer().evaluate(p.optimizer_request())
    resp.selected_price = 110  # tamper: choose the price that breaks max_increase
    d = finalize(p, resp)
    assert d.final_price != 110 and any(w.startswith("OPTIMIZER_SELECTION_OVERRIDDEN") for w in d.warnings)
