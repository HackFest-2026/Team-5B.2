"""Phase 4b/4c: strategy generation (mock + fake LLMs) and shadow validation through the API."""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.llm.mock import generate_strategy
from tests.api.conftest import ANALYST, CUSTOMER, MANAGER, hotel_quote
from tests.conftest import load_bundle

HOTEL_BRIEF = ("Create a hotel-room pricing strategy. Increase prices when demand is high and occupancy is above 80%. "
               "Offer a discount when demand is low, but never price below the minimum acceptable margin. "
               "Limit the total price increase to 20%.")


class FakeLlm:
    """Scripted JSON responses, one per call."""

    name = "gemini"
    model = "fake"

    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate_json(self, system: str, user: str, schema) -> Any:
        self.prompts.append(user)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def good_llm_output() -> dict:
    return {
        "name": "Demand strategy", "objective_text": HOTEL_BRIEF,
        "rules": [{"name": "High demand", "condition": {"match": "all", "conditions": [
            {"attr": "demand_index", "op": ">=", "value": 70}, {"attr": "occupancy", "op": ">", "value": 80}]},
            "action": {"type": "percent", "value": 12}, "priority": 10, "stage": "demand", "customer_message": "High demand."},
            {"name": "Not corporate", "condition": {"conditions": [{"attr": "channel", "op": "==", "value": "Corporate", "negate": True}]},
             "action": {"type": "none"}, "priority": 20, "stage": "channel"}],
        "constraints": [{"type": "margin_floor", "value": 25}, {"type": "max_increase", "value": 20}],
        "assumptions": ["high demand = index ≥ 70"],
    }


class TestMockGenerator:
    def test_hotel_brief(self) -> None:
        out = generate_strategy(load_bundle("hospitality").domain, HOTEL_BRIEF)
        types = {c["type"] for c in out["constraints"]}
        assert {"margin_floor", "max_increase"} <= types
        up = out["rules"][0]
        assert up["action"]["value"] > 0 and {c["attr"] for c in up["condition"]["conditions"]} == {"occupancy", "demand_index"}
        assert out["rules"][1]["action"]["value"] < 0
        assert out["assumptions"]

    def test_generate_endpoint_labels_mock(self, client) -> None:
        r = client.post("/api/v1/domains/hospitality/generate-strategy", headers=ANALYST, json={"description": HOTEL_BRIEF})
        j = r.json()
        assert r.status_code == 200 and j["provider"] == "mock" and "Mock" in j["provider_label"]
        assert j["status"] == "DRAFT" and j["source"] == "mock" and j["strategy"]["rules"]
        assert client.post("/api/v1/domains/hospitality/validate", headers=ANALYST).json()["status"] == "VALIDATED"


class TestRealProviderPipeline:
    def test_valid_output_becomes_a_draft(self, make_client) -> None:
        c = make_client(llm=FakeLlm(good_llm_output()))
        j = c.post("/api/v1/domains/hospitality/generate-strategy", headers=ANALYST, json={"description": HOTEL_BRIEF}).json()
        assert j["provider"] == "gemini" and j["source"] == "llm"
        rules = j["strategy"]["rules"]
        assert rules[0]["id"] == "LLM-01" and "all" in rules[0]["condition"]
        assert rules[1]["condition"] == {"not": {"attr": "channel", "op": "==", "value": "Corporate"}}

    def test_repair_retry_then_success(self, make_client) -> None:
        bad = good_llm_output()
        bad["rules"][0]["condition"]["conditions"][0]["attr"] = "moon_phase"
        llm = FakeLlm(bad, good_llm_output())
        c = make_client(llm=llm)
        r = c.post("/api/v1/domains/hospitality/generate-strategy", headers=ANALYST, json={"description": HOTEL_BRIEF})
        assert r.status_code == 200
        assert "moon_phase" in llm.prompts[1]  # the validation error was sent back for repair

    @pytest.mark.parametrize("junk", [
        {"rules": "lots"},
        {"name": "x", "rules": [{"name": "r", "priority": 1, "stage": "demand", "action": {"type": "formula", "value": 1}}]},
        {"name": "x", "rules": [{"name": "r", "priority": 1, "stage": "demand", "action": {"type": "percent", "value": 5},
                                 "condition": {"conditions": [{"attr": "occupancy", "op": "eval", "value": "__import__('os')"}]}}]},
        {"name": "x", "final_price": 4999},
    ])
    def test_malformed_output_is_rejected_and_draft_unchanged(self, make_client, junk) -> None:
        c = make_client(llm=FakeLlm(junk, junk))
        before = c.get("/api/v1/domains/hospitality/strategy", headers=ANALYST).json()
        r = c.post("/api/v1/domains/hospitality/generate-strategy", headers=ANALYST, json={"description": HOTEL_BRIEF})
        assert r.status_code == 422 and r.json()["error"]["code"] == "GENERATED_STRATEGY_INVALID"
        assert c.get("/api/v1/domains/hospitality/strategy", headers=ANALYST).json() == before
        logs = c.get("/api/v1/audit-logs?operation=GENERATE_STRATEGY", headers=ANALYST).json()
        assert logs[0]["outcome"] == "REJECTED"

    def test_provider_failure_is_502(self, make_client) -> None:
        c = make_client(llm=FakeLlm(RuntimeError("quota exceeded")))
        r = c.post("/api/v1/domains/hospitality/generate-strategy", headers=ANALYST, json={"description": HOTEL_BRIEF})
        assert r.status_code == 502 and r.json()["error"]["code"] == "LLM_UNAVAILABLE"

    def test_missing_key_falls_back_to_labelled_mock(self, make_client) -> None:
        c = make_client(llm_provider="gemini", gemini_api_key="")
        assert c.get("/api/v1/health").json()["llm"]["provider"] == "mock"


class TestShadowValidation:
    def test_shadow_invariance(self, make_client) -> None:
        off = make_client(validation_mode="off", db="off.db").post("/api/v1/pricing/quote", headers=ANALYST, json=hotel_quote()).json()
        on = make_client(validation_mode="sync", db="on.db").post("/api/v1/pricing/quote", headers=ANALYST,
                                                      json=hotel_quote(include_personas=True)).json()
        assert on["final_price"] == off["final_price"] and on["status"] == off["status"]
        assert [c["selection_status"] for c in on["candidates"]] == [c["selection_status"] for c in off["candidates"]]
        sv = on["shadow_validation"]
        assert sv["status"] in ("PASS", "REVIEW_REQUIRED", "FAIL") and sv["advisory_only"] is True
        assert [c["key"] for c in sv["checks"]] == ["A_behaviour", "B_heterogeneity", "C_scenarios", "D_ranking",
                                                    "E_model_disagreement", "F_ml_monotonicity"]
        assert len(sv["personas"]) == 7 and len(sv["personas"][0]) == len(on["candidates"])  # 6 defaults + 1 domain persona
        assert sv["personas"][-1][0]["persona_id"] == "business_traveller"
        assert len(on["persona_summary"]) == 7 and all(p["price"] == on["final_price"] for p in on["persona_summary"])

    def test_background_validation_pending_then_stored(self, make_client) -> None:
        c = make_client(validation_mode="background")
        d = c.post("/api/v1/pricing/quote", headers=ANALYST, json=hotel_quote()).json()
        # TestClient runs background tasks before returning, so the stored result is already final.
        v = c.get(f"/api/v1/pricing/decisions/{d['decision_id']}/validation", headers=ANALYST).json()
        assert v["status"] in ("PASS", "REVIEW_REQUIRED", "FAIL")
        listed = c.get("/api/v1/pricing/decisions", headers=ANALYST).json()
        assert listed[0]["validation_status"] == v["status"]

    def test_customer_never_sees_validation(self, make_client) -> None:
        q = make_client(validation_mode="sync").post("/api/v1/pricing/quote", headers=CUSTOMER, json=hotel_quote()).json()
        assert "shadow" not in json.dumps(q) and "persona" not in json.dumps(q)

    def test_simulator_exception_is_contained(self, make_client, monkeypatch) -> None:
        import app.validation.service as svc

        def boom(**kwargs):
            raise RuntimeError("simulator exploded")

        monkeypatch.setattr(svc.cas, "validate_offer", boom)
        d = make_client(validation_mode="sync").post("/api/v1/pricing/quote", headers=ANALYST, json=hotel_quote()).json()
        assert d["status"] == "OK" and d["final_price"] is not None
        assert d["shadow_validation"]["status"] == "ERROR" and "exploded" in d["shadow_validation"]["error"]

    def test_simulate_endpoint(self, client) -> None:
        body = {"as_of": "2026-10-11", "scenarios": [
            {"label": "High demand", "product_id": "DLX", "attributes": {"check_in": "2026-10-24", "demand_index": 90, "occupancy": 95}},
            {"label": "Bad input", "product_id": "DLX", "attributes": {"check_in": "someday"}},
        ]}
        r = client.post("/api/v1/domains/hospitality/simulate", headers=ANALYST, json=body).json()
        assert r["version_id"].startswith("draft:") and r["validation"]["ok"]
        hi = r["results"][0]["decision"]
        assert hi["shadow_validation"]["personas"] and hi["applied_rules"][0]["rule_id"] == "HOS-DEM-HI"
        assert r["results"][1]["error"] and r["results"][1]["decision"] is None
        v1 = client.post("/api/v1/domains/hospitality/simulate", headers=ANALYST,
                         json={**body, "version_id": "hospitality-v1", "scenarios": body["scenarios"][:1]}).json()
        assert v1["version_id"] == "hospitality-v1"
        assert client.post("/api/v1/domains/hospitality/simulate", headers=CUSTOMER, json=body).status_code == 403
        assert MANAGER
