"""Phase 4: domains, drafts, lifecycle, auth, quotes, audit, errors."""

from __future__ import annotations

import json

from tests.api.conftest import ADMIN, ANALYST, ANALYST2, CUSTOMER, H, MANAGER, MANAGER2, hotel_quote
from tests.conftest import EXAMPLES


def car_rental() -> dict:
    return json.loads((EXAMPLES / "car-rental.json").read_text(encoding="utf-8"))


def ready_to_approve(client, domain="hospitality", who=ANALYST):
    """Edit the draft (as `who`) and validate it."""
    draft = client.get(f"/api/v1/domains/{domain}/strategy", headers=who).json()["strategy"]
    draft["rules"][0]["action"]["value"] = 14
    assert client.put(f"/api/v1/domains/{domain}/strategy", headers=who, json=draft).status_code == 200
    r = client.post(f"/api/v1/domains/{domain}/validate", headers=who)
    assert r.json()["status"] == "VALIDATED", r.json()
    return draft


class TestDomains:
    def test_seeded_domains_and_health(self, client) -> None:
        ids = {d["id"] for d in client.get("/api/v1/domains", headers=ANALYST).json()}
        assert ids == {"hospitality", "gpu-compute", "edtech", "ecommerce", "banking"}
        h = client.get("/api/v1/health").json()
        assert h["llm"]["mock"] is True and h["optimizer"]["kind"] == "simulated"

    def test_create_fourth_domain_from_config(self, client) -> None:
        r = client.post("/api/v1/domains", headers=ANALYST, json=car_rental())
        assert r.status_code == 201 and r.json()["id"] == "car-rental"
        assert client.post("/api/v1/domains", headers=ANALYST, json=car_rental()).json()["error"]["code"] == "DOMAIN_EXISTS"
        assert client.get("/api/v1/domains/car-rental/strategy", headers=ANALYST).json()["status"] == "DRAFT"

    def test_invalid_domain_returns_paths(self, client) -> None:
        bad = car_rental()
        bad["strategy"]["rules"][0]["condition"]["op"] = "regex"
        r = client.post("/api/v1/domains", headers=ANALYST, json=bad)
        assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_STRATEGY"
        assert any("rules.0.condition.op" in d["path"] for d in r.json()["error"]["details"])

    def test_suggest_domain_mock(self, client) -> None:
        r = client.post("/api/v1/domains/suggest", headers=ANALYST, json={"description": "Coworking desks booked in advance, demand varies, loyal members"})
        assert r.status_code == 200 and r.json()["provider"] == "mock"
        keys = {a["key"] for a in r.json()["domain"]["attributes"]}
        assert {"demand_index", "start_date", "loyalty_score"} <= keys


class TestLifecycle:
    def test_edit_invalidates_validation(self, client) -> None:
        draft = ready_to_approve(client)
        draft["rules"][0]["action"]["value"] = 15
        r = client.put("/api/v1/domains/hospitality/strategy", headers=ANALYST, json=draft)
        assert r.json()["status"] == "DRAFT"
        r = client.post("/api/v1/domains/hospitality/approve", headers=MANAGER)
        assert r.status_code == 409 and r.json()["error"]["code"] == "STRATEGY_NOT_VALIDATED"

    def test_invalid_draft_cannot_be_validated_or_published(self, client) -> None:
        draft = client.get("/api/v1/domains/hospitality/strategy", headers=ANALYST).json()["strategy"]
        draft["constraints"] += [{"type": "floor", "value": 90000}]
        client.put("/api/v1/domains/hospitality/strategy", headers=ANALYST, json=draft)
        r = client.post("/api/v1/domains/hospitality/validate", headers=ANALYST).json()
        assert r["status"] == "DRAFT" and not r["report"]["ok"]
        assert client.post("/api/v1/domains/hospitality/approve", headers=MANAGER).status_code == 409

    def test_schema_invalid_put_is_rejected(self, client) -> None:
        draft = client.get("/api/v1/domains/hospitality/strategy", headers=ANALYST).json()["strategy"]
        draft["rules"][0]["action"] = {"type": "python", "expression": "__import__('os')"}
        r = client.put("/api/v1/domains/hospitality/strategy", headers=ANALYST, json=draft)
        assert r.status_code == 422

    def test_maker_checker(self, client) -> None:
        ready_to_approve(client, who=MANAGER)  # the manager edited it, so the manager may not approve
        r = client.post("/api/v1/domains/hospitality/approve", headers=MANAGER)
        assert r.status_code == 403 and r.json()["error"]["code"] == "MAKER_CHECKER"
        assert client.post("/api/v1/domains/hospitality/approve", headers=MANAGER2).status_code == 200

    def test_approve_publish_rollback_keep_one_active_and_versions_immutable(self, client) -> None:
        ready_to_approve(client)
        v2 = client.post("/api/v1/domains/hospitality/approve", headers=MANAGER, json={"notes": "uplift 14%"}).json()
        assert v2["id"] == "hospitality-v2" and v2["status"] == "APPROVED"
        assert client.post("/api/v1/domains/hospitality/publish", headers=MANAGER, json={"version_id": v2["id"]}).json()["status"] == "ACTIVE"
        versions = client.get("/api/v1/domains/hospitality/versions", headers=ANALYST).json()
        assert [v["status"] for v in versions] == ["ACTIVE", "RETIRED"]
        # Editing the draft never touches a published version.
        draft = client.get("/api/v1/domains/hospitality/strategy", headers=ANALYST).json()["strategy"]
        draft["rules"][0]["action"]["value"] = 30
        client.put("/api/v1/domains/hospitality/strategy", headers=ANALYST, json=draft)
        v2_after = next(v for v in client.get("/api/v1/domains/hospitality/versions", headers=ANALYST).json() if v["id"] == v2["id"])
        assert v2_after["strategy"]["rules"][0]["action"]["value"] == 14
        # Re-publishing an active version is an invalid transition.
        r = client.post("/api/v1/domains/hospitality/publish", headers=MANAGER, json={"version_id": v2["id"]})
        assert r.status_code == 409 and r.json()["error"]["code"] == "INVALID_TRANSITION"
        r = client.post("/api/v1/domains/hospitality/rollback", headers=MANAGER, json={"version_id": "hospitality-v1"})
        assert r.json()["status"] == "ACTIVE"
        assert [v["status"] for v in client.get("/api/v1/domains/hospitality/versions", headers=ANALYST).json()] == ["RETIRED", "ACTIVE"]
        # Rollback only targets retired versions.
        assert client.post("/api/v1/domains/hospitality/rollback", headers=MANAGER, json={"version_id": "hospitality-v1"}).status_code == 409

    def test_diff(self, client) -> None:
        ready_to_approve(client)
        d = client.get("/api/v1/domains/hospitality/versions/hospitality-v1/diff/draft", headers=ANALYST).json()
        assert d["has_changes"] and d["changed"][0]["fields"] == ["action"]


class TestAuth:
    def test_matrix(self, client) -> None:
        assert client.get("/api/v1/domains").status_code == 401
        assert client.get("/api/v1/domains", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert client.get("/api/v1/domains", headers=CUSTOMER).status_code == 403
        assert client.get("/api/v1/audit-logs", headers=CUSTOMER).status_code == 403
        assert client.get("/api/v1/pricing/decisions", headers=CUSTOMER).status_code == 403
        assert client.post("/api/v1/domains/hospitality/approve", headers=ANALYST).status_code == 403
        assert client.post("/api/v1/domains/hospitality/publish", headers=ANALYST, json={"version_id": "x"}).status_code == 403
        assert client.post("/api/v1/admin/reseed", headers=MANAGER).status_code == 403
        assert client.get("/api/v1/catalog", headers=CUSTOMER).status_code == 200
        assert client.post("/api/v1/pricing/quote", headers=CUSTOMER, json=hotel_quote()).status_code == 200


class TestQuotes:
    def test_business_quote_traces_everything(self, client) -> None:
        d = client.post("/api/v1/pricing/quote", headers=ANALYST, json=hotel_quote()).json()
        assert d["status"] == "OK" and d["strategy_version_id"] == "hospitality-v1" and d["version_number"] == 1
        assert d["candidates"] and d["applied_rules"] and d["constraints_evaluated"] and d["explanation"]["business"]
        assert d["provenance"]["optimizer"]["kind"] == "simulated"
        again = client.get(f"/api/v1/pricing/decisions/{d['decision_id']}", headers=ANALYST).json()
        assert again["final_price"] == d["final_price"]
        listed = client.get("/api/v1/pricing/decisions?domain_id=hospitality", headers=ANALYST).json()
        assert listed[0]["decision_id"] == d["decision_id"]

    def test_identical_quote_is_idempotent(self, client) -> None:
        a = client.post("/api/v1/pricing/quote", headers=ANALYST, json=hotel_quote()).json()
        b = client.post("/api/v1/pricing/quote", headers=ANALYST, json=hotel_quote()).json()
        assert a["decision_id"] == b["decision_id"] and a["created_at"] == b["created_at"]

    def test_customer_quote_is_safe(self, client) -> None:
        q = client.post("/api/v1/pricing/quote", headers=CUSTOMER, json=hotel_quote()).json()
        text = json.dumps(q)
        for word in ("cost", "margin", "p_buy", "rule", "expected_profit", "shadow", "violations", "signals"):
            assert word not in text, word
        assert q["factors"] and q["final_price"]
        assert "Member discount applied." in q["factors"]  # the customer's own profile (guest-gold) was used

    def test_customer_cannot_override_inventory(self, client) -> None:
        r = client.post("/api/v1/pricing/quote", headers=CUSTOMER, json=hotel_quote(attributes={"check_in": "2026-10-24", "occupancy": 10}))
        assert r.status_code == 422 and "occupancy" in json.dumps(r.json())

    def test_caller_price_rejected(self, client) -> None:
        r = client.post("/api/v1/pricing/quote", headers=ANALYST, json=hotel_quote(final_price=1))
        assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_bad_attributes_and_unknown_product(self, client) -> None:
        r = client.post("/api/v1/pricing/quote", headers=ANALYST, json=hotel_quote(attributes={"check_in": "soon"}))
        assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_INPUT"
        assert client.post("/api/v1/pricing/quote", headers=ANALYST, json=hotel_quote(product_id="NOPE")).status_code == 404

    def test_no_active_version(self, client) -> None:
        client.post("/api/v1/domains", headers=ANALYST, json=car_rental())
        r = client.post("/api/v1/pricing/quote", headers=ANALYST,
                        json={"domain_id": "car-rental", "product_id": "SUV", "attributes": {"pickup_date": "2026-10-20"}})
        assert r.status_code == 409 and r.json()["error"]["code"] == "NO_ACTIVE_VERSION"

    def test_no_valid_price_is_a_200(self, client) -> None:
        bundle = car_rental()
        bundle["domain"]["id"] = bundle["strategy"]["domain_id"] = "tight"
        bundle["strategy"]["constraints"] = [{"type": "max_increase", "value": 0}, {"type": "max_discount", "value": 0},
                                             {"type": "rounding", "value": 7}]
        client.post("/api/v1/domains", headers=ANALYST, json=bundle)
        client.post("/api/v1/domains/tight/validate", headers=ANALYST2)
        v = client.post("/api/v1/domains/tight/approve", headers=MANAGER).json()
        client.post("/api/v1/domains/tight/publish", headers=MANAGER, json={"version_id": v["id"]})
        r = client.post("/api/v1/pricing/quote", headers=ANALYST, json={"domain_id": "tight", "product_id": "SUV",
                                                                       "attributes": {"pickup_date": "2026-10-20"}})
        assert r.status_code == 200 and r.json()["status"] == "NO_VALID_PRICE" and r.json()["final_price"] is None

    def test_catalog_hides_internals(self, client) -> None:
        cat = client.get("/api/v1/catalog", headers=CUSTOMER).json()
        hotel = next(c for c in cat if c["domain_id"] == "hospitality")
        assert {i["key"] for i in hotel["inputs"]} == {"check_in", "nights", "channel"}
        def keys(o):
            if isinstance(o, dict):
                return set(o) | set().union(*(keys(v) for v in o.values()))
            if isinstance(o, list):
                return set().union(*(keys(v) for v in o)) if o else set()
            return set()
        assert not keys(cat) & {"cost", "base_price", "inventory", "rules", "constraints", "expression", "previous_price"}


class TestAudit:
    def test_mutations_are_audited_including_failures(self, client) -> None:
        ready_to_approve(client)
        client.post("/api/v1/domains/hospitality/approve", headers=ANALYST)  # 403, not audited (no access)
        client.post("/api/v1/domains/hospitality/publish", headers=MANAGER, json={"version_id": "hospitality-v1"})  # 409
        logs = client.get("/api/v1/audit-logs?resource_type=version", headers=ANALYST).json()
        failed = [l for l in logs if l["operation"] == "PUBLISH" and l["outcome"] == "FAILED"]
        assert failed and failed[0]["actor"] == "maya"
        ops = {l["operation"] for l in client.get("/api/v1/audit-logs?limit=500", headers=ANALYST).json()}
        assert {"CREATE_DOMAIN", "UPDATE_DRAFT", "VALIDATE", "APPROVE", "PUBLISH"} <= ops

    def test_reseed_admin_only(self, client) -> None:
        client.post("/api/v1/domains", headers=ANALYST, json=car_rental())
        r = client.post("/api/v1/admin/reseed", headers=ADMIN)
        assert r.json()["seeded"] == ["hospitality", "gpu-compute", "edtech", "ecommerce", "banking"]
        assert "car-rental" not in {d["id"] for d in client.get("/api/v1/domains", headers=ANALYST).json()}
        assert H("customer")  # helper sanity
