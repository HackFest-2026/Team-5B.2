"""End to end: a new domain from configuration → generated strategy → edit → validate → approve →
publish → quote → decision → v2 → rollback → quote cites v1. Plus: the same engine prices all seeded
domains, and the Alembic schema matches the ORM models."""

from __future__ import annotations

import json

from sqlalchemy import create_engine, inspect

from app.db import Base, run_migrations
from app import models  # noqa: F401
from tests.api.conftest import ANALYST, CUSTOMER, MANAGER, MANAGER2, client, make_client  # noqa: F401 (fixtures)
from tests.conftest import EXAMPLES

CAR = {"domain_id": "car-rental", "product_id": "SUV",
       "attributes": {"pickup_date": "2026-10-20", "rental_days": 7, "driver_age": 23}, "as_of": "2026-10-11"}


def test_full_pipeline(make_client) -> None:
    c = make_client(validation_mode="sync")
    bundle = json.loads((EXAMPLES / "car-rental.json").read_text(encoding="utf-8"))
    domain_only = bundle["domain"]
    assert c.post("/api/v1/domains", headers=ANALYST, json=domain_only).status_code == 201

    gen = c.post("/api/v1/domains/car-rental/generate-strategy", headers=ANALYST, json={
        "description": "Increase prices when fleet utilization is above 85%. Offer a 10% discount when rental days is at least 7. "
                       "Keep a 20% margin. Limit the total price increase to 25%."}).json()
    assert gen["provider"] == "mock" and gen["strategy"]["rules"]

    draft = gen["strategy"]
    draft["rules"].append({"id": "CR-YOUNG", "name": "Young driver surcharge", "priority": 50, "stage": "driver",
                           "condition": {"attr": "driver_age", "op": "<", "value": 25}, "action": {"type": "fixed", "value": 15},
                           "customer_message": "Young driver surcharge."})
    draft["constraints"].append({"type": "max_discount", "value": 20})
    assert c.put("/api/v1/domains/car-rental/strategy", headers=ANALYST, json=draft).status_code == 200
    assert c.post("/api/v1/domains/car-rental/validate", headers=ANALYST).json()["status"] == "VALIDATED"

    sim = c.post("/api/v1/domains/car-rental/simulate", headers=ANALYST,
                 json={"as_of": CAR["as_of"], "scenarios": [{"label": "weekly young", "product_id": "SUV", "attributes": CAR["attributes"]}]}).json()
    assert sim["results"][0]["decision"]["status"] == "OK"

    v1 = c.post("/api/v1/domains/car-rental/approve", headers=MANAGER).json()
    assert c.post("/api/v1/domains/car-rental/publish", headers=MANAGER, json={"version_id": v1["id"]}).status_code == 200

    q1 = c.post("/api/v1/pricing/quote", headers=ANALYST, json=CAR).json()
    assert q1["status"] == "OK" and q1["strategy_version_id"] == "car-rental-v1"
    assert any(r["rule_id"] == "CR-YOUNG" for r in q1["applied_rules"])
    assert q1["shadow_validation"]["status"] in ("PASS", "REVIEW_REQUIRED", "FAIL")
    fetched = c.get(f"/api/v1/pricing/decisions/{q1['decision_id']}", headers=ANALYST).json()
    assert fetched["final_price"] == q1["final_price"]

    cust = c.post("/api/v1/pricing/quote", headers=CUSTOMER, json={**CAR, "attributes": CAR["attributes"]}).json()
    assert cust["final_price"] == q1["final_price"] and "Young driver surcharge." in cust["factors"]

    draft["rules"][-1]["action"]["value"] = 25
    c.put("/api/v1/domains/car-rental/strategy", headers=ANALYST, json=draft)
    c.post("/api/v1/domains/car-rental/validate", headers=ANALYST)
    v2 = c.post("/api/v1/domains/car-rental/approve", headers=MANAGER2).json()
    c.post("/api/v1/domains/car-rental/publish", headers=MANAGER2, json={"version_id": v2["id"]})
    q2 = c.post("/api/v1/pricing/quote", headers=ANALYST, json=CAR).json()
    assert q2["strategy_version_id"] == "car-rental-v2"

    assert c.post("/api/v1/domains/car-rental/rollback", headers=MANAGER, json={"version_id": v1["id"]}).json()["status"] == "ACTIVE"
    q3 = c.post("/api/v1/pricing/quote", headers=ANALYST, json={**CAR, "as_of": "2026-10-12"}).json()
    assert q3["strategy_version_id"] == "car-rental-v1"

    trail = [(l["operation"], l["outcome"]) for l in c.get("/api/v1/audit-logs?resource_id=car-rental-v1", headers=ANALYST).json()]
    assert ("APPROVE", "SUCCESS") in trail and ("PUBLISH", "SUCCESS") in trail and ("ROLLBACK", "SUCCESS") in trail


def test_same_engine_prices_all_seeded_domains(client) -> None:
    requests = [
        {"domain_id": "hospitality", "product_id": "STD", "attributes": {"check_in": "2026-10-20"}},
        {"domain_id": "gpu-compute", "product_id": "ACC-B", "attributes": {"commitment": "reserved_12m", "hours": 300}},
        {"domain_id": "edtech", "product_id": "EXAM-PREP", "attributes": {"session_date": "2026-10-12", "session_hour": 19}},
        {"domain_id": "ecommerce", "product_id": "WATCH-S", "attributes": {"quantity": 1, "channel": "marketplace"}},
        {"domain_id": "banking", "product_id": "PERSONAL", "attributes": {"credit_score": 640, "amount": 300000}},
    ]
    results = [client.post("/api/v1/pricing/quote", headers=ANALYST, json={**r, "as_of": "2026-10-11"}).json() for r in requests]
    assert all(r["status"] == "OK" for r in results)
    assert {r["currency"] for r in results} == {"INR", "USD"}
    assert len({json.dumps(r["provenance"]["optimizer"], sort_keys=True) for r in results}) == 1  # one optimizer, one engine


def test_alembic_schema_matches_models(tmp_path) -> None:
    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    run_migrations(url)
    insp = inspect(create_engine(url))
    for table in Base.metadata.sorted_tables:
        cols = {c["name"] for c in insp.get_columns(table.name)}
        assert cols == {c.name for c in table.columns}, table.name
    idx = {i["name"]: i for i in insp.get_indexes("strategy_versions")}
    assert idx["uq_one_active"]["unique"]
