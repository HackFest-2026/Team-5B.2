"""Optional remote optimizer: POSTs an OptimizerRequest and validates the response contract."""

from __future__ import annotations

import httpx

from app.schemas.optimizer import ModelInfo, OptimizerRequest, OptimizerResponse


class HttpPriceOptimizer:
    def __init__(self, url: str, timeout_s: float = 10.0, client: httpx.Client | None = None):
        self.url = url
        self.client = client or httpx.Client(timeout=timeout_s)
        self.info = ModelInfo(name="http-optimizer", version=url, kind="ml")

    def evaluate(self, req: OptimizerRequest, overrides: dict[str, float] | None = None) -> OptimizerResponse:
        payload = req.model_dump(mode="json")
        if overrides:
            payload["signals"] = {**payload["signals"], **overrides}
        r = self.client.post(self.url, json=payload)
        r.raise_for_status()
        resp = OptimizerResponse.model_validate(r.json())  # never trust an unvalidated payload
        self.info = resp.model
        return resp
