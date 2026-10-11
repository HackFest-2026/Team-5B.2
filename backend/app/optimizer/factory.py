"""Choose the PriceOptimizer from settings: the real ML model when available, else the simulator."""

from __future__ import annotations

import logging
from typing import Optional

from app.config import Settings

from .base import PriceOptimizer
from .simulated import SimulatedPriceOptimizer

log = logging.getLogger(__name__)


class OptimizerStatus:
    def __init__(self) -> None:
        self.fallback_reason: Optional[str] = None


def build_optimizer(settings: Settings, status: Optional[OptimizerStatus] = None) -> PriceOptimizer:
    provider = settings.optimizer_provider
    if provider == "simulated":
        return SimulatedPriceOptimizer()
    if provider == "http":
        from .http import HttpPriceOptimizer

        if not settings.optimizer_url:
            raise RuntimeError("OPTIMIZER_PROVIDER=http needs OPTIMIZER_URL")
        return HttpPriceOptimizer(settings.optimizer_url)
    try:
        from .ml import MLPriceOptimizer

        return MLPriceOptimizer(settings.ml_artifacts_dir, settings.optimizer_model, settings.ml_artifact_sha256)
    except Exception as e:  # noqa: BLE001 - any load failure falls back (auto) or is fatal (ml)
        if provider == "ml":
            raise
        log.warning("ML optimizer unavailable (%s); using the simulated optimizer", e)
        if status is not None:
            status.fallback_reason = str(e)
        return SimulatedPriceOptimizer()
