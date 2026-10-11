"""Adapter for the existing ML_Updated model (GradientBoosting, or the LogisticRegression baseline).

Artifacts are loaded read-only, once, from the local ML_ARTIFACTS_DIR. Pickle files can execute
code when loaded, so: only local artifacts are ever loaded (never uploads), the scikit-learn
version must match the one the model was trained with, and an optional SHA-256 pin is enforced.
Nothing here writes to ml_updated/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Literal, Optional

from app.schemas.optimizer import CandidateEvaluation, ModelInfo, OptimizerRequest, OptimizerResponse

from .base import finish
from .features import build_features

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pickled_sklearn_version(path: Path) -> Optional[str]:
    # Read the version string from the pickle bytes without unpickling.
    m = re.search(rb"_sklearn_version.{0,8}?(\d+\.\d+\.\d+)", path.read_bytes(), re.S)
    return m.group(1).decode() if m else None


class MLPriceOptimizer:
    def __init__(self, artifacts_dir: str | Path, model: Literal["gbt", "lr"] = "gbt", expected_sha256: str = ""):
        import numpy as np  # noqa: F401  (fail early if the ML stack is missing)
        import sklearn

        self.dir = Path(artifacts_dir)
        self.kind = model
        model_path = self.dir / ("model_gbt.pkl" if model == "gbt" else "model_lr.pkl")
        if not model_path.is_file():
            raise ModelLoadError(f"model artifact not found: {model_path}")
        digest = _sha256(model_path)
        if expected_sha256 and digest != expected_sha256.lower():
            raise ModelLoadError(f"artifact hash mismatch for {model_path.name}: {digest[:12]}… != pinned value")
        trained_with = _pickled_sklearn_version(model_path)
        if trained_with and trained_with != sklearn.__version__:
            raise ModelLoadError(
                f"{model_path.name} was trained with scikit-learn {trained_with}, but {sklearn.__version__} is installed; "
                f"install scikit-learn=={trained_with}"
            )
        with model_path.open("rb") as f:
            self.model = pickle.load(f)  # noqa: S301 - local, version-checked, optionally hash-pinned artifact
        self.scaler = None
        if model == "lr":
            with (self.dir / "scaler.pkl").open("rb") as f:
                self.scaler = pickle.load(f)  # noqa: S301
        meta_path = self.dir / "training_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        self.metadata = meta
        names = meta.get("features", {}).get("names")
        from app.schemas.optimizer import ML_FEATURE_COLUMNS

        if names and tuple(names) != ML_FEATURE_COLUMNS:
            raise ModelLoadError("training_metadata feature order differs from the adapter's feature contract")
        if getattr(self.model, "n_features_in_", 14) != 14:
            raise ModelLoadError("model does not expect 14 features")
        self.info = ModelInfo(
            name=f"ml_updated-{model}",
            version=f"{meta.get('pipeline', 'unknown')}/{type(self.model).__name__}/sklearn-{trained_with or sklearn.__version__}",
            kind="ml",
            training_data="synthetic",
            artifact_sha256=digest,
        )
        log.info("Loaded ML optimizer %s (%s)", self.info.name, self.info.version)

    def predict(self, matrix: list[list[float]]) -> list[float]:
        import numpy as np

        X = np.asarray(matrix, dtype=float)
        if self.scaler is not None:
            import pandas as pd

            from app.schemas.optimizer import ML_FEATURE_COLUMNS

            X = self.scaler.transform(pd.DataFrame(X, columns=list(ML_FEATURE_COLUMNS)))  # fitted with column names
        return [float(p) for p in self.model.predict_proba(X)[:, 1]]

    def evaluate(self, req: OptimizerRequest, overrides: dict[str, float] | None = None) -> OptimizerResponse:
        batch = build_features(req, overrides)
        probs = self.predict(batch.matrix())
        by_price = {c.price: c for c in req.candidates}
        evals = []
        for price, row, p in zip(batch.prices, batch.rows, probs):
            p = round(min(max(p, 0.0), 1.0), 6)
            evals.append(CandidateEvaluation(
                price=price, features=row, p_buy=p,
                expected_revenue=round(p * price, 6), expected_profit=round(p * (price - req.unit_cost), 6),
                eligible=by_price[price].eligible, selection_status="NOT_SELECTED",
                explanation=(f"ML ({self.kind.upper()}, trained on synthetic data): P(buy) {p:.0%} at "
                             f"{row.price_ratio:.2f}× base, {row.price_vs_competitor:.2f}× competitor"),
            ))
        return finish(evals, req, self.info, batch.warnings)
