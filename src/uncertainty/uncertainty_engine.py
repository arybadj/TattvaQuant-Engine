"""Prediction uncertainty and distribution-shift detection."""

from __future__ import annotations

import json
import math
import pickle
import warnings
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import mlflow
except ImportError:  # pragma: no cover - optional runtime dependency
    mlflow = None

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


class DistributionalShiftWarning(RuntimeWarning):
    """Raised when live features diverge materially from the training reference distribution."""


@dataclass
class UncertaintyOutput:
    confidence_score: float
    prediction_variance: float
    risk_level: str
    shift_detected: bool
    mmd_score: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class MonteCarloDropout:
    """
    Run forward pass N=100 times with dropout ENABLED at inference.
    Compute: mean prediction, std (prediction_variance).
    confidence_score = 1 - normalized(std)
    """

    def __init__(self, n_samples: int = 100, variance_cap: float = 0.5) -> None:
        self.n_samples = n_samples
        self.variance_cap = variance_cap

    def run(self, model: Any, inputs: Any) -> dict[str, float]:
        if torch is None:
            prediction = self._call_model(model, inputs)
            return {
                "mean_prediction": float(prediction),
                "prediction_variance": 0.0,
                "confidence_score": 1.0,
            }

        was_training = getattr(model, "training", False)
        if hasattr(model, "train"):
            model.train()
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                value = self._call_model(model, inputs)
                predictions.append(float(value))
        if hasattr(model, "train"):
            model.train(was_training)
        mean_prediction = sum(predictions) / max(len(predictions), 1)
        variance = sum((prediction - mean_prediction) ** 2 for prediction in predictions) / max(
            len(predictions), 1
        )
        std = math.sqrt(variance)
        confidence = 1.0 - _clip01(std / max(self.variance_cap, 1e-6))
        return {
            "mean_prediction": float(mean_prediction),
            "prediction_variance": float(std),
            "confidence_score": float(confidence),
        }

    def _call_model(self, model: Any, inputs: Any) -> float:
        output = model(inputs)
        if hasattr(output, "combined_signal"):
            return float(output.combined_signal)
        if hasattr(output, "predicted_return_5d"):
            return float(output.predicted_return_5d)
        if isinstance(output, int | float):
            return float(output)
        if torch is not None and hasattr(output, "detach"):
            detached = output.detach()
            if detached.numel() == 1:
                return float(detached.item())
            return float(detached.reshape(-1)[0].item())
        raise TypeError("MonteCarloDropout model output must be scalar-compatible.")


class DistributionalShiftDetector:
    """
    CRITICAL - this prevents silent model failure.

    At training time: fit a KDE or Gaussian mixture on
    all feature distributions. Save as reference_distribution.

    At inference time: compute MMD (Maximum Mean Discrepancy)
    between live features and reference_distribution.

    If MMD > threshold: raise DistributionalShiftWarning
    Log alert, reduce position sizes by 50%, page on-call.

    Retrain reference_distribution monthly.
    """

    def __init__(
        self,
        reference_path: Path = Path("data/reference/reference_distribution.pkl"),
        alert_path: Path = Path("data/alerts/distribution_shift.json"),
        threshold: float = 0.1,
    ) -> None:
        self.reference_path = reference_path
        self.alert_path = alert_path
        self.threshold = threshold
        self.last_trained_on: date | None = None
        self.reference_distribution: dict[str, Any] | None = None

    def fit(self, training_features: pd.DataFrame, as_of_date: date) -> Path:
        numeric = training_features.select_dtypes(include=["number"]).copy()
        if numeric.empty:
            raise ValueError("DistributionalShiftDetector requires numeric training features.")
        reference = {
            "columns": list(numeric.columns),
            "samples": numeric.to_numpy(),
            "means": numeric.mean().to_dict(),
            "stds": numeric.std(ddof=0).replace(0, 1.0).to_dict(),
            "as_of_date": as_of_date.isoformat(),
            "density_model": self._fit_density_model(numeric),
        }
        self.reference_distribution = reference
        self.last_trained_on = as_of_date
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        self.reference_path.write_bytes(pickle.dumps(reference))
        self._log_mlflow(reference)
        return self.reference_path

    def maybe_retrain(self, training_features: pd.DataFrame, as_of_date: date) -> Path | None:
        if self.last_trained_on is None or (as_of_date - self.last_trained_on) >= timedelta(
            days=30
        ):
            return self.fit(training_features=training_features, as_of_date=as_of_date)
        return None

    def evaluate(
        self, live_features: pd.DataFrame, as_of_date: date | None = None
    ) -> UncertaintyOutput:
        reference = self.reference_distribution or self._load_reference()
        if reference is None:
            raise ValueError("Reference distribution is not fitted.")
        numeric = live_features.loc[:, reference["columns"]].astype(float)
        mmd_score = self._compute_mmd(reference["samples"], numeric.to_numpy())
        shift_detected = bool(mmd_score > self.threshold)
        if shift_detected:
            self._log_alert(mmd_score=mmd_score, as_of_date=as_of_date)
            warnings.warn(
                "Distributional shift detected. Reduce position sizes by 50% and page on-call.",
                DistributionalShiftWarning,
                stacklevel=2,
            )
        confidence = max(0.0, min(1.0, 1.0 - (mmd_score / max(self.threshold * 2.0, 1e-6))))
        return UncertaintyOutput(
            confidence_score=float(confidence),
            prediction_variance=float(mmd_score),
            risk_level=self._risk_level(confidence, shift_detected),
            shift_detected=shift_detected,
            mmd_score=float(mmd_score),
        )

    def recommended_position_multiplier(self, shift_detected: bool) -> float:
        return 0.5 if shift_detected else 1.0

    def _load_reference(self) -> dict[str, Any] | None:
        if not self.reference_path.exists():
            return None
        reference = pickle.loads(self.reference_path.read_bytes())
        self.reference_distribution = reference
        if reference.get("as_of_date"):
            self.last_trained_on = date.fromisoformat(reference["as_of_date"])
        return reference

    def _fit_density_model(self, numeric: pd.DataFrame) -> dict[str, Any]:
        try:
            from sklearn.neighbors import KernelDensity
        except ImportError:
            return {"type": "summary_stats"}
        try:
            bandwidth = max(float(numeric.std(ddof=0).mean()), 0.05)
            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
            kde.fit(numeric.to_numpy())
            return {
                "type": "kde",
                "bandwidth": bandwidth,
                "score_samples_mean": float(kde.score_samples(numeric.to_numpy()).mean()),
            }
        except Exception:
            return {"type": "summary_stats"}

    def _compute_mmd(self, reference_samples: Any, live_samples: Any, gamma: float = 1.0) -> float:
        x = self._to_rows(reference_samples)
        y = self._to_rows(live_samples)
        if not x or not y:
            return 0.0
        k_xx = self._kernel_mean(x, x, gamma)
        k_yy = self._kernel_mean(y, y, gamma)
        k_xy = self._kernel_mean(x, y, gamma)
        return float(max(k_xx + k_yy - (2.0 * k_xy), 0.0))

    def _kernel_mean(
        self, left: list[list[float]], right: list[list[float]], gamma: float
    ) -> float:
        total = 0.0
        count = 0
        for l_row in left:
            for r_row in right:
                squared_distance = sum(
                    (l_value - r_value) ** 2 for l_value, r_value in zip(l_row, r_row, strict=False)
                )
                total += math.exp(-gamma * squared_distance)
                count += 1
        return total / max(count, 1)

    def _to_rows(self, values: Any) -> list[list[float]]:
        if hasattr(values, "tolist"):
            values = values.tolist()
        rows = []
        for row in values:
            if isinstance(row, list):
                rows.append([float(item) for item in row])
            else:
                rows.append([float(row)])
        return rows

    def _risk_level(self, confidence: float, shift_detected: bool) -> str:
        if shift_detected or confidence < 0.4:
            return "high"
        if confidence < 0.7:
            return "medium"
        return "low"

    def _log_alert(self, mmd_score: float, as_of_date: date | None) -> None:
        payload = {
            "event": "distributional_shift_detected",
            "as_of_date": as_of_date.isoformat() if as_of_date else None,
            "mmd_score": mmd_score,
            "action": "reduce_position_sizes_by_50_percent",
            "page_on_call": True,
        }
        self.alert_path.parent.mkdir(parents=True, exist_ok=True)
        self.alert_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _log_mlflow(self, reference: dict[str, Any]) -> None:
        if mlflow is None:
            return
        try:
            with mlflow.start_run(
                run_name=f"reference-distribution-{reference['as_of_date']}", nested=True
            ):
                mlflow.log_param("reference_as_of_date", reference["as_of_date"])
                mlflow.log_param("reference_columns", json.dumps(reference["columns"]))
                mlflow.log_artifact(str(self.reference_path))
        except Exception:
            return


class UncertaintyEngine:
    """Combine Monte Carlo dropout and shift detection into one uncertainty output."""

    def __init__(
        self,
        dropout: MonteCarloDropout | None = None,
        shift_detector: DistributionalShiftDetector | None = None,
    ) -> None:
        self.dropout = dropout or MonteCarloDropout()
        self.shift_detector = shift_detector or DistributionalShiftDetector()

    def fit_reference(self, training_features: pd.DataFrame, as_of_date: date) -> Path:
        return self.shift_detector.fit(training_features=training_features, as_of_date=as_of_date)

    def evaluate(
        self,
        model: Any,
        model_inputs: Any,
        live_features: pd.DataFrame,
        as_of_date: date,
        training_features: pd.DataFrame | None = None,
    ) -> UncertaintyOutput:
        if training_features is not None:
            self.shift_detector.maybe_retrain(
                training_features=training_features, as_of_date=as_of_date
            )
        mc_result = self.dropout.run(model=model, inputs=model_inputs)
        shift_result = self.shift_detector.evaluate(
            live_features=live_features, as_of_date=as_of_date
        )
        return combine_uncertainty(mc_result=mc_result, shift_result=shift_result)


def combine_uncertainty(
    mc_result: dict[str, float],
    shift_result: UncertaintyOutput,
) -> UncertaintyOutput:
    confidence = max(
        0.0, min(1.0, (mc_result["confidence_score"] * 0.6) + (shift_result.confidence_score * 0.4))
    )
    prediction_variance = float(
        (mc_result["prediction_variance"] * 0.7) + (shift_result.mmd_score * 0.3)
    )
    shift_detected = bool(shift_result.shift_detected)
    if shift_detected or confidence < 0.4:
        risk_level = "high"
    elif confidence < 0.7:
        risk_level = "medium"
    else:
        risk_level = "low"
    return UncertaintyOutput(
        confidence_score=confidence,
        prediction_variance=prediction_variance,
        risk_level=risk_level,
        shift_detected=shift_detected,
        mmd_score=shift_result.mmd_score,
    )
