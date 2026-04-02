from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.uncertainty.uncertainty_engine import (
    DistributionalShiftDetector,
    DistributionalShiftWarning,
    MonteCarloDropout,
    combine_uncertainty,
)


def test_monte_carlo_dropout_returns_confident_scalar_for_deterministic_model() -> None:
    class ConstantModel:
        def __call__(self, inputs):
            return 0.25

    result = MonteCarloDropout(n_samples=10).run(ConstantModel(), inputs={"x": 1})
    assert result["mean_prediction"] == 0.25
    assert result["prediction_variance"] == 0.0
    assert result["confidence_score"] == 1.0


def test_distributional_shift_detector_flags_large_mmd(tmp_path) -> None:
    detector = DistributionalShiftDetector(
        reference_path=tmp_path / "reference.pkl",
        alert_path=tmp_path / "shift_alert.json",
        threshold=0.05,
    )
    training = pd.DataFrame(
        [
            {"f1": 0.0, "f2": 0.1},
            {"f1": 0.1, "f2": 0.0},
            {"f1": 0.05, "f2": 0.05},
        ]
    )
    live = pd.DataFrame(
        [
            {"f1": 2.0, "f2": 2.1},
            {"f1": 1.9, "f2": 2.2},
        ]
    )
    detector.fit(training_features=training, as_of_date=date(2026, 3, 1))
    with pytest.warns(DistributionalShiftWarning):
        output = detector.evaluate(live_features=live, as_of_date=date(2026, 3, 30))
    assert output.shift_detected is True
    assert output.risk_level == "high"
    assert detector.recommended_position_multiplier(output.shift_detected) == 0.5
    assert (tmp_path / "shift_alert.json").exists()


def test_distributional_shift_detector_passes_similar_distribution(tmp_path) -> None:
    detector = DistributionalShiftDetector(reference_path=tmp_path / "reference.pkl", threshold=0.50)
    training = pd.DataFrame(
        [
            {"f1": 0.0, "f2": 0.1},
            {"f1": 0.1, "f2": 0.0},
            {"f1": 0.05, "f2": 0.05},
        ]
    )
    live = pd.DataFrame(
        [
            {"f1": 0.02, "f2": 0.08},
            {"f1": 0.06, "f2": 0.04},
        ]
    )
    detector.fit(training_features=training, as_of_date=date(2026, 3, 1))
    output = detector.evaluate(live_features=live, as_of_date=date(2026, 3, 30))
    assert output.shift_detected is False
    assert output.risk_level in {"low", "medium"}


def test_combined_uncertainty_merges_mc_and_shift_outputs(tmp_path) -> None:
    detector = DistributionalShiftDetector(reference_path=tmp_path / "reference.pkl", threshold=1.0)
    detector.fit(training_features=pd.DataFrame([{"f1": 0.0, "f2": 0.0}]), as_of_date=date(2026, 3, 1))
    shift = detector.evaluate(live_features=pd.DataFrame([{"f1": 0.0, "f2": 0.0}]), as_of_date=date(2026, 3, 30))
    combined = combine_uncertainty(
        mc_result={"confidence_score": 0.8, "prediction_variance": 0.1, "mean_prediction": 0.2},
        shift_result=shift,
    )
    assert 0.0 <= combined.confidence_score <= 1.0
    assert combined.mmd_score == shift.mmd_score
