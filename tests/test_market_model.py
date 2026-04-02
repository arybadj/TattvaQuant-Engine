from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import torch

from src.data.timegate import LookaheadError
from src.models.market_model import LSTMMarketModel, MARKET_FEATURE_COLUMNS, MarketModelTrainer, MarketTrainingConfig


def _trainer() -> MarketModelTrainer:
    return MarketModelTrainer(config=MarketTrainingConfig(epochs=1, batch_size=64, train_months=24, test_months=1, step_months=1))


def _feature_frame() -> pd.DataFrame:
    return _trainer().build_feature_frame(
        symbols=["AAPL"],
        start_date=date(2022, 1, 3),
        end_date=date(2025, 3, 31),
        price_source="synthetic",
    )


def test_market_signal_fields_are_all_between_zero_and_one() -> None:
    trainer = _trainer()
    bundle = trainer.build_sequences(_feature_frame())
    trainer.fit(bundle.sequences[:96], bundle.targets[:96])
    signal = trainer.predict_signal(bundle.sequences[96])
    assert 0.0 <= signal.trend_signal <= 1.0
    assert 0.0 <= signal.momentum_score <= 1.0
    assert 0.0 <= signal.volatility_risk <= 1.0


def test_model_raises_error_on_wrong_sequence_length() -> None:
    model = LSTMMarketModel(n_features=len(MARKET_FEATURE_COLUMNS))
    with pytest.raises(ValueError, match="Expected sequence length 60"):
        model(torch.zeros(1, 59, len(MARKET_FEATURE_COLUMNS)))


def test_walk_forward_produces_at_least_four_folds(tmp_path) -> None:
    result = _trainer().walk_forward_validate(_feature_frame(), artifact_root=tmp_path)
    assert len(result.folds) >= 4
    assert len(result.directional_accuracy_series) == len(result.folds)


def test_no_future_data_leaks_into_any_training_fold(tmp_path) -> None:
    feature_frame = _feature_frame()
    result = _trainer().walk_forward_validate(feature_frame, artifact_root=tmp_path)
    assert all(fold.no_lookahead for fold in result.folds)
    assert all(fold.max_train_available_at <= fold.train_end for fold in result.folds)

    leaked = feature_frame.copy()
    leaked.loc[0, "available_at"] = pd.Timestamp(leaked.loc[0, "as_of_date"]) + pd.Timedelta(days=2)
    with pytest.raises(LookaheadError):
        _trainer().walk_forward_validate(leaked, artifact_root=tmp_path / "leaked")


def test_directional_accuracy_across_folds_is_stored_as_float_between_zero_and_one(tmp_path) -> None:
    result = _trainer().walk_forward_validate(_feature_frame(), artifact_root=tmp_path)
    assert isinstance(result.mean_directional_accuracy, float)
    assert 0.0 <= result.mean_directional_accuracy <= 1.0
    assert all(isinstance(fold.directional_accuracy, float) for fold in result.folds)
    assert all(0.0 <= fold.directional_accuracy <= 1.0 for fold in result.folds)
