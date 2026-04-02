from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data.timegate import LookaheadError
from src.models.fundamental_model import CompanyQualityModel


def _training_frame() -> pd.DataFrame:
    model = CompanyQualityModel(random_state=7)
    return model.build_training_frame(
        symbols=["AAPL", "MSFT", "NVDA"],
        start_date=date(2020, 1, 1),
        end_date=date(2025, 12, 1),
        price_source="synthetic",
    )


def test_walk_forward_produces_at_least_four_folds() -> None:
    model = CompanyQualityModel(random_state=7)
    result = model.walk_forward_validate(_training_frame())
    assert len(result.folds) >= 4
    assert len(result.sharpe_series) == len(result.folds)


def test_fundamental_signal_fields_are_all_between_zero_and_one() -> None:
    training = _training_frame()
    model = CompanyQualityModel(random_state=7).fit(training.iloc[:-6].copy())
    feature_row = training.iloc[-1].drop(labels=["forward_30d_return", "direction_label"]).to_dict()
    signal = model.predict_signal(feature_row, industry_context="steady software demand")
    assert 0.0 <= signal.fundamental_score <= 1.0
    assert 0.0 <= signal.valuation_score <= 1.0
    assert 0.0 <= signal.financial_health <= 1.0


def test_model_rejects_features_with_nan_values() -> None:
    training = _training_frame()
    training.loc[0, "ebitda_margin"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        CompanyQualityModel().fit(training)


def test_no_future_data_leaks_into_training_folds() -> None:
    training = _training_frame()
    result = CompanyQualityModel(random_state=7).walk_forward_validate(training)
    assert all(fold.no_lookahead for fold in result.folds)
    assert all(fold.max_train_available_at <= fold.train_end for fold in result.folds)

    leaked = training.copy()
    leaked.loc[0, "available_at"] = pd.Timestamp(leaked.loc[0, "as_of_date"]) + pd.Timedelta(days=2)
    with pytest.raises(LookaheadError):
        CompanyQualityModel(random_state=7).walk_forward_validate(leaked)
