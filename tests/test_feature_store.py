from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from src.data.timegate import PointInTimeRecord, TimeGate
from src.features.feature_store import FeatureError, FeatureStore, TimeSeriesFeatures


def _build_gate() -> TimeGate:
    records: list[PointInTimeRecord] = []
    start = date(2026, 1, 1)
    for offset in range(60):
        current_day = start + timedelta(days=offset)
        close = 100.0 + offset
        records.append(
            PointInTimeRecord(
                symbol="INFY",
                data_as_of=current_day,
                available_at=datetime.combine(current_day, datetime.min.time()).replace(hour=16),
                data_type="price",
                payload={"close": close, "volume": 1_000_000 + (offset * 1000)},
            )
        )
        records.append(
            PointInTimeRecord(
                symbol="INFY",
                data_as_of=current_day,
                available_at=datetime.combine(current_day, datetime.min.time()).replace(hour=15, minute=31),
                data_type="order_book",
                payload={"best_bid": close - 0.1, "best_ask": close + 0.1},
            )
        )
        records.append(
            PointInTimeRecord(
                symbol="INFY",
                data_as_of=current_day,
                available_at=datetime.combine(current_day, datetime.min.time()).replace(hour=17),
                data_type="options",
                payload={"unusual_activity_flag": float(offset % 2)},
            )
        )

    for current_day, revenue, quarter in zip(
        [date(2024, 12, 28), date(2025, 3, 28), date(2025, 6, 28), date(2025, 9, 28), date(2025, 12, 28)],
        [950.0, 1000.0, 1050.0, 1100.0, 1200.0],
        [0, 1, 2, 3, 4],
        strict=True,
    ):
        records.append(
            PointInTimeRecord(
                symbol="INFY",
                data_as_of=current_day,
                available_at=datetime.combine(current_day + timedelta(days=20), datetime.min.time()).replace(hour=9),
                data_type="fundamental",
                payload={
                    "revenue": revenue,
                    "ebitda": revenue * 0.25,
                    "market_cap": 10_000.0,
                    "free_cash_flow": 600.0 + quarter * 10,
                    "roe": 0.18 + (quarter * 0.005),
                    "debt_to_equity": 0.40 - (quarter * 0.01),
                    "eps_actual": 10.0 + quarter,
                    "eps_estimate": 9.5 + quarter,
                    "net_income": 200.0 + quarter * 5,
                    "operating_cash_flow": 220.0 + quarter * 5,
                    "roa": 0.08 + (quarter * 0.005),
                    "long_term_debt_ratio": 0.30 - (quarter * 0.01),
                    "current_ratio": 1.5 + (quarter * 0.05),
                    "shares_outstanding": 100.0,
                    "gross_margin": 0.45 + (quarter * 0.01),
                    "asset_turnover": 0.7 + (quarter * 0.02),
                },
            )
        )

    for offset in range(10):
        current_day = start + timedelta(days=offset * 5)
        records.append(
            PointInTimeRecord(
                symbol="INFY",
                data_as_of=current_day,
                available_at=datetime.combine(current_day, datetime.min.time()).replace(hour=8),
                data_type="news",
                payload={
                    "finbert_sentiment_score": 0.2,
                    "topic_vector": [0.1, 0.2, 0.3],
                    "risk_flags": ["regulatory"] if offset % 3 == 0 else [],
                },
            )
        )
        records.append(
            PointInTimeRecord(
                symbol="INFY",
                data_as_of=current_day,
                available_at=datetime.combine(current_day, datetime.min.time()).replace(hour=18),
                data_type="transcript",
                payload={"tone_score": 0.5 + (offset * 0.01)},
            )
        )

    for offset in range(3):
        current_day = date(2026, 1, 31) + timedelta(days=offset * 30)
        records.append(
            PointInTimeRecord(
                symbol="GLOBAL",
                data_as_of=current_day,
                available_at=datetime.combine(current_day, datetime.min.time()).replace(hour=7),
                data_type="macro",
                payload={
                    "policy_rate": 6.0 + offset * 0.25,
                    "sector_relative_strength": 0.04 + offset * 0.01,
                    "ff5_market_beta": 1.1,
                    "ff5_size_beta": -0.2,
                    "ff5_value_beta": 0.3,
                    "ff5_profitability_beta": 0.15,
                    "ff5_investment_beta": -0.1,
                    "currency_momentum_usdinr": 0.02,
                    "vix": 18.0 + offset,
                },
            )
        )

    return TimeGate(records=records)


def test_materialize_writes_registry_and_offline_snapshot(tmp_path) -> None:
    gate = _build_gate()
    store = FeatureStore(
        offline_root=tmp_path / "features",
        registry_path=tmp_path / "feature_registry.yaml",
        redis_url="",
    )

    snapshot = store.materialize(symbol="INFY", as_of_date=date(2026, 3, 1), gate=gate)
    assert snapshot.symbol == "INFY"
    assert (tmp_path / "feature_registry.yaml").exists()
    assert (tmp_path / "features" / "INFY").exists()
    assert (tmp_path / "features" / "INFY" / "2026-03-01.parquet").exists() or (
        tmp_path / "features" / "INFY" / "2026-03-01.json"
    ).exists()


def test_time_series_schema_rejects_nan() -> None:
    with pytest.raises(FeatureError):
        TimeSeriesFeatures(
            symbol="INFY",
            as_of_date=date(2026, 3, 1),
            log_return_1d=float("nan"),
            log_return_5d=0.1,
            log_return_20d=0.2,
            rolling_mean_10d=1.0,
            rolling_mean_20d=1.0,
            rolling_mean_50d=1.0,
            rolling_std_10d=1.0,
            rolling_std_20d=1.0,
            rolling_std_50d=1.0,
            rsi_14=50.0,
            macd_line=0.1,
            macd_signal=0.1,
            macd_histogram=0.0,
            bollinger_band_width=0.2,
            realized_volatility_20d=0.3,
            volume_zscore_30d=0.0,
            bid_ask_spread=0.2,
            adv_30d=1_000_000.0,
            unusual_options_activity=0.0,
        )


def test_fit_normalizer_requires_clean_training_window(tmp_path) -> None:
    store = FeatureStore(offline_root=tmp_path / "features", registry_path=tmp_path / "registry.yaml", redis_url="")
    dataset = pd.DataFrame(
        [
            {
                "as_of_date": "2026-03-01",
                "available_at": "2026-03-02T09:00:00",
                "log_return_1d": 0.1,
            }
        ]
    )

    with pytest.raises(Exception):
        store.fit_normalizer(dataset, ["log_return_1d"], version="v1")
