from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from src.data.timegate import LookaheadError, PointInTimeRecord, TimeGate


def test_get_raises_lookahead_for_future_available_record() -> None:
    gate = TimeGate(
        records=[
            PointInTimeRecord(
                symbol="INFY",
                data_as_of=date(2025, 12, 31),
                available_at=datetime(2026, 2, 14, 9, 0, 0),
                data_type="fundamental",
                payload={"revenue": 1000},
            )
        ]
    )

    with pytest.raises(LookaheadError):
        gate.get(symbol="INFY", as_of_date=date(2026, 1, 31), data_type="fundamental")


def test_get_returns_records_when_all_data_was_available() -> None:
    gate = TimeGate(
        records=[
            PointInTimeRecord(
                symbol="RELIANCE",
                data_as_of=date(2026, 3, 28),
                available_at=datetime(2026, 3, 28, 15, 30, 0),
                data_type="price",
                payload={"close": 1450.25},
            )
        ]
    )

    records = gate.get(symbol="RELIANCE", as_of_date=date(2026, 3, 30), data_type="price")
    assert len(records) == 1
    assert records[0].payload["close"] == 1450.25


def test_validate_no_lookahead_raises_for_future_rows() -> None:
    gate = TimeGate()
    dataset = pd.DataFrame(
        [
            {
                "symbol": "TCS",
                "as_of_date": "2026-03-30",
                "available_at": "2026-03-31T09:15:00",
                "feature_value": 1.23,
            }
        ]
    )

    with pytest.raises(LookaheadError):
        gate.validate_no_lookahead(dataset)


def test_validate_no_lookahead_passes_for_clean_dataset() -> None:
    gate = TimeGate()
    dataset = pd.DataFrame(
        [
            {
                "symbol": "HDFCBANK",
                "as_of_date": "2026-03-30",
                "available_at": "2026-03-30T08:00:00",
                "feature_value": 0.87,
            }
        ]
    )

    assert gate.validate_no_lookahead(dataset) is True
