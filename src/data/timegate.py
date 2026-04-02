"""Point-in-time access control for all data reads."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Iterable

import pandas as pd
from pydantic import BaseModel, Field


class LookaheadError(ValueError):
    """Raised when a dataset exposes information before it was available."""


class PointInTimeRecord(BaseModel):
    symbol: str
    data_as_of: date
    available_at: datetime
    data_type: str
    payload: dict = Field(default_factory=dict)


class TimeGate:
    """
    Single entry point for ALL data reads.
    Raises LookaheadError if any record's available_at
    is after the requested as_of_date.
    Must be used by every feature pipeline - no exceptions.
    """

    def __init__(self, records: Iterable[PointInTimeRecord] | None = None) -> None:
        self._records: list[PointInTimeRecord] = list(records or [])

    def add(self, record: PointInTimeRecord) -> None:
        self._records.append(record)

    def extend(self, records: Iterable[PointInTimeRecord]) -> None:
        self._records.extend(records)

    def get(self, symbol: str, as_of_date: date, data_type: str) -> list[PointInTimeRecord]:
        cutoff = datetime.combine(as_of_date, time.max)
        candidate_records = [
            record
            for record in self._records
            if record.symbol == symbol and record.data_type == data_type and record.data_as_of <= as_of_date
        ]

        violating_records = [record for record in candidate_records if record.available_at > cutoff]
        if violating_records:
            violation = violating_records[0]
            raise LookaheadError(
                "Lookahead detected for "
                f"{symbol}/{data_type}: available_at={violation.available_at.isoformat()} "
                f"exceeds as_of_date={as_of_date.isoformat()}"
            )

        return [record for record in candidate_records if record.available_at <= cutoff]

    def validate_no_lookahead(self, dataset: pd.DataFrame) -> bool:
        if "available_at" not in dataset.columns:
            raise ValueError("Dataset must contain an 'available_at' column.")

        if "as_of_date" in dataset.columns:
            as_of_series = pd.to_datetime(dataset["as_of_date"])
        elif "requested_as_of" in dataset.columns:
            as_of_series = pd.to_datetime(dataset["requested_as_of"])
        elif "as_of_date" in dataset.attrs:
            as_of_series = pd.Series(pd.to_datetime(dataset.attrs["as_of_date"]), index=dataset.index)
        else:
            raise ValueError("Dataset must provide 'as_of_date' as a column or DataFrame attribute.")

        available_series = pd.to_datetime(dataset["available_at"])
        mask = available_series > as_of_series.apply(lambda value: value.replace(hour=23, minute=59, second=59))
        if bool(mask.any()):
            row_index = dataset.index[mask][0]
            raise LookaheadError(
                "Training dataset contains future data at row "
                f"{row_index}: available_at={available_series.loc[row_index].isoformat()} "
                f"exceeds as_of_date={as_of_series.loc[row_index].date().isoformat()}"
            )
        return True
