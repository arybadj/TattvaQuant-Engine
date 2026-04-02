from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TimeGate:
    as_of_date: date

    @property
    def cutoff(self) -> datetime:
        return datetime.combine(self.as_of_date, time.max)

    def filter_records(self, records: Iterable[T], timestamp_getter: Callable[[T], datetime]) -> list[T]:
        return [record for record in records if timestamp_getter(record) <= self.cutoff]

    def assert_not_future(self, timestamp: datetime) -> None:
        if timestamp > self.cutoff:
            raise ValueError(f"Point-in-time violation: {timestamp.isoformat()} exceeds {self.cutoff.isoformat()}")
