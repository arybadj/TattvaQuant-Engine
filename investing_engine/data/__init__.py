"""Data access layer with strict point-in-time controls."""

from investing_engine.data.ingestion import PointInTimeDataPipeline
from investing_engine.data.timegate import TimeGate

__all__ = ["PointInTimeDataPipeline", "TimeGate"]
