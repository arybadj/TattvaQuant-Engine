"""Compatibility wrapper for the standalone shift detector module."""

from src.uncertainty.uncertainty_engine import (
    DistributionalShiftDetector,
    DistributionalShiftWarning,
)

__all__ = ["DistributionalShiftDetector", "DistributionalShiftWarning"]
