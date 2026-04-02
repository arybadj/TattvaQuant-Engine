"""Compatibility wrapper for the fundamental model module name requested in the project structure."""

from src.models.fundamental_model import (
    CompanyQualityModel,
    CompanyQualitySignal,
    FundamentalModelEnsemble,
    FundamentalSignal,
    FutureIndustryModel,
    FutureIndustrySignal,
    IndustryHistoryModel,
    IndustryHistorySignal,
    WalkForwardBacktestResult,
    WalkForwardFoldResult,
)

__all__ = [
    "CompanyQualityModel",
    "CompanyQualitySignal",
    "FundamentalModelEnsemble",
    "FundamentalSignal",
    "FutureIndustryModel",
    "FutureIndustrySignal",
    "IndustryHistoryModel",
    "IndustryHistorySignal",
    "WalkForwardBacktestResult",
    "WalkForwardFoldResult",
]
