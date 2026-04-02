"""Compatibility wrapper for the alternate fundamental-model module name."""

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
