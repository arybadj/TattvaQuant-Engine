from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RegimeLabel(str, Enum):
    bull = "bull"
    bear = "bear"
    sideways = "sideways"
    high_volatility = "high_volatility"


class SignalSource(str, Enum):
    market = "market"
    text = "text"
    tabular = "tabular"
    fused = "fused"


class MarketBar(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class NewsItem(BaseModel):
    symbol: str
    timestamp: datetime
    headline: str
    sentiment_hint: float = 0.0
    source: str = "synthetic"


class FeatureVector(BaseModel):
    symbol: str
    as_of_date: date
    momentum_5d: float
    momentum_20d: float
    realized_volatility: float
    average_volume: float
    text_sentiment: float
    feature_quality: float
    extras: dict[str, float] = Field(default_factory=dict)


class SignalPayload(BaseModel):
    symbol: str
    as_of_date: date
    source: SignalSource
    score: float
    confidence: float
    rationale: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegimeState(BaseModel):
    as_of_date: date
    label: RegimeLabel
    volatility_level: float
    trend_strength: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class UncertaintyEstimate(BaseModel):
    symbol: str
    as_of_date: date
    model_dispersion: float
    data_quality_risk: float
    regime_risk: float
    total_uncertainty: float
    confidence_interval: tuple[float, float]


class DecisionContext(BaseModel):
    symbol: str
    as_of_date: date
    fused_signal: float
    signal_confidence: float
    total_uncertainty: float
    current_weight: float
    target_weight: float
    expected_return: float
    estimated_cost: float
    risk_penalty: float
    reward_estimate: float
    rationale: str


class OrderRequest(BaseModel):
    symbol: str
    as_of_date: date
    current_weight: float
    target_weight: float
    turnover: float
    estimated_cost: float
    trade_required: bool


class FeedbackRecord(BaseModel):
    symbol: str
    as_of_date: date
    realized_return: float
    realized_cost: float
    reward: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    as_of_date: date
    regime: RegimeState
    features: list[FeatureVector]
    signals: list[SignalPayload]
    uncertainties: list[UncertaintyEstimate]
    decisions: list[DecisionContext]
    orders: list[OrderRequest]
    feedback: list[FeedbackRecord]

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
