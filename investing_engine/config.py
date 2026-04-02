from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field


class RewardWeights(BaseModel):
    return_weight: float = 1.0
    volatility_penalty: float = 0.20
    drawdown_penalty: float = 0.15
    cvar_penalty: float = 0.15
    turnover_penalty: float = 0.05
    diversification_bonus: float = 0.03


class EngineSettings(BaseModel):
    symbols: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")
    lookback_days: int = 90
    as_of_date: date = Field(default_factory=date.today)
    feature_path: Path = Path("data/features")
    feedback_path: Path = Path("data/feedback")
    news_source: str = "synthetic"
    market_source: str = "synthetic"
    postgres_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/investing"
    redis_url: str = "redis://localhost:6379/0"
    kafka_bootstrap: str = "localhost:9092"
    model_device: str = "cpu"
    risk_free_rate: float = 0.01
    max_gross_leverage: float = 1.0
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def load_settings() -> EngineSettings:
    raw_symbols = _env("ENGINE_SYMBOLS", "AAPL,MSFT,NVDA")
    return EngineSettings(
        symbols=tuple(symbol.strip().upper() for symbol in raw_symbols.split(",") if symbol.strip()),
        lookback_days=int(_env("ENGINE_LOOKBACK_DAYS", "90")),
        as_of_date=date.fromisoformat(_env("ENGINE_AS_OF_DATE", date.today().isoformat())),
        feature_path=Path(_env("ENGINE_FEATURE_PATH", "data/features")),
        feedback_path=Path(_env("ENGINE_FEEDBACK_PATH", "data/feedback")),
        news_source=_env("ENGINE_NEWS_SOURCE", "synthetic"),
        market_source=_env("ENGINE_MARKET_SOURCE", "synthetic"),
        postgres_url=_env("ENGINE_POSTGRES_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/investing"),
        redis_url=_env("ENGINE_REDIS_URL", "redis://localhost:6379/0"),
        kafka_bootstrap=_env("ENGINE_KAFKA_BOOTSTRAP", "localhost:9092"),
        model_device=_env("ENGINE_MODEL_DEVICE", "cpu"),
    )
