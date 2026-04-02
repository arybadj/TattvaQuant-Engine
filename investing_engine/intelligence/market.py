from __future__ import annotations

from dataclasses import dataclass

from investing_engine.models import FeatureVector, SignalPayload, SignalSource


@dataclass
class MarketIntelligenceModel:
    def score(self, feature: FeatureVector) -> SignalPayload:
        raw_score = (feature.momentum_5d * 0.6) + (feature.momentum_20d * 0.4) - (feature.realized_volatility * 0.2)
        bounded_score = max(min(raw_score * 8.0, 1.0), -1.0)
        confidence = max(0.05, min(0.95, 1.0 - feature.realized_volatility * 2.0))
        return SignalPayload(
            symbol=feature.symbol,
            as_of_date=feature.as_of_date,
            source=SignalSource.market,
            score=bounded_score,
            confidence=confidence,
            rationale="Price trend and volatility encoder.",
            metadata={"momentum_5d": feature.momentum_5d, "momentum_20d": feature.momentum_20d},
        )
