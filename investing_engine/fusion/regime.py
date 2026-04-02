from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from investing_engine.models import FeatureVector, RegimeLabel, RegimeState


@dataclass
class RegimeDetector:
    def detect(self, as_of_date: date, features: list[FeatureVector]) -> RegimeState:
        avg_trend = sum(feature.momentum_20d for feature in features) / max(len(features), 1)
        avg_volatility = sum(feature.realized_volatility for feature in features) / max(len(features), 1)
        if avg_volatility > 0.045:
            label = RegimeLabel.high_volatility
        elif avg_trend > 0.03:
            label = RegimeLabel.bull
        elif avg_trend < -0.03:
            label = RegimeLabel.bear
        else:
            label = RegimeLabel.sideways
        return RegimeState(
            as_of_date=as_of_date,
            label=label,
            volatility_level=avg_volatility,
            trend_strength=avg_trend,
            metadata={"feature_count": len(features)},
        )
