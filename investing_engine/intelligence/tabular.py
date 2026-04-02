from __future__ import annotations

from dataclasses import dataclass

from investing_engine.models import FeatureVector, SignalPayload, SignalSource


@dataclass
class TabularIntelligenceModel:
    def score(self, feature: FeatureVector) -> SignalPayload:
        liquidity_bonus = min(feature.average_volume / 5_000_000.0, 1.0) * 0.1
        quality_bonus = feature.feature_quality * 0.2
        raw_score = (feature.momentum_20d * 4.0) + liquidity_bonus + quality_bonus
        score = max(min(raw_score, 1.0), -1.0)
        confidence = max(0.05, min(0.92, 0.4 + feature.feature_quality * 0.5))
        return SignalPayload(
            symbol=feature.symbol,
            as_of_date=feature.as_of_date,
            source=SignalSource.tabular,
            score=score,
            confidence=confidence,
            rationale="Cross-sectional alpha proxy compatible with gradient-boosted models.",
            metadata={"liquidity_bonus": liquidity_bonus, "quality_bonus": quality_bonus},
        )
