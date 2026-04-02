from __future__ import annotations

from dataclasses import dataclass

from investing_engine.models import FeatureVector, SignalPayload, SignalSource


@dataclass
class TextIntelligenceModel:
    model_name: str = "ProsusAI/finbert"

    def score(self, feature: FeatureVector) -> SignalPayload:
        score = max(min(feature.text_sentiment, 1.0), -1.0)
        confidence = max(0.10, min(0.90, 0.55 + abs(score) * 0.25))
        rationale = "Headline sentiment proxy compatible with FinBERT-style pipelines."
        return SignalPayload(
            symbol=feature.symbol,
            as_of_date=feature.as_of_date,
            source=SignalSource.text,
            score=score,
            confidence=confidence,
            rationale=rationale,
            metadata={"model_name": self.model_name},
        )
