from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from investing_engine.intelligence.market import MarketIntelligenceModel
from investing_engine.intelligence.tabular import TabularIntelligenceModel
from investing_engine.intelligence.text import TextIntelligenceModel
from investing_engine.models import FeatureVector, SignalPayload


@dataclass
class ParallelIntelligence:
    market_model: MarketIntelligenceModel = field(default_factory=MarketIntelligenceModel)
    text_model: TextIntelligenceModel = field(default_factory=TextIntelligenceModel)
    tabular_model: TabularIntelligenceModel = field(default_factory=TabularIntelligenceModel)

    def run(self, features: list[FeatureVector]) -> list[SignalPayload]:
        signals: list[SignalPayload] = []
        for feature in features:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(self.market_model.score, feature),
                    executor.submit(self.text_model.score, feature),
                    executor.submit(self.tabular_model.score, feature),
                ]
            signals.extend(future.result() for future in futures)
        return signals
