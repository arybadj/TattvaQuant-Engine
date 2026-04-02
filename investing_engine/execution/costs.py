from __future__ import annotations

from dataclasses import dataclass

from investing_engine.models import FeatureVector


@dataclass
class TransactionCostModel:
    commission_bps: float = 1.0
    spread_bps: float = 4.0
    impact_multiplier: float = 8.0

    def estimate(self, feature: FeatureVector, current_weight: float, target_weight: float) -> float:
        turnover = abs(target_weight - current_weight)
        commission = turnover * (self.commission_bps / 10_000.0)
        spread = turnover * (self.spread_bps / 10_000.0)
        impact = turnover * feature.realized_volatility * self.impact_multiplier / 100.0
        return commission + spread + impact
