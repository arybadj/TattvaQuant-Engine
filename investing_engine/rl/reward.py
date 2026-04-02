from __future__ import annotations

from dataclasses import dataclass

from investing_engine.config import RewardWeights


@dataclass
class RiskAdjustedReward:
    weights: RewardWeights

    def compute(
        self,
        expected_return: float,
        volatility: float,
        drawdown: float,
        cvar: float,
        turnover: float,
        diversification: float,
    ) -> float:
        return (
            expected_return * self.weights.return_weight
            - volatility * self.weights.volatility_penalty
            - drawdown * self.weights.drawdown_penalty
            - cvar * self.weights.cvar_penalty
            - turnover * self.weights.turnover_penalty
            + diversification * self.weights.diversification_bonus
        )
