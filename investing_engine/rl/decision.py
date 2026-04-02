from __future__ import annotations

from dataclasses import dataclass

from investing_engine.config import EngineSettings
from investing_engine.models import DecisionContext, FeatureVector, SignalPayload, UncertaintyEstimate
from investing_engine.rl.reward import RiskAdjustedReward


@dataclass
class DecisionEngine:
    settings: EngineSettings
    reward_model: RiskAdjustedReward

    def decide(
        self,
        features: list[FeatureVector],
        fused_signals: list[SignalPayload],
        uncertainties: list[UncertaintyEstimate],
        estimated_costs: dict[str, float],
        current_weights: dict[str, float] | None = None,
    ) -> list[DecisionContext]:
        current_weights = current_weights or {}
        feature_map = {feature.symbol: feature for feature in features}
        uncertainty_map = {item.symbol: item for item in uncertainties}
        decisions: list[DecisionContext] = []
        for signal in fused_signals:
            feature = feature_map[signal.symbol]
            uncertainty = uncertainty_map[signal.symbol]
            current_weight = current_weights.get(signal.symbol, 0.0)
            edge = signal.score * signal.confidence * (1.0 - uncertainty.total_uncertainty)
            target_weight = max(
                -self.settings.max_gross_leverage,
                min(self.settings.max_gross_leverage, edge * 0.75),
            )
            estimated_cost = estimated_costs[signal.symbol]
            turnover = abs(target_weight - current_weight)
            expected_return = signal.score * 0.03
            risk_penalty = feature.realized_volatility + uncertainty.total_uncertainty
            reward_estimate = self.reward_model.compute(
                expected_return=expected_return - estimated_cost,
                volatility=feature.realized_volatility,
                drawdown=max(0.0, -feature.momentum_20d),
                cvar=uncertainty.total_uncertainty,
                turnover=turnover,
                diversification=1.0 - min(abs(target_weight), 1.0),
            )
            decisions.append(
                DecisionContext(
                    symbol=signal.symbol,
                    as_of_date=signal.as_of_date,
                    fused_signal=signal.score,
                    signal_confidence=signal.confidence,
                    total_uncertainty=uncertainty.total_uncertainty,
                    current_weight=current_weight,
                    target_weight=target_weight,
                    expected_return=expected_return,
                    estimated_cost=estimated_cost,
                    risk_penalty=risk_penalty,
                    reward_estimate=reward_estimate,
                    rationale="Target weight derived after subtracting estimated execution cost.",
                )
            )
        return decisions
