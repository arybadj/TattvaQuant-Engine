from __future__ import annotations

from dataclasses import dataclass

from investing_engine.models import FeatureVector, RegimeState, SignalPayload, SignalSource, UncertaintyEstimate


@dataclass
class UncertaintyEngine:
    def estimate(
        self,
        features: list[FeatureVector],
        component_signals: list[SignalPayload],
        regime: RegimeState,
    ) -> list[UncertaintyEstimate]:
        by_symbol: dict[str, list[SignalPayload]] = {}
        by_feature = {feature.symbol: feature for feature in features}
        for signal in component_signals:
            if signal.source != SignalSource.fused:
                by_symbol.setdefault(signal.symbol, []).append(signal)

        outputs: list[UncertaintyEstimate] = []
        for symbol, signals in by_symbol.items():
            scores = [signal.score for signal in signals]
            mean_score = sum(scores) / max(len(scores), 1)
            dispersion = sum(abs(score - mean_score) for score in scores) / max(len(scores), 1)
            feature = by_feature[symbol]
            data_quality_risk = 1.0 - feature.feature_quality
            regime_risk = min(regime.volatility_level * 8.0, 1.0)
            total_uncertainty = min(1.0, (dispersion * 0.5) + (data_quality_risk * 0.2) + (regime_risk * 0.3))
            spread = max(0.02, total_uncertainty * 0.1)
            outputs.append(
                UncertaintyEstimate(
                    symbol=symbol,
                    as_of_date=feature.as_of_date,
                    model_dispersion=dispersion,
                    data_quality_risk=data_quality_risk,
                    regime_risk=regime_risk,
                    total_uncertainty=total_uncertainty,
                    confidence_interval=(mean_score - spread, mean_score + spread),
                )
            )
        return outputs
