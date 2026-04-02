from __future__ import annotations

from dataclasses import dataclass

from investing_engine.models import RegimeLabel, RegimeState, SignalPayload, SignalSource


@dataclass
class FusionEngine:
    def fuse(self, regime: RegimeState, signals: list[SignalPayload]) -> list[SignalPayload]:
        by_symbol: dict[str, list[SignalPayload]] = {}
        for signal in signals:
            by_symbol.setdefault(signal.symbol, []).append(signal)

        fused: list[SignalPayload] = []
        regime_weight = self._regime_weights(regime.label)
        for symbol, symbol_signals in by_symbol.items():
            weighted_scores = []
            weighted_confidences = []
            for signal in symbol_signals:
                weight = regime_weight.get(signal.source, 1.0)
                weighted_scores.append(signal.score * signal.confidence * weight)
                weighted_confidences.append(signal.confidence * weight)
            total_weight = sum(weighted_confidences) or 1.0
            fused_score = sum(weighted_scores) / total_weight
            fused_confidence = min(0.99, sum(weighted_confidences) / len(weighted_confidences))
            fused.append(
                SignalPayload(
                    symbol=symbol,
                    as_of_date=regime.as_of_date,
                    source=SignalSource.fused,
                    score=fused_score,
                    confidence=fused_confidence,
                    rationale=f"Regime-aware fusion under {regime.label.value}.",
                    metadata={"component_count": len(symbol_signals), "regime": regime.label.value},
                )
            )
        return fused

    def _regime_weights(self, regime: RegimeLabel) -> dict[SignalSource, float]:
        if regime == RegimeLabel.bull:
            return {SignalSource.market: 1.2, SignalSource.text: 0.9, SignalSource.tabular: 1.0}
        if regime == RegimeLabel.bear:
            return {SignalSource.market: 0.8, SignalSource.text: 1.1, SignalSource.tabular: 1.0}
        if regime == RegimeLabel.high_volatility:
            return {SignalSource.market: 0.7, SignalSource.text: 1.0, SignalSource.tabular: 1.2}
        return {SignalSource.market: 1.0, SignalSource.text: 1.0, SignalSource.tabular: 1.0}
