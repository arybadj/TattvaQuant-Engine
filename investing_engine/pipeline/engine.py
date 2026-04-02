from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from investing_engine.config import EngineSettings, load_settings
from investing_engine.data.events import EventPublisher
from investing_engine.data.ingestion import PointInTimeDataPipeline
from investing_engine.execution.broker import ExecutionEngine
from investing_engine.execution.costs import TransactionCostModel
from investing_engine.features.store import FeatureStore
from investing_engine.feedback.loop import FeedbackLoop
from investing_engine.fusion.engine import FusionEngine
from investing_engine.fusion.regime import RegimeDetector
from investing_engine.intelligence.parallel import ParallelIntelligence
from investing_engine.models import FeatureVector, PipelineResult
from investing_engine.rl.decision import DecisionEngine
from investing_engine.rl.reward import RiskAdjustedReward
from investing_engine.uncertainty.engine import UncertaintyEngine

try:
    from prometheus_client import Counter
except ImportError:  # pragma: no cover - optional runtime dependency
    Counter = None


RUN_COUNTER = Counter("investing_engine_runs_total", "Completed investing engine runs") if Counter else None


@dataclass
class InvestingEngine:
    settings: EngineSettings = field(default_factory=load_settings)

    def __post_init__(self) -> None:
        self.data_pipeline = PointInTimeDataPipeline(
            market_source_name=self.settings.market_source,
            news_source_name=self.settings.news_source,
        )
        self.feature_store = FeatureStore(root=self.settings.feature_path, redis_url=self.settings.redis_url)
        self.parallel_intelligence = ParallelIntelligence()
        self.regime_detector = RegimeDetector()
        self.fusion_engine = FusionEngine()
        self.uncertainty_engine = UncertaintyEngine()
        self.cost_model = TransactionCostModel()
        reward_model = RiskAdjustedReward(weights=self.settings.reward_weights)
        self.decision_engine = DecisionEngine(settings=self.settings, reward_model=reward_model)
        self.execution_engine = ExecutionEngine()
        self.feedback_loop = FeedbackLoop(root=self.settings.feedback_path)
        self.publisher = EventPublisher(bootstrap_servers=self.settings.kafka_bootstrap)

    def run_once(self) -> PipelineResult:
        bundle = self.data_pipeline.load(
            symbols=self.settings.symbols,
            as_of_date=self.settings.as_of_date,
            lookback_days=self.settings.lookback_days,
        )
        features = self._build_features(bundle.market_bars, bundle.news_items)
        self.feature_store.persist(features)
        component_signals = self.parallel_intelligence.run(features)
        regime = self.regime_detector.detect(self.settings.as_of_date, features)
        fused_signals = self.fusion_engine.fuse(regime, component_signals)

        provisional_targets = {
            signal.symbol: max(min(signal.score * signal.confidence, 1.0), -1.0) * 0.75 for signal in fused_signals
        }
        estimated_costs = {
            feature.symbol: self.cost_model.estimate(
                feature=feature,
                current_weight=0.0,
                target_weight=provisional_targets[feature.symbol],
            )
            for feature in features
        }

        uncertainties = self.uncertainty_engine.estimate(features, component_signals, regime)
        decisions = self.decision_engine.decide(features, fused_signals, uncertainties, estimated_costs)
        orders = self.execution_engine.build_orders(decisions)
        feedback = self.feedback_loop.capture(decisions, orders)

        result = PipelineResult(
            as_of_date=self.settings.as_of_date,
            regime=regime,
            features=features,
            signals=component_signals + fused_signals,
            uncertainties=uncertainties,
            decisions=decisions,
            orders=orders,
            feedback=feedback,
        )
        self._emit_metrics(result)
        return result

    def _emit_metrics(self, result: PipelineResult) -> None:
        if RUN_COUNTER:
            RUN_COUNTER.inc()
        payload: dict[str, Any] = {
            "event": "pipeline_run_completed",
            "as_of_date": result.as_of_date.isoformat(),
            "order_count": len(result.orders),
            "decision_count": len(result.decisions),
            "regime": result.regime.label.value,
        }
        self.publisher.publish(payload)
        self._track_mlflow(payload)

    def _track_mlflow(self, payload: dict[str, Any]) -> None:
        try:
            import mlflow
        except ImportError:
            return

        try:
            with mlflow.start_run(run_name=f"prototype-{payload['as_of_date']}"):
                for key, value in payload.items():
                    mlflow.log_param(key, value)
        except Exception:
            return

    def _build_features(
        self,
        market_bars: dict[str, list],
        news_items: dict[str, list],
    ) -> list[FeatureVector]:
        features: list[FeatureVector] = []
        for symbol in self.settings.symbols:
            bars = market_bars[symbol]
            closes = [bar.close for bar in bars]
            volumes = [bar.volume for bar in bars]
            if len(closes) < 21:
                raise ValueError(f"Insufficient history for {symbol}")
            momentum_5d = (closes[-1] / closes[-6]) - 1.0
            momentum_20d = (closes[-1] / closes[-21]) - 1.0
            daily_returns = [(closes[index] / closes[index - 1]) - 1.0 for index in range(1, len(closes))]
            realized_volatility = (
                (sum(return_value * return_value for return_value in daily_returns) / max(len(daily_returns), 1)) ** 0.5
            )
            average_volume = sum(volumes) / max(len(volumes), 1)
            sentiment_values = [item.sentiment_hint for item in news_items[symbol]]
            text_sentiment = sum(sentiment_values) / max(len(sentiment_values), 1)
            feature_quality = min(1.0, len(bars) / self.settings.lookback_days)
            features.append(
                FeatureVector(
                    symbol=symbol,
                    as_of_date=self.settings.as_of_date,
                    momentum_5d=momentum_5d,
                    momentum_20d=momentum_20d,
                    realized_volatility=realized_volatility,
                    average_volume=average_volume,
                    text_sentiment=text_sentiment,
                    feature_quality=feature_quality,
                    extras={"news_count": float(len(news_items[symbol]))},
                )
            )
        return features
