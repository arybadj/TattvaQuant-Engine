from __future__ import annotations

from datetime import date, datetime, timedelta

from hypothesis import given
from hypothesis import strategies as st

from investing_engine.config import EngineSettings
from investing_engine.data.timegate import TimeGate
from investing_engine.execution.broker import ExecutionEngine
from investing_engine.execution.costs import TransactionCostModel
from investing_engine.models import FeatureVector, RegimeLabel
from investing_engine.pipeline.engine import InvestingEngine
from investing_engine.rl.decision import DecisionEngine
from investing_engine.rl.reward import RiskAdjustedReward


@given(st.integers(min_value=1, max_value=30))
def test_timegate_blocks_future_records(days_forward: int) -> None:
    as_of_date = date(2026, 3, 30)
    gate = TimeGate(as_of_date=as_of_date)
    future_timestamp = datetime(2026, 3, 30, 23, 59, 59) + timedelta(days=days_forward)
    past_timestamp = datetime(2026, 3, 29, 16, 0, 0)

    class Record:
        def __init__(self, timestamp: datetime) -> None:
            self.timestamp = timestamp

    filtered = gate.filter_records([Record(past_timestamp), Record(future_timestamp)], lambda record: record.timestamp)
    assert len(filtered) == 1
    assert filtered[0].timestamp == past_timestamp


def test_reward_penalizes_risk_and_turnover() -> None:
    reward_model = RiskAdjustedReward(weights=EngineSettings().reward_weights)
    low_risk = reward_model.compute(0.05, 0.01, 0.00, 0.02, 0.05, 0.60)
    high_risk = reward_model.compute(0.05, 0.04, 0.10, 0.20, 0.50, 0.10)
    assert low_risk > high_risk


def test_costs_are_modeled_before_order_generation() -> None:
    settings = EngineSettings(symbols=("AAPL",), as_of_date=date(2026, 3, 30))
    reward_model = RiskAdjustedReward(weights=settings.reward_weights)
    decision_engine = DecisionEngine(settings=settings, reward_model=reward_model)
    feature = FeatureVector(
        symbol="AAPL",
        as_of_date=settings.as_of_date,
        momentum_5d=0.02,
        momentum_20d=0.06,
        realized_volatility=0.03,
        average_volume=1_500_000,
        text_sentiment=0.10,
        feature_quality=1.0,
    )
    from investing_engine.models import SignalPayload, SignalSource, UncertaintyEstimate

    fused_signal = SignalPayload(
        symbol="AAPL",
        as_of_date=settings.as_of_date,
        source=SignalSource.fused,
        score=0.7,
        confidence=0.8,
        rationale="test",
    )
    uncertainty = UncertaintyEstimate(
        symbol="AAPL",
        as_of_date=settings.as_of_date,
        model_dispersion=0.1,
        data_quality_risk=0.0,
        regime_risk=0.1,
        total_uncertainty=0.1,
        confidence_interval=(0.6, 0.8),
    )
    cost_model = TransactionCostModel()
    expected_cost = cost_model.estimate(feature, current_weight=0.0, target_weight=0.42)
    decisions = decision_engine.decide([feature], [fused_signal], [uncertainty], {"AAPL": expected_cost})
    orders = ExecutionEngine().build_orders(decisions)
    assert decisions[0].estimated_cost == expected_cost
    assert orders[0].estimated_cost == expected_cost


def test_pipeline_emits_structured_json_output(tmp_path) -> None:
    settings = EngineSettings(
        symbols=("AAPL", "MSFT"),
        as_of_date=date(2026, 3, 30),
        feature_path=tmp_path / "features",
        feedback_path=tmp_path / "feedback",
        market_source="synthetic",
        news_source="synthetic",
        redis_url="",
    )
    engine = InvestingEngine(settings=settings)
    result = engine.run_once().to_json_dict()
    assert result["as_of_date"] == "2026-03-30"
    assert result["regime"]["label"] in {label.value for label in RegimeLabel}
    assert len(result["features"]) == 2
    assert all("symbol" in item for item in result["orders"])
