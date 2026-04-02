from __future__ import annotations

from datetime import date, datetime

from src.execution.execution_engine import ExecutionEngine, LivePipeline, MarketDataSnapshot, Order, PaperTradingBroker
from src.feedback.feedback_loop import FinalDecision
from src.fusion.fusion_engine import FusionEngine

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - optional runtime dependency
    TestClient = None

try:
    from src.api.app import app
except ImportError:  # pragma: no cover - optional runtime dependency
    app = None


def _market_data() -> dict[str, MarketDataSnapshot]:
    return {
        "INFY": MarketDataSnapshot(symbol="INFY", spread=0.0015, adv_20d=1_500_000.0, rolling_volatility_20d=0.02),
    }


def test_live_pipeline_returns_valid_final_decision(tmp_path) -> None:
    pipeline = LivePipeline(symbols=["INFY"], artifact_root=tmp_path / "phase7")
    decision = pipeline.run(as_of_date=date(2026, 3, 31))
    assert isinstance(decision, FinalDecision)
    assert decision.decision in {"BUY", "SELL", "HOLD"}


def test_final_decision_schema_has_all_required_fields(tmp_path) -> None:
    pipeline = LivePipeline(symbols=["INFY"], artifact_root=tmp_path / "schema")
    payload = pipeline.run(as_of_date=date(2026, 3, 31)).model_dump(mode="python")
    assert set(payload) == {
        "decision",
        "confidence",
        "position_size",
        "investment_horizon",
        "expected_return",
        "reasoning",
        "risk_factors",
        "exit_conditions",
        "estimated_cost_bps",
        "shift_warning",
    }
    assert set(payload["investment_horizon"]) == {"recommended_range", "dynamic_adjustment"}
    assert set(payload["expected_return"]) == {"min", "max", "median"}
    assert set(payload["reasoning"]) == {"fundamentals", "market", "sentiment", "regime"}


def test_paper_trading_broker_tracks_portfolio_state_correctly_after_orders() -> None:
    broker = PaperTradingBroker(initial_cash=100_000.0)
    execution = ExecutionEngine(broker=broker)
    FusionEngine.set_expected_alpha("INFY", 50_000.0)
    orders = [
        Order(
            symbol="INFY",
            size=100.0,
            value=10_000.0,
            side="buy",
            metadata={"reference_price": 100.0},
        )
    ]
    fills = execution.execute_orders(orders=orders, market_data=_market_data(), executed_at=datetime(2026, 3, 31, 15, 30))
    state = broker.portfolio_state({"INFY": 100.0})
    assert fills
    assert state["fill_count"] == 1
    assert "INFY" in state["positions"]
    assert state["positions"]["INFY"]["quantity"] > 0.0
    assert state["total_equity"] > 0.0


def test_estimated_cost_bps_is_always_present_and_positive(tmp_path) -> None:
    pipeline = LivePipeline(symbols=["INFY"], artifact_root=tmp_path / "costs")
    decision = pipeline.run(as_of_date=date(2026, 3, 31))
    assert decision.estimated_cost_bps > 0.0


def test_fastapi_health_endpoint_returns_200() -> None:
    assert app is not None and TestClient is not None
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_shift_warning_field_is_always_bool(tmp_path) -> None:
    pipeline = LivePipeline(symbols=["INFY"], artifact_root=tmp_path / "shift")
    decision = pipeline.run(as_of_date=date(2026, 3, 31))
    assert isinstance(decision.shift_warning, bool)
