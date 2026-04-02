from __future__ import annotations

from src.execution.execution_engine import LiquidityFilter, MarketDataSnapshot, Order, OrderDecision, TransactionCostModel
from src.fusion.fusion_engine import FusionEngine


def _market_data() -> dict[str, MarketDataSnapshot]:
    return {
        "RELIANCE": MarketDataSnapshot(symbol="RELIANCE", spread=0.002, adv_20d=1_000_000.0, rolling_volatility_20d=0.02),
        "INFY": MarketDataSnapshot(symbol="INFY", spread=0.001, adv_20d=2_000_000.0, rolling_volatility_20d=0.015),
    }


def test_transaction_cost_model_matches_formula() -> None:
    model = TransactionCostModel(_market_data())
    order = Order(symbol="RELIANCE", size=10_000.0, value=500_000.0)
    cost = model.total_cost(order)
    brokerage = 0.0003 * order.value
    stt = 0.001 * order.value
    slippage = 0.5 * 0.002 * order.value
    participation = order.size / 1_000_000.0
    impact = 0.1 * 0.02 * (participation ** 0.5) * order.value
    expected = brokerage + stt + slippage + impact
    assert abs(cost - expected) < 1e-9


def test_liquidity_filter_caps_size_before_decision() -> None:
    market_data = _market_data()
    FusionEngine.set_expected_alpha("RELIANCE", 100_000.0)
    liquidity = LiquidityFilter(market_data=market_data)
    order = Order(symbol="RELIANCE", size=100_000.0, value=5_000_000.0)
    decision = liquidity.check(order)
    assert order.size == 0.05 * market_data["RELIANCE"].adv_20d
    assert decision == OrderDecision.EXECUTE


def test_liquidity_filter_aborts_when_cost_kills_edge() -> None:
    market_data = _market_data()
    FusionEngine.set_expected_alpha("INFY", 10.0)
    liquidity = LiquidityFilter(market_data=market_data)
    order = Order(symbol="INFY", size=10_000.0, value=1_000_000.0)
    decision = liquidity.check(order)
    assert decision == OrderDecision.ABORT
