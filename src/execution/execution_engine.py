"""Execution logic, paper trading, and live pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from enum import Enum
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.timegate import PointInTimeRecord, TimeGate
from src.features.feature_store import FeatureSnapshot, FeatureStore
from src.feedback.feedback_loop import FinalDecision, FinalDecisionFactory
from src.fusion.fusion_engine import AttentionFusion, FusionEngine, FusionOutput, RegimeClassification, RegimeClassifier
from src.models.event_model import EventModelPipeline
from src.models.fundamental_model import FundamentalSignal, FundamentalModelEnsemble
from src.models.market_model import MARKET_FEATURE_COLUMNS, MarketModelTrainer, MarketSignal
from src.models.parallel import ParallelIntelligenceLayer, ParallelSignals
from src.rl.environment import FusionState, PortfolioEnv
from src.uncertainty.uncertainty_engine import UncertaintyEngine, UncertaintyOutput

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _stable_hash(text: str) -> int:
    return sum(ord(character) * (index + 1) for index, character in enumerate(text))


@dataclass
class MarketDataSnapshot:
    symbol: str
    spread: float
    adv_20d: float
    rolling_volatility_20d: float


@dataclass
class Order:
    symbol: str
    size: float
    value: float
    side: str = "buy"
    metadata: dict[str, Any] | None = None


class OrderDecision(str, Enum):
    EXECUTE = "EXECUTE"
    ABORT = "ABORT"


@dataclass
class TransactionCostBreakdown:
    brokerage: float
    stt: float
    slippage: float
    market_impact: float
    total: float

    def to_json(self) -> dict[str, float]:
        return asdict(self)


class TransactionCostModel:
    """Brokerage + tax + spread + square-root impact."""

    def __init__(self, market_data: dict[str, MarketDataSnapshot]) -> None:
        self.market_data = market_data

    def get_spread(self, symbol: str) -> float:
        return float(self.market_data[symbol].spread)

    def avg_daily_volume(self, symbol: str, window: int = 20) -> float:
        _ = window
        return float(self.market_data[symbol].adv_20d)

    def rolling_volatility(self, symbol: str, window: int = 20) -> float:
        _ = window
        return float(self.market_data[symbol].rolling_volatility_20d)

    def total_cost(self, order: Order) -> float:
        return float(self.breakdown(order).total)

    def breakdown(self, order: Order) -> TransactionCostBreakdown:
        brokerage = 0.0003 * order.value
        stt = 0.001 * order.value
        slippage = 0.5 * self.get_spread(order.symbol) * order.value
        market_impact = self.square_root_impact(order)
        total = brokerage + stt + slippage + market_impact
        return TransactionCostBreakdown(
            brokerage=float(brokerage),
            stt=float(stt),
            slippage=float(slippage),
            market_impact=float(market_impact),
            total=float(total),
        )

    def square_root_impact(self, order: Order) -> float:
        return square_root_impact(order, self)


def square_root_impact(order: Order, market: TransactionCostModel) -> float:
    """Industry-standard market impact model."""
    adv = market.avg_daily_volume(order.symbol, window=20)
    participation = order.size / adv if adv > 0 else 0.0
    vol = market.rolling_volatility(order.symbol, window=20)
    impact = 0.1 * vol * sqrt(max(participation, 0.0))
    return float(impact * order.value)


@dataclass
class LiquidityFilter:
    market_data: dict[str, MarketDataSnapshot]
    cost_model: TransactionCostModel | None = None

    MAX_ADV_PARTICIPATION = 0.05

    def __post_init__(self) -> None:
        if self.cost_model is None:
            self.cost_model = TransactionCostModel(self.market_data)

    def get_adv(self, symbol: str) -> float:
        return float(self.market_data[symbol].adv_20d)

    def check(self, order: Order) -> OrderDecision:
        max_size = self.MAX_ADV_PARTICIPATION * self.get_adv(order.symbol)
        if order.size > max_size:
            unit_price = order.value / max(order.size, 1e-9)
            order.size = max_size
            order.value = unit_price * max_size
        cost = self.cost_model.total_cost(order)
        alpha = FusionEngine.expected_alpha(order.symbol)
        if cost > alpha:
            return OrderDecision.ABORT
        return OrderDecision.EXECUTE


@dataclass
class PaperFill:
    symbol: str
    side: str
    quantity: float
    price: float
    notional: float
    transaction_cost: float
    slippage_bps: float
    executed_at: datetime

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["executed_at"] = self.executed_at.isoformat()
        return payload


@dataclass
class PaperPosition:
    quantity: float
    average_price: float

    def to_json(self, market_price: float) -> dict[str, float]:
        return {
            "quantity": float(self.quantity),
            "average_price": float(self.average_price),
            "market_price": float(market_price),
            "market_value": float(self.quantity * market_price),
        }


class PaperTradingBroker:
    """In-memory broker for paper execution with costs and slippage."""

    def __init__(self, initial_cash: float = 100_000.0) -> None:
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[str, PaperPosition] = {}
        self.fills: list[PaperFill] = []
        self.last_prices: dict[str, float] = {}

    def position_quantity(self, symbol: str) -> float:
        position = self.positions.get(symbol)
        return float(position.quantity if position is not None else 0.0)

    def mark_price(self, symbol: str, price: float) -> None:
        self.last_prices[symbol] = float(price)

    def total_equity(self, prices: dict[str, float] | None = None) -> float:
        reference_prices = {**self.last_prices, **(prices or {})}
        market_value = 0.0
        for symbol, position in self.positions.items():
            market_value += float(position.quantity) * float(reference_prices.get(symbol, position.average_price))
        return float(self.cash + market_value)

    def current_weight(self, symbol: str, prices: dict[str, float] | None = None) -> float:
        total_equity = self.total_equity(prices)
        if total_equity <= 0.0:
            return 0.0
        reference_prices = {**self.last_prices, **(prices or {})}
        position = self.positions.get(symbol)
        if position is None:
            return 0.0
        market_price = float(reference_prices.get(symbol, position.average_price))
        return float((position.quantity * market_price) / total_equity)

    def execute(
        self,
        order: Order,
        reference_price: float,
        transaction_cost: float,
        slippage_bps: float,
        executed_at: datetime | None = None,
    ) -> PaperFill | None:
        side = order.side.lower()
        quantity = abs(float(order.size))
        if quantity <= 0.0:
            return None
        slip_multiplier = 1.0 + (slippage_bps / 10_000.0) if side == "buy" else 1.0 - (slippage_bps / 10_000.0)
        execution_price = max(float(reference_price) * slip_multiplier, 0.01)
        if side == "buy":
            max_affordable = max((self.cash - transaction_cost) / execution_price, 0.0)
            quantity = min(quantity, max_affordable)
            if quantity <= 0.0:
                return None
            notional = quantity * execution_price
            current = self.positions.get(order.symbol, PaperPosition(quantity=0.0, average_price=0.0))
            new_quantity = current.quantity + quantity
            average_price = execution_price if new_quantity <= 0.0 else ((current.quantity * current.average_price) + notional) / new_quantity
            self.positions[order.symbol] = PaperPosition(quantity=float(new_quantity), average_price=float(average_price))
            self.cash -= float(notional + transaction_cost)
        else:
            current = self.positions.get(order.symbol)
            if current is None or current.quantity <= 0.0:
                return None
            quantity = min(quantity, current.quantity)
            notional = quantity * execution_price
            remaining = current.quantity - quantity
            if remaining <= 1e-9:
                self.positions.pop(order.symbol, None)
            else:
                self.positions[order.symbol] = PaperPosition(quantity=float(remaining), average_price=float(current.average_price))
            self.cash += float(notional - transaction_cost)
        self.mark_price(order.symbol, execution_price)
        fill = PaperFill(
            symbol=order.symbol,
            side=side,
            quantity=float(quantity),
            price=float(execution_price),
            notional=float(quantity * execution_price),
            transaction_cost=float(transaction_cost),
            slippage_bps=float(slippage_bps),
            executed_at=executed_at or datetime.now(timezone.utc),
        )
        self.fills.append(fill)
        return fill

    def portfolio_state(self, prices: dict[str, float] | None = None) -> dict[str, Any]:
        reference_prices = {**self.last_prices, **(prices or {})}
        positions = {
            symbol: position.to_json(reference_prices.get(symbol, position.average_price))
            for symbol, position in self.positions.items()
        }
        total_market_value = sum(float(payload["market_value"]) for payload in positions.values())
        return {
            "cash": float(self.cash),
            "positions": positions,
            "total_market_value": float(total_market_value),
            "total_equity": float(self.cash + total_market_value),
            "fill_count": len(self.fills),
            "fills": [fill.to_json() for fill in self.fills[-10:]],
        }


class ExecutionEngine:
    """Translate target weights into paper orders and simulate execution."""

    def __init__(self, broker: PaperTradingBroker | None = None) -> None:
        self.broker = broker or PaperTradingBroker()

    def build_orders(
        self,
        target_weights: dict[str, float],
        latest_prices: dict[str, float],
        market_data: dict[str, MarketDataSnapshot],
        expected_alpha: dict[str, float] | None = None,
    ) -> list[Order]:
        equity = self.broker.total_equity(latest_prices)
        orders: list[Order] = []
        for symbol, target_weight in target_weights.items():
            price = float(latest_prices[symbol])
            self.broker.mark_price(symbol, price)
            current_quantity = self.broker.position_quantity(symbol)
            target_quantity = max(float(target_weight), 0.0) * equity / max(price, 1e-9)
            delta_quantity = target_quantity - current_quantity
            if abs(delta_quantity) <= 1e-9:
                continue
            side = "buy" if delta_quantity > 0.0 else "sell"
            value = abs(delta_quantity) * price
            alpha = float((expected_alpha or {}).get(symbol, max(value * 0.03, 50.0)))
            FusionEngine.set_expected_alpha(symbol, alpha)
            orders.append(
                Order(
                    symbol=symbol,
                    size=float(abs(delta_quantity)),
                    value=float(value),
                    side=side,
                    metadata={"target_weight": float(target_weight), "reference_price": float(price)},
                )
            )
        return orders

    def estimate_cost_bps(self, order: Order, market_data: dict[str, MarketDataSnapshot]) -> float:
        if order.value <= 0.0:
            return 0.0
        total_cost = TransactionCostModel(market_data).total_cost(order)
        return float((total_cost / order.value) * 10_000.0)

    def execute_orders(
        self,
        orders: list[Order],
        market_data: dict[str, MarketDataSnapshot],
        executed_at: datetime | None = None,
    ) -> list[PaperFill]:
        fills: list[PaperFill] = []
        cost_model = TransactionCostModel(market_data)
        liquidity_filter = LiquidityFilter(market_data=market_data, cost_model=cost_model)
        for order in orders:
            if liquidity_filter.check(order) != OrderDecision.EXECUTE or order.value <= 0.0:
                continue
            breakdown = cost_model.breakdown(order)
            reference_price = float(order.metadata.get("reference_price", order.value / max(order.size, 1e-9))) if order.metadata else float(order.value / max(order.size, 1e-9))
            slippage_bps = float((breakdown.slippage / max(order.value, 1e-9)) * 10_000.0)
            fill = self.broker.execute(
                order=order,
                reference_price=reference_price,
                transaction_cost=float(breakdown.total),
                slippage_bps=slippage_bps,
                executed_at=executed_at,
            )
            if fill is not None:
                fills.append(fill)
        return fills


class HeuristicMarketModel:
    """Deterministic market-signal adapter for runnable live inference."""

    def __call__(self, market_inputs: Any) -> MarketSignal:
        frame = self._to_frame(market_inputs)
        latest = frame.iloc[-1]
        trend_signal = _clip01(0.5 + (float(latest["log_return"]) * 8.0) + ((float(latest["rsi_14"]) - 0.5) * 0.4))
        momentum_score = _clip01(0.5 + (float(latest["macd"]) * 12.0) + (float(latest["log_return"]) * 6.0))
        volatility_risk = _clip01((float(latest["realized_volatility"]) * 8.0) + max(float(latest["bollinger_width"]), 0.0))
        predicted_return = ((trend_signal - 0.5) * 0.03) + ((momentum_score - 0.5) * 0.02) - ((volatility_risk - 0.5) * 0.01)
        return MarketSignal(
            trend_signal=float(trend_signal),
            momentum_score=float(momentum_score),
            volatility_risk=float(volatility_risk),
            predicted_return_5d=float(predicted_return),
        )

    def _to_frame(self, market_inputs: Any) -> pd.DataFrame:
        if isinstance(market_inputs, pd.DataFrame):
            return market_inputs.loc[:, MARKET_FEATURE_COLUMNS].reset_index(drop=True)
        if torch is not None and hasattr(market_inputs, "detach"):
            array = market_inputs.detach().cpu().numpy()
            if array.ndim == 3:
                array = array[0]
            return pd.DataFrame(array, columns=MARKET_FEATURE_COLUMNS)
        array = np.asarray(market_inputs, dtype=float)
        if array.ndim == 3:
            array = array[0]
        return pd.DataFrame(array, columns=MARKET_FEATURE_COLUMNS)


class LivePipeline:
    """Runnable single-symbol pipeline: features -> brains -> fusion -> uncertainty -> RL env -> execution."""

    def __init__(
        self,
        symbols: list[str],
        broker: PaperTradingBroker | None = None,
        artifact_root: Path = Path("data/live_pipeline"),
        redis_url: str | None = None,
    ) -> None:
        if not symbols:
            raise ValueError("LivePipeline requires at least one symbol.")
        self.symbols = list(symbols)
        self.broker = broker or PaperTradingBroker()
        self.artifact_root = artifact_root
        self.feature_store = FeatureStore(
            offline_root=self.artifact_root / "features",
            registry_path=self.artifact_root / "feature_registry.yaml",
            redis_url=redis_url,
        )
        self.market_model = HeuristicMarketModel()
        self.parallel_intelligence = ParallelIntelligenceLayer(
            market_model=self.market_model,
            event_model=EventModelPipeline(),
            fundamental_model=FundamentalModelEnsemble(),
        )
        self.regime_classifier = RegimeClassifier(artifact_path=self.artifact_root / "regime_classifier.json")
        self.attention_fusion = AttentionFusion()
        self.uncertainty_engine = UncertaintyEngine()
        self.execution_engine = ExecutionEngine(broker=self.broker)
        self.last_run_timestamp: datetime | None = None

    def run(self, as_of_date: date) -> FinalDecision:
        symbol = self.symbols[0]
        gate = self._build_timegate(symbol=symbol, as_of_date=as_of_date)
        snapshot = self.feature_store.materialize(symbol=symbol, as_of_date=as_of_date, gate=gate)
        market_inputs, market_feature_frame, latest_price = self._build_market_inputs(symbol=symbol, as_of_date=as_of_date, gate=gate)
        parallel_signals = self.parallel_intelligence.run(
            market_inputs=market_inputs,
            symbol=symbol,
            as_of_date=as_of_date,
            gate=gate,
            fundamental_features=self._fundamental_feature_row(snapshot),
            industry_context=f"{symbol} operates in a durable, innovation-led industry.",
        )
        regime_history, regime_features = self._regime_inputs(snapshot=snapshot, gate=gate, as_of_date=as_of_date)
        self.regime_classifier.maybe_retrain(history=regime_history, as_of_date=as_of_date)
        regime = self.regime_classifier.predict(regime_features, as_of_date=as_of_date)
        fused = self.attention_fusion.fuse(parallel_signals.market, parallel_signals.event, parallel_signals.fundamental, regime)
        training_frame = market_feature_frame.iloc[:-1] if len(market_feature_frame) > 1 else market_feature_frame
        uncertainty = self.uncertainty_engine.evaluate(
            model=self.market_model,
            model_inputs=market_inputs,
            live_features=market_feature_frame.tail(1),
            as_of_date=as_of_date,
            training_features=training_frame,
        )
        env = self._build_portfolio_env(symbol=symbol, fused=fused, uncertainty=uncertainty, regime=regime, realized_return=float(market_feature_frame["log_return"].iloc[-1]))
        env.reset()
        action, decision_label, expected_return = self._policy_action(
            symbol=symbol,
            latest_price=latest_price,
            fused=fused,
            uncertainty=uncertainty,
            parallel_signals=parallel_signals,
            regime=regime,
        )
        env.step(action)
        target_weight = float(env.current_weights[0])
        market_snapshot = self._market_snapshot(symbol=symbol, snapshot=snapshot, latest_price=latest_price)
        self.broker.mark_price(symbol, latest_price)
        target_alpha = max(abs(expected_return["median"]) * self.broker.total_equity({symbol: latest_price}), latest_price * 0.40)
        orders = self.execution_engine.build_orders(
            target_weights={symbol: target_weight},
            latest_prices={symbol: latest_price},
            market_data={symbol: market_snapshot},
            expected_alpha={symbol: target_alpha},
        )
        estimated_cost_bps = self._estimated_cost_bps(orders=orders, market_data={symbol: market_snapshot}, symbol=symbol, latest_price=latest_price)
        self.execution_engine.execute_orders(
            orders=orders,
            market_data={symbol: market_snapshot},
            executed_at=datetime.combine(as_of_date, time(hour=15, minute=30)),
        )
        position_size = self.broker.current_weight(symbol, {symbol: latest_price})
        decision = FinalDecisionFactory.build(
            decision=decision_label,
            confidence=float(uncertainty.confidence_score),
            position_size=float(position_size),
            expected_return_min=float(expected_return["min"]),
            expected_return_max=float(expected_return["max"]),
            expected_return_median=float(expected_return["median"]),
            fundamentals_reason=self._fundamentals_reason(parallel_signals.fundamental),
            market_reason=self._market_reason(parallel_signals.market, fused),
            sentiment_reason=self._sentiment_reason(parallel_signals.event),
            regime_reason=self._regime_reason(regime, uncertainty),
            risk_factors=self._risk_factors(fused=fused, uncertainty=uncertainty, event_signal=parallel_signals.event, regime=regime),
            exit_conditions=self._exit_conditions(decision=decision_label, latest_price=latest_price, uncertainty=uncertainty),
            estimated_cost_bps=float(estimated_cost_bps),
            shift_warning=bool(uncertainty.shift_detected),
            recommended_range=self._investment_horizon(regime, decision_label),
            dynamic_adjustment=True,
        )
        self.last_run_timestamp = datetime.now(timezone.utc)
        return decision

    def portfolio_state(self) -> dict[str, Any]:
        return self.broker.portfolio_state()

    def _build_timegate(self, symbol: str, as_of_date: date) -> TimeGate:
        records: list[PointInTimeRecord] = []
        business_days = pd.bdate_range(end=pd.Timestamp(as_of_date), periods=130)
        base_price = 90.0 + (_stable_hash(symbol) % 35)
        for index, timestamp in enumerate(business_days):
            close = base_price * (1.0 + (0.0008 * index) + (0.018 * np.sin(index / 14.0)))
            volume = 900_000.0 + (index * 4_500.0) + (_stable_hash(symbol) % 100_000)
            records.append(
                PointInTimeRecord(
                    symbol=symbol,
                    data_as_of=timestamp.date(),
                    available_at=datetime.combine(timestamp.date(), time(hour=16)),
                    data_type="price",
                    payload={
                        "date": timestamp.isoformat(),
                        "open": float(close * 0.996),
                        "high": float(close * 1.008),
                        "low": float(close * 0.992),
                        "close": float(close),
                        "volume": float(volume),
                    },
                )
            )
            records.append(
                PointInTimeRecord(
                    symbol=symbol,
                    data_as_of=timestamp.date(),
                    available_at=datetime.combine(timestamp.date(), time(hour=16, minute=1)),
                    data_type="order_book",
                    payload={"best_bid": float(close * 0.9994), "best_ask": float(close * 1.0006)},
                )
            )
            records.append(
                PointInTimeRecord(
                    symbol=symbol,
                    data_as_of=timestamp.date(),
                    available_at=datetime.combine(timestamp.date(), time(hour=16, minute=2)),
                    data_type="options",
                    payload={"unusual_activity_flag": float(0.15 + (0.15 * ((index % 7) / 6.0)))},
                )
            )

        quarter_ends = pd.date_range(end=pd.Timestamp(as_of_date), periods=12, freq="QE")
        for index, timestamp in enumerate(quarter_ends):
            revenue = 950.0 + (45.0 * index)
            ebitda = revenue * (0.19 + (0.006 * (index % 4)))
            eps_estimate = 1.50 + (0.04 * index)
            eps_actual = eps_estimate * (1.02 + (0.01 * (index % 3)))
            publish_date = pd.Timestamp(timestamp).date() + pd.Timedelta(days=45)
            if pd.Timestamp(publish_date).date() > as_of_date:
                continue
            records.append(
                PointInTimeRecord(
                    symbol=symbol,
                    data_as_of=pd.Timestamp(timestamp).date(),
                    available_at=datetime.combine(pd.Timestamp(publish_date).date(), time(hour=8)),
                    data_type="fundamental",
                    payload={
                        "revenue": float(revenue),
                        "ebitda": float(ebitda),
                        "market_cap": float(25_000.0 + (520.0 * index)),
                        "free_cash_flow": float(185.0 + (12.0 * index)),
                        "roe": float(0.12 + (0.004 * index)),
                        "debt_to_equity": float(0.68 - (0.016 * index)),
                        "eps_actual": float(eps_actual),
                        "eps_estimate": float(eps_estimate),
                        "net_income": float(115.0 + (8.0 * index)),
                        "operating_cash_flow": float(128.0 + (9.0 * index)),
                        "roa": float(0.07 + (0.003 * index)),
                        "long_term_debt_ratio": float(0.48 - (0.012 * index)),
                        "current_ratio": float(1.15 + (0.025 * index)),
                        "shares_outstanding": float(100.0),
                        "gross_margin": float(0.42 + (0.006 * index)),
                        "asset_turnover": float(0.62 + (0.01 * index)),
                    },
                )
            )

        news_dates = pd.date_range(end=pd.Timestamp(as_of_date), periods=8, freq="7D")
        for index, timestamp in enumerate(news_dates):
            risk_flags = ["guidance cut"] if index == 0 else []
            records.append(
                PointInTimeRecord(
                    symbol=symbol,
                    data_as_of=timestamp.date(),
                    available_at=datetime.combine(timestamp.date(), time(hour=9, minute=30)),
                    data_type="news",
                    payload={
                        "headline": f"{symbol} reports strong growth and improving cash generation",
                        "body": f"{symbol} highlighted strong demand, improved margins, and resilient bookings update {index}.",
                        "finbert_sentiment_score": float(0.56 + (0.03 * index)),
                        "topic_vector": [0.12, 0.09, 0.08, 0.18, 0.11, 0.12, 0.10, 0.20],
                        "risk_flags": risk_flags,
                    },
                )
            )
            records.append(
                PointInTimeRecord(
                    symbol=symbol,
                    data_as_of=timestamp.date(),
                    available_at=datetime.combine(timestamp.date(), time(hour=10)),
                    data_type="filing",
                    payload={"summary": f"{symbol} filing notes steady liquidity, strong demand, and manageable leverage."},
                )
            )

        transcript_dates = pd.date_range(end=pd.Timestamp(as_of_date), periods=4, freq="35D")
        for index, timestamp in enumerate(transcript_dates):
            records.append(
                PointInTimeRecord(
                    symbol=symbol,
                    data_as_of=timestamp.date(),
                    available_at=datetime.combine(timestamp.date(), time(hour=18)),
                    data_type="transcript",
                    payload={"tone_score": float(0.08 + (0.04 * index))},
                )
            )

        macro_dates = pd.date_range(end=pd.Timestamp(as_of_date), periods=24, freq="MS")
        for index, timestamp in enumerate(macro_dates):
            records.append(
                PointInTimeRecord(
                    symbol="GLOBAL",
                    data_as_of=timestamp.date(),
                    available_at=datetime.combine(timestamp.date(), time(hour=7)),
                    data_type="macro",
                    payload={
                        "policy_rate": float(4.1 + (0.02 * index)),
                        "sector_relative_strength": float(0.08 + (0.01 * index)),
                        "ff5_market_beta": 1.0,
                        "ff5_size_beta": 0.18,
                        "ff5_value_beta": -0.08,
                        "ff5_profitability_beta": 0.28,
                        "ff5_investment_beta": -0.04,
                        "currency_momentum_usdinr": float(0.01 + (0.002 * (index % 4))),
                        "vix": float(15.0 + (0.6 * (index % 7))),
                        "yield_curve_slope": float(1.1 - (0.015 * index)),
                        "credit_spread": float(1.0 + (0.05 * (index % 4))),
                    },
                )
            )
        return TimeGate(records=records)

    def _build_market_inputs(self, symbol: str, as_of_date: date, gate: TimeGate) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        trainer = MarketModelTrainer.__new__(MarketModelTrainer)
        records = gate.get(symbol=symbol, as_of_date=as_of_date, data_type="price")
        history = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(record.payload["date"]).normalize(),
                    "open": float(record.payload["open"]),
                    "high": float(record.payload["high"]),
                    "low": float(record.payload["low"]),
                    "close": float(record.payload["close"]),
                    "volume": float(record.payload["volume"]),
                }
                for record in records
            ]
        ).sort_values("date")
        rows = [trainer._compute_feature_row(history.iloc[: end_index + 1].copy()) for end_index in range(59, len(history))]
        feature_frame = pd.DataFrame(rows).reset_index(drop=True)
        if feature_frame.empty:
            raise ValueError("Insufficient market history for live inference.")
        market_inputs = feature_frame.loc[:, MARKET_FEATURE_COLUMNS].tail(60).reset_index(drop=True)
        return market_inputs, feature_frame, float(history["close"].iloc[-1])

    def _fundamental_feature_row(self, snapshot: FeatureSnapshot) -> dict[str, float]:
        piotroski = float(snapshot.fundamental.piotroski_f_score)
        component_value = piotroski / 9.0
        row = {
            "symbol": snapshot.symbol,
            "as_of_date": snapshot.as_of_date.isoformat(),
            "piotroski_f_score": piotroski,
            "roe_3y_average": float(snapshot.fundamental.roe_3y_average),
            "free_cash_flow_yield": float(snapshot.fundamental.free_cash_flow_yield),
            "debt_to_equity_delta": float(snapshot.fundamental.debt_to_equity_delta),
            "revenue_growth_yoy": float(snapshot.fundamental.revenue_growth_yoy),
            "ebitda_margin": float(max(snapshot.fundamental.ebitda_margin_trend + 0.18, 0.02)),
            "earnings_surprise_pct": float(snapshot.fundamental.earnings_surprise_pct),
            "sector_cagr_3y": float(max(snapshot.macro.sector_relative_strength * 0.2, 0.05)),
            "sector_cagr_5y": float(max(snapshot.macro.sector_relative_strength * 0.25, 0.08)),
            "cyclicality_flag": float(_clip01(1.0 - snapshot.macro.sector_relative_strength)),
            "tam_growth_estimate": float(max(snapshot.fundamental.revenue_growth_yoy, 0.0)),
            "patent_filings_trend": float(0.07 + (snapshot.text.finbert_sentiment_score * 0.03)),
            "rate_regime": float(snapshot.macro.rate_regime),
            "currency_momentum_usdinr": float(snapshot.macro.currency_momentum_usdinr),
        }
        for key in (
            "piotroski_positive_roa",
            "piotroski_positive_cfo",
            "piotroski_delta_roa",
            "piotroski_cfo_gt_net_income",
            "piotroski_lower_leverage",
            "piotroski_higher_current_ratio",
            "piotroski_no_new_shares",
            "piotroski_higher_gross_margin",
            "piotroski_higher_asset_turnover",
        ):
            row[key] = component_value
        return row

    def _regime_inputs(self, snapshot: FeatureSnapshot, gate: TimeGate, as_of_date: date) -> tuple[pd.DataFrame, dict[str, float]]:
        macro_records = gate.get(symbol="GLOBAL", as_of_date=as_of_date, data_type="macro")
        rows: list[dict[str, Any]] = []
        for record in macro_records:
            rows.append(
                {
                    "as_of_date": record.data_as_of.isoformat(),
                    "vix_level": float(record.payload.get("vix", 18.0)),
                    "return_20d": float(snapshot.time_series.log_return_20d),
                    "volume_zscore": float(snapshot.time_series.volume_zscore_30d),
                    "yield_curve_slope": float(record.payload.get("yield_curve_slope", 1.0)),
                    "credit_spread": float(record.payload.get("credit_spread", 1.0)),
                }
            )
        history = pd.DataFrame(rows).sort_values("as_of_date").reset_index(drop=True)
        latest = history.iloc[-1].to_dict()
        return history, {
            "vix_level": float(latest["vix_level"]),
            "return_20d": float(snapshot.time_series.log_return_20d),
            "volume_zscore": float(snapshot.time_series.volume_zscore_30d),
            "yield_curve_slope": float(latest["yield_curve_slope"]),
            "credit_spread": float(latest["credit_spread"]),
        }

    def _build_portfolio_env(
        self,
        symbol: str,
        fused: FusionOutput,
        uncertainty: UncertaintyOutput,
        regime: RegimeClassification,
        realized_return: float,
    ) -> PortfolioEnv:
        bias_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
        current_cash = _clip01(1.0 - self.broker.current_weight(symbol, self.broker.last_prices))
        return PortfolioEnv(
            asset_symbols=[symbol],
            sector_map={symbol: "core"},
            fusion_states=[
                FusionState(
                    combined_signal=float(fused.combined_signal),
                    short_bias=float(bias_map.get(fused.short_term_bias, 0.0)),
                    long_bias=float(bias_map.get(fused.long_term_bias, 0.0)),
                    confidence_score=float(uncertainty.confidence_score),
                    risk_level=uncertainty.risk_level,
                    regime=regime,
                    uncertainty=uncertainty,
                )
            ],
            uncertainty_states=[uncertainty],
            realized_returns=[{symbol: float(realized_return)}],
            initial_cash=max(current_cash, 0.05),
            include_hedge_action=True,
        )

    def _market_snapshot(self, symbol: str, snapshot: FeatureSnapshot, latest_price: float) -> MarketDataSnapshot:
        spread_fraction = max(snapshot.time_series.bid_ask_spread / max(latest_price, 1e-9), 0.0002)
        return MarketDataSnapshot(
            symbol=symbol,
            spread=float(spread_fraction),
            adv_20d=float(max(snapshot.time_series.adv_30d, 10_000.0)),
            rolling_volatility_20d=float(max(snapshot.time_series.realized_volatility_20d, 0.01)),
        )

    def _policy_action(
        self,
        symbol: str,
        latest_price: float,
        fused: FusionOutput,
        uncertainty: UncertaintyOutput,
        parallel_signals: ParallelSignals,
        regime: RegimeClassification,
    ) -> tuple[list[float], str, dict[str, float]]:
        edge = (float(fused.combined_signal) - 0.5) * 2.0
        decision = "BUY"
        if edge < -0.08:
            decision = "SELL" if self.broker.position_quantity(symbol) > 0.0 else "HOLD"
        elif abs(edge) <= 0.08:
            decision = "HOLD"

        raw_weight = max(edge, 0.0) * float(uncertainty.confidence_score) * 0.35
        if uncertainty.shift_detected:
            raw_weight *= 0.5
        if regime.regime_name == "crisis":
            raw_weight *= 0.7
        if decision == "SELL":
            target_weight = 0.0
        elif decision == "HOLD":
            target_weight = self.broker.current_weight(symbol, {symbol: latest_price})
        else:
            target_weight = min(raw_weight, 0.25)
        cash_allocation = max(0.05, 1.0 - target_weight)
        hedge_flag = 1.0 if uncertainty.risk_level == "high" or regime.regime_name == "crisis" else 0.0

        market_component = float(parallel_signals.market.predicted_return_5d)
        fundamental_component = (float(parallel_signals.fundamental.long_term_strength) - 0.5) * 0.04
        event_component = (float(parallel_signals.event.sentiment_score) - 0.5) * 0.03
        median = market_component + fundamental_component + event_component
        spread = max(0.01, abs(median) * 0.5)
        return [float(target_weight), float(cash_allocation), float(hedge_flag)], decision, {
            "min": float(median - spread),
            "median": float(median),
            "max": float(median + spread),
        }

    def _estimated_cost_bps(
        self,
        orders: list[Order],
        market_data: dict[str, MarketDataSnapshot],
        symbol: str,
        latest_price: float,
    ) -> float:
        if orders:
            return max(self.execution_engine.estimate_cost_bps(orders[0], market_data), 0.01)
        hypothetical = Order(symbol=symbol, size=1.0, value=max(latest_price, 1.0), side="buy", metadata={"reference_price": latest_price})
        return max(self.execution_engine.estimate_cost_bps(hypothetical, market_data), 0.01)

    def _fundamentals_reason(self, signal: FundamentalSignal) -> str:
        return (
            f"Long-term strength {signal.long_term_strength:.2f}, valuation {signal.valuation_score:.2f}, "
            f"and health {signal.financial_health:.2f} support the structural view."
        )

    def _market_reason(self, signal: MarketSignal, fused: FusionOutput) -> str:
        return (
            f"Trend {signal.trend_signal:.2f}, momentum {signal.momentum_score:.2f}, and fused signal "
            f"{fused.combined_signal:.2f} point to a {fused.short_term_bias} market bias."
        )

    def _sentiment_reason(self, signal: Any) -> str:
        suffix = f" Risk flags: {', '.join(signal.risk_flags)}." if signal.risk_flags else ""
        return f"Event sentiment is {signal.event_impact} with score {signal.sentiment_score:.2f}.{suffix}"

    def _regime_reason(self, regime: RegimeClassification, uncertainty: UncertaintyOutput) -> str:
        return (
            f"Regime classifier assigns {regime.regime_name} with probability {max(regime.regime_proba):.2f}; "
            f"uncertainty is {uncertainty.risk_level}."
        )

    def _risk_factors(
        self,
        fused: FusionOutput,
        uncertainty: UncertaintyOutput,
        event_signal: Any,
        regime: RegimeClassification,
    ) -> list[str]:
        factors = [f"regime:{regime.regime_name}", f"uncertainty:{uncertainty.risk_level}"]
        if uncertainty.shift_detected:
            factors.append("distributional_shift_detected")
        if fused.long_term_bias == "negative":
            factors.append("long_term_bias_negative")
        factors.extend(f"event:{flag}" for flag in event_signal.risk_flags)
        return factors

    def _exit_conditions(self, decision: str, latest_price: float, uncertainty: UncertaintyOutput) -> list[str]:
        return [
            f"Take profit above {latest_price * 1.08:.2f}",
            f"Stop loss below {latest_price * 0.94:.2f}",
            "Cut position if distributional shift persists" if uncertainty.shift_detected else "Review on regime change",
            "Reduce exposure if uncertainty rises to high" if decision != "SELL" else "Remain in cash after exit",
        ]

    def _investment_horizon(self, regime: RegimeClassification, decision: str) -> str:
        if regime.regime_name == "crisis":
            return "1-4 weeks"
        if decision == "HOLD":
            return "1-3 months"
        return "3-6 months"
