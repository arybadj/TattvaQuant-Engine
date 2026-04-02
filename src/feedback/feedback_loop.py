"""Feedback metrics, audit persistence, and final decision schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.rl.reward import rolling_sharpe

try:
    from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, create_engine
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
except ImportError:  # pragma: no cover - optional runtime dependency
    create_engine = None
    sessionmaker = None
    DeclarativeBase = object
    Mapped = Any
    mapped_column = None
    JSON = Boolean = DateTime = Float = Integer = String = None


class RetrainingSignal(str, Enum):
    SHARPE_DECAY = "sharpe_decay"
    ACCURACY_DECAY = "accuracy_decay"
    DISTRIBUTIONAL_SHIFT = "distributional_shift"
    COST_RATIO_ALERT = "cost_ratio_alert"


@dataclass
class ClosedPositionEvent:
    symbol: str
    closed_at: datetime
    gross_pnl: float
    total_costs: float
    predicted_direction: Literal["UP", "DOWN", "FLAT"]
    actual_return: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def realized_pnl(self) -> float:
        return float(self.gross_pnl - self.total_costs)

    @property
    def actual_direction(self) -> Literal["UP", "DOWN", "FLAT"]:
        if self.actual_return > 0:
            return "UP"
        if self.actual_return < 0:
            return "DOWN"
        return "FLAT"

    @property
    def prediction_correct(self) -> bool:
        return self.predicted_direction == self.actual_direction

    @property
    def cost_ratio(self) -> float:
        if self.gross_pnl <= 0:
            return float("inf")
        return float(self.total_costs / self.gross_pnl)


@dataclass
class FeedbackMetrics:
    realized_pnl: float
    rolling_sharpe_30d: float
    rolling_sharpe_90d: float
    rolling_sharpe_252d: float
    max_drawdown: float
    current_drawdown: float
    prediction_accuracy: float
    cost_ratio: float
    retraining_triggers: list[RetrainingSignal]

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["retraining_triggers"] = [trigger.value for trigger in self.retraining_triggers]
        return payload


if create_engine is not None:

    class Base(DeclarativeBase):
        pass


    class FeedbackAuditRow(Base):
        __tablename__ = "feedback_audit_events"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
        closed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
        realized_pnl: Mapped[float] = mapped_column(Float, nullable=False)
        gross_pnl: Mapped[float] = mapped_column(Float, nullable=False)
        total_costs: Mapped[float] = mapped_column(Float, nullable=False)
        predicted_direction: Mapped[str] = mapped_column(String(8), nullable=False)
        actual_direction: Mapped[str] = mapped_column(String(8), nullable=False)
        actual_return: Mapped[float] = mapped_column(Float, nullable=False)
        prediction_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
        cost_ratio: Mapped[float] = mapped_column(Float, nullable=False)
        shift_detected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
        retraining_triggers: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=list)
        metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)

else:
    Base = object
    FeedbackAuditRow = object


class InvestmentHorizon(BaseModel):
    recommended_range: str
    dynamic_adjustment: bool


class ExpectedReturnRange(BaseModel):
    min: float
    max: float
    median: float


class DecisionReasoning(BaseModel):
    fundamentals: str
    market: str
    sentiment: str
    regime: str


class FinalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    position_size: float
    investment_horizon: InvestmentHorizon
    expected_return: ExpectedReturnRange
    reasoning: DecisionReasoning
    risk_factors: list[str] = Field(default_factory=list)
    exit_conditions: list[str] = Field(default_factory=list)
    estimated_cost_bps: float
    shift_warning: bool

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass
class FeedbackLoop:
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/investing"

    def __post_init__(self) -> None:
        self._events: list[ClosedPositionEvent] = []
        self._engine = None
        self._session_factory = None
        if create_engine is not None and sessionmaker is not None:
            try:
                self._engine = create_engine(self.database_url, future=True)
                if hasattr(Base, "metadata"):
                    Base.metadata.create_all(self._engine)
                self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
            except Exception:
                self._engine = None
                self._session_factory = None

    def track_closed_position(self, event: ClosedPositionEvent, shift_detected: bool = False) -> FeedbackMetrics:
        self._events.append(event)
        metrics = self._compute_metrics(event=event, shift_detected=shift_detected)
        self._store_audit_event(event=event, metrics=metrics, shift_detected=shift_detected)
        return metrics

    def _compute_metrics(self, event: ClosedPositionEvent, shift_detected: bool) -> FeedbackMetrics:
        realized_pnls = [item.realized_pnl for item in self._events]
        rolling_30 = rolling_sharpe(len(realized_pnls) - 1, realized_pnls, 30)
        rolling_90 = rolling_sharpe(len(realized_pnls) - 1, realized_pnls, 90)
        rolling_252 = rolling_sharpe(len(realized_pnls) - 1, realized_pnls, 252)
        max_drawdown, current_drawdown = self._drawdowns(realized_pnls)
        prediction_accuracy = self._prediction_accuracy(window=20)
        triggers = self._retraining_triggers(
            rolling_sharpe_30d=rolling_30,
            prediction_accuracy=prediction_accuracy,
            shift_detected=shift_detected,
            latest_cost_ratio=event.cost_ratio,
        )
        return FeedbackMetrics(
            realized_pnl=event.realized_pnl,
            rolling_sharpe_30d=float(rolling_30),
            rolling_sharpe_90d=float(rolling_90),
            rolling_sharpe_252d=float(rolling_252),
            max_drawdown=float(max_drawdown),
            current_drawdown=float(current_drawdown),
            prediction_accuracy=float(prediction_accuracy),
            cost_ratio=float(event.cost_ratio),
            retraining_triggers=triggers,
        )

    def _drawdowns(self, pnl_series: list[float]) -> tuple[float, float]:
        cumulative = []
        running = 0.0
        for pnl in pnl_series:
            running += pnl
            cumulative.append(running)
        peaks = []
        current_peak = 0.0
        for value in cumulative:
            current_peak = max(current_peak, value)
            peaks.append(current_peak)
        drawdowns = [max(0.0, peak - value) for peak, value in zip(peaks, cumulative, strict=False)]
        max_drawdown = max(drawdowns, default=0.0)
        current_drawdown = drawdowns[-1] if drawdowns else 0.0
        return float(max_drawdown), float(current_drawdown)

    def _prediction_accuracy(self, window: int = 20) -> float:
        sample = self._events[-window:]
        if not sample:
            return 1.0
        correct = sum(1 for item in sample if item.prediction_correct)
        return float(correct / len(sample))

    def _retraining_triggers(
        self,
        rolling_sharpe_30d: float,
        prediction_accuracy: float,
        shift_detected: bool,
        latest_cost_ratio: float,
    ) -> list[RetrainingSignal]:
        triggers: list[RetrainingSignal] = []
        if self._five_day_sharpe_decay():
            triggers.append(RetrainingSignal.SHARPE_DECAY)
        if len(self._events) >= 20 and prediction_accuracy < 0.52:
            triggers.append(RetrainingSignal.ACCURACY_DECAY)
        if shift_detected:
            triggers.append(RetrainingSignal.DISTRIBUTIONAL_SHIFT)
        if latest_cost_ratio > 0.30:
            triggers.append(RetrainingSignal.COST_RATIO_ALERT)
        return triggers

    def _five_day_sharpe_decay(self) -> bool:
        if len(self._events) < 5:
            return False
        realized_pnls = [item.realized_pnl for item in self._events]
        recent_flags = []
        for end_index in range(len(realized_pnls) - 5, len(realized_pnls)):
            recent_flags.append(rolling_sharpe(end_index, realized_pnls, 30) < 0.5)
        return all(recent_flags)

    def _store_audit_event(self, event: ClosedPositionEvent, metrics: FeedbackMetrics, shift_detected: bool) -> None:
        if self._session_factory is None or create_engine is None or FeedbackAuditRow is object:
            return
        try:
            session = self._session_factory()
            row = FeedbackAuditRow(
                symbol=event.symbol,
                closed_at=event.closed_at,
                realized_pnl=event.realized_pnl,
                gross_pnl=event.gross_pnl,
                total_costs=event.total_costs,
                predicted_direction=event.predicted_direction,
                actual_direction=event.actual_direction,
                actual_return=event.actual_return,
                prediction_correct=event.prediction_correct,
                cost_ratio=event.cost_ratio,
                shift_detected=shift_detected,
                retraining_triggers=[trigger.value for trigger in metrics.retraining_triggers],
                metadata_json=event.metadata,
            )
            session.add(row)
            session.commit()
            session.close()
        except Exception:
            return


class FinalDecisionFactory:
    @staticmethod
    def build(
        decision: Literal["BUY", "SELL", "HOLD"],
        confidence: float,
        position_size: float,
        expected_return_min: float,
        expected_return_max: float,
        expected_return_median: float,
        fundamentals_reason: str,
        market_reason: str,
        sentiment_reason: str,
        regime_reason: str,
        risk_factors: list[str],
        exit_conditions: list[str],
        estimated_cost_bps: float,
        shift_warning: bool,
        recommended_range: str = "3-6 months",
        dynamic_adjustment: bool = True,
    ) -> FinalDecision:
        return FinalDecision(
            decision=decision,
            confidence=float(confidence),
            position_size=float(position_size),
            investment_horizon=InvestmentHorizon(
                recommended_range=recommended_range,
                dynamic_adjustment=dynamic_adjustment,
            ),
            expected_return=ExpectedReturnRange(
                min=float(expected_return_min),
                max=float(expected_return_max),
                median=float(expected_return_median),
            ),
            reasoning=DecisionReasoning(
                fundamentals=fundamentals_reason,
                market=market_reason,
                sentiment=sentiment_reason,
                regime=regime_reason,
            ),
            risk_factors=list(risk_factors),
            exit_conditions=list(exit_conditions),
            estimated_cost_bps=float(estimated_cost_bps),
            shift_warning=bool(shift_warning),
        )
