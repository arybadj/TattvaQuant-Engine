"""FastAPI inference endpoint for the investing engine."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.api.logging import configure_logging
from src.execution.execution_engine import LivePipeline, PaperTradingBroker
from src.feedback.feedback_loop import FinalDecision, FinalDecisionFactory

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - optional runtime dependency
    FastAPI = None

try:
    import structlog
except ImportError:  # pragma: no cover - optional runtime dependency
    structlog = None


logger = structlog.get_logger(__name__) if structlog is not None else None
configure_logging()


class InferenceRequest(BaseModel):
    symbol: str
    decision: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    position_size: float
    expected_return_min: float
    expected_return_max: float
    expected_return_median: float
    fundamentals_reason: str
    market_reason: str
    sentiment_reason: str
    regime_reason: str
    risk_factors: list[str] = Field(default_factory=list)
    exit_conditions: list[str] = Field(default_factory=list)
    estimated_cost_bps: float
    shift_warning: bool = False
    recommended_range: str = "3-6 months"
    dynamic_adjustment: bool = True


class DecideRequest(BaseModel):
    symbol: str
    as_of_date: date


if FastAPI is not None:
    app = FastAPI(title="Investing Engine API")
    app.state.paper_broker = PaperTradingBroker()
    app.state.last_run_timestamp = None

    def _pipeline_for(symbol: str) -> LivePipeline:
        return LivePipeline(symbols=[symbol], broker=app.state.paper_broker)

    @app.post("/inference", response_model=FinalDecision)
    def infer(request: InferenceRequest) -> FinalDecision:
        if logger is not None:
            logger.info("inference_request", symbol=request.symbol, decision=request.decision)
        return FinalDecisionFactory.build(
            decision=request.decision,
            confidence=request.confidence,
            position_size=request.position_size,
            expected_return_min=request.expected_return_min,
            expected_return_max=request.expected_return_max,
            expected_return_median=request.expected_return_median,
            fundamentals_reason=request.fundamentals_reason,
            market_reason=request.market_reason,
            sentiment_reason=request.sentiment_reason,
            regime_reason=request.regime_reason,
            risk_factors=request.risk_factors,
            exit_conditions=request.exit_conditions,
            estimated_cost_bps=request.estimated_cost_bps,
            shift_warning=request.shift_warning,
            recommended_range=request.recommended_range,
            dynamic_adjustment=request.dynamic_adjustment,
        )

    @app.post("/decide", response_model=FinalDecision)
    def decide(request: DecideRequest) -> FinalDecision:
        pipeline = _pipeline_for(request.symbol)
        decision = pipeline.run(as_of_date=request.as_of_date)
        timestamp = pipeline.last_run_timestamp or datetime.now(UTC)
        app.state.last_run_timestamp = timestamp.isoformat()
        if logger is not None:
            logger.info(
                "live_decision_completed",
                symbol=request.symbol,
                as_of_date=request.as_of_date.isoformat(),
            )
        return decision

    @app.get("/health")
    def healthcheck() -> dict[str, Any]:
        return {
            "status": "ok",
            "system_status": "ready",
            "last_run_timestamp": app.state.last_run_timestamp,
        }

    @app.get("/portfolio")
    def portfolio() -> dict[str, Any]:
        return app.state.paper_broker.portfolio_state()
