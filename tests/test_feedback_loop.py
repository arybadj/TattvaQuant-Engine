from __future__ import annotations

from datetime import datetime, timedelta

from src.feedback.feedback_loop import ClosedPositionEvent, FeedbackLoop, FinalDecisionFactory, RetrainingSignal


def _event(
    day_offset: int,
    gross_pnl: float,
    total_costs: float,
    predicted_direction: str,
    actual_return: float,
) -> ClosedPositionEvent:
    return ClosedPositionEvent(
        symbol="INFY",
        closed_at=datetime(2026, 3, 1) + timedelta(days=day_offset),
        gross_pnl=gross_pnl,
        total_costs=total_costs,
        predicted_direction=predicted_direction,
        actual_return=actual_return,
    )


def test_feedback_loop_tracks_realized_pnl_and_cost_ratio_trigger() -> None:
    loop = FeedbackLoop(database_url="sqlite+pysqlite:///:memory:")
    metrics = loop.track_closed_position(
        _event(day_offset=0, gross_pnl=100.0, total_costs=40.0, predicted_direction="UP", actual_return=0.01)
    )
    assert metrics.realized_pnl == 60.0
    assert metrics.cost_ratio == 0.4
    assert RetrainingSignal.COST_RATIO_ALERT in metrics.retraining_triggers


def test_feedback_loop_triggers_accuracy_decay_under_52_percent() -> None:
    loop = FeedbackLoop(database_url="sqlite+pysqlite:///:memory:")
    for index in range(20):
        predicted = "UP"
        actual = 0.01 if index < 10 else -0.01
        loop.track_closed_position(_event(day_offset=index, gross_pnl=50.0, total_costs=1.0, predicted_direction=predicted, actual_return=actual))
    metrics = loop.track_closed_position(
        _event(day_offset=21, gross_pnl=50.0, total_costs=1.0, predicted_direction="UP", actual_return=-0.01)
    )
    assert RetrainingSignal.ACCURACY_DECAY in metrics.retraining_triggers


def test_feedback_loop_triggers_sharpe_decay_after_five_bad_days() -> None:
    loop = FeedbackLoop(database_url="sqlite+pysqlite:///:memory:")
    metrics = None
    for index in range(6):
        metrics = loop.track_closed_position(
            _event(day_offset=index, gross_pnl=-20.0, total_costs=0.5, predicted_direction="UP", actual_return=-0.02)
        )
    assert metrics is not None
    assert RetrainingSignal.SHARPE_DECAY in metrics.retraining_triggers


def test_feedback_loop_triggers_distributional_shift() -> None:
    loop = FeedbackLoop(database_url="sqlite+pysqlite:///:memory:")
    metrics = loop.track_closed_position(
        _event(day_offset=0, gross_pnl=50.0, total_costs=1.0, predicted_direction="UP", actual_return=0.02),
        shift_detected=True,
    )
    assert RetrainingSignal.DISTRIBUTIONAL_SHIFT in metrics.retraining_triggers


def test_final_decision_schema_matches_required_shape() -> None:
    decision = FinalDecisionFactory.build(
        decision="BUY",
        confidence=0.82,
        position_size=0.12,
        expected_return_min=0.03,
        expected_return_max=0.11,
        expected_return_median=0.07,
        fundamentals_reason="Balance sheet remains strong and cash generation is improving.",
        market_reason="Trend remains constructive above the medium-term base.",
        sentiment_reason="Recent filings and news flow skew positive.",
        regime_reason="Bull regime with contained tail risk.",
        risk_factors=["valuation compression", "macro slowdown"],
        exit_conditions=["thesis change", "trend break"],
        estimated_cost_bps=14.5,
        shift_warning=False,
    )
    payload = decision.to_json()
    assert set(payload.keys()) == {
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
