from __future__ import annotations

from datetime import date

from src.fusion.fusion_engine import AttentionFusion, FusedSignal, LambdaController, RegimeClassifier, RegimeState
from src.models.event_model import EventSignal
from src.models.fundamental_model import (
    CompanyQualitySignal,
    FundamentalSignal,
    FutureIndustrySignal,
    IndustryHistorySignal,
)
from src.models.market_model import MarketSignal


def _sample_regime(regime_id: int = 3) -> RegimeState:
    probabilities = {
        0: [0.72, 0.12, 0.11, 0.05],
        1: [0.10, 0.68, 0.15, 0.07],
        2: [0.14, 0.10, 0.67, 0.09],
        3: [0.05, 0.08, 0.10, 0.77],
    }
    names = {0: "bull", 1: "bear", 2: "sideways", 3: "crisis"}
    return RegimeState(
        regime_id=regime_id,
        regime_label=names[regime_id],
        regime_proba=probabilities[regime_id],
        as_of_date=date(2026, 3, 30),
    )


def _sample_market_signal() -> MarketSignal:
    return MarketSignal(
        trend_signal=0.64,
        momentum_score=0.58,
        volatility_risk=0.31,
        predicted_return_5d=0.018,
    )


def _sample_event_signal() -> EventSignal:
    return EventSignal(
        sentiment_score=0.61,
        event_impact="positive",
        risk_flags=["guidance cut"],
        confidence=0.73,
    )


def _sample_fundamental_signal() -> FundamentalSignal:
    return FundamentalSignal(
        long_term_strength=0.74,
        growth_potential=0.68,
        risk_score=0.24,
        company_quality=CompanyQualitySignal(fundamental_score=0.71, valuation_score=0.66, health_score=0.81),
        industry_history=IndustryHistorySignal(industry_cagr_5y=0.12, stability_score=0.70),
        future_industry=FutureIndustrySignal(future_growth_score=0.69, disruption_risk=0.27, macro_tailwind=0.63),
        fundamental_score=0.71,
        valuation_score=0.66,
        financial_health=0.81,
    )


def test_regime_classifier_outputs_valid_state_and_probabilities() -> None:
    classifier = RegimeClassifier()
    regime = classifier.predict(
        {
            "vix_level": 16.5,
            "return_20d": 0.045,
            "volume_zscore": 0.4,
            "yield_curve_slope": 0.8,
            "credit_spread": 1.2,
        },
        as_of_date=date(2026, 3, 30),
    )
    assert regime.regime_id in {0, 1, 2, 3}
    assert len(regime.regime_proba) == 4
    assert abs(sum(regime.regime_proba) - 1.0) < 1e-6


def test_lambda_controller_returns_exact_weights_for_all_regimes() -> None:
    controller = LambdaController()
    expected = {
        0: (0.50, 0.30, 0.20, 0.5, 0.3),
        1: (0.30, 0.45, 0.25, 1.0, 0.8),
        2: (0.45, 0.35, 0.20, 0.7, 0.5),
        3: (0.20, 0.50, 0.30, 2.0, 1.5),
    }
    for regime_id, values in expected.items():
        lambdas = controller.get(regime_id)
        assert (lambdas.fundamental, lambdas.market, lambdas.event, lambdas.lambda_cvar, lambdas.lambda_dd) == values


def test_attention_fusion_combined_signal_is_bounded() -> None:
    fusion = AttentionFusion()
    output = fusion.forward(
        _sample_market_signal(),
        _sample_event_signal(),
        _sample_fundamental_signal(),
        _sample_regime(2),
        LambdaController().get(2),
    )
    assert isinstance(output, FusedSignal)
    assert 0.0 <= output.combined_signal <= 1.0


def test_crisis_regime_increases_cvar_lambda_vs_bull() -> None:
    controller = LambdaController()
    assert controller.get(3).lambda_cvar > controller.get(0).lambda_cvar


def test_feature_importance_keys_sum_to_one() -> None:
    output = AttentionFusion().forward(
        _sample_market_signal(),
        _sample_event_signal(),
        _sample_fundamental_signal(),
        _sample_regime(3),
        LambdaController().get(3),
    )
    assert set(output.feature_importance) == {"market", "event", "fundamental"}
    assert abs(sum(output.feature_importance.values()) - 1.0) < 1e-5
