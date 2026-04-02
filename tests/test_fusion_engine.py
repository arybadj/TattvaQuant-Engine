from __future__ import annotations

from datetime import date

import pandas as pd

from src.fusion.fusion_engine import AttentionFusion, LambdaController, RegimeClassification, RegimeClassifier
from src.models.event_model import EventSignal
from src.models.fundamental_model import CompanyQualitySignal, FundamentalSignal, FutureIndustrySignal, IndustryHistorySignal
from src.models.market_model import MarketSignal


def test_lambda_controller_matches_spec() -> None:
    controller = LambdaController()
    bull = controller.get(0)
    crisis = controller.get(3)
    assert (bull.fundamental, bull.market, bull.event) == (0.50, 0.30, 0.20)
    assert (crisis.fundamental, crisis.market, crisis.event) == (0.20, 0.50, 0.30)
    assert bull.lambda_cvar == 0.5
    assert crisis.lambda_dd == 1.5
    assert crisis.lambda_hhi == 0.35


def test_regime_classifier_heuristic_detects_crisis() -> None:
    classifier = RegimeClassifier()
    regime = classifier.predict(
        {
            "vix_level": 38.0,
            "return_20d": -0.15,
            "volume_zscore": 2.2,
            "yield_curve_slope": -0.4,
            "credit_spread": 3.1,
        },
        as_of_date=date(2026, 3, 30),
    )
    assert regime.regime_id == 3
    assert regime.regime_label == "crisis"
    assert len(regime.regime_proba) == 4


def test_regime_classifier_persists_fit_metadata(tmp_path) -> None:
    classifier = RegimeClassifier(artifact_path=tmp_path / "regime_classifier.json")
    history = pd.DataFrame(
        [
            {
                "as_of_date": f"2026-01-{day:02d}",
                "vix_level": 14.0 + (day % 3),
                "return_20d": 0.04,
                "volume_zscore": 0.2,
                "yield_curve_slope": 0.7,
                "credit_spread": 1.0,
            }
            for day in range(1, 25)
        ]
    )
    classifier.fit(history=history, as_of_date=date(2026, 1, 24))
    assert (tmp_path / "regime_classifier.json").exists()


def test_attention_fusion_produces_bounded_signal() -> None:
    regime = RegimeClassification(regime_id=3, regime_label="crisis", regime_proba=[0.05, 0.10, 0.10, 0.75])
    lambda_weights = LambdaController().get(regime.regime_id)
    market = MarketSignal(trend_signal=0.2, momentum_score=0.3, volatility_risk=0.7, predicted_return_5d=0.01)
    event = EventSignal(sentiment_score=-0.1, event_impact="negative", risk_flags=["legal"])
    fundamental = FundamentalSignal(
        long_term_strength=0.6,
        growth_potential=0.4,
        risk_score=0.5,
        company_quality=CompanyQualitySignal(fundamental_score=0.5, valuation_score=0.4, health_score=0.6),
        industry_history=IndustryHistorySignal(industry_cagr_5y=0.1, stability_score=0.5),
        future_industry=FutureIndustrySignal(future_growth_score=0.4, disruption_risk=0.6, macro_tailwind=0.3),
    )
    fusion = AttentionFusion()
    output = fusion.forward(market, event, fundamental, regime, lambda_weights)
    assert 0.0 <= output.combined_signal <= 1.0
    assert set(output.feature_importance) == {"market", "event", "fundamental"}
