from __future__ import annotations

from investing_engine.data.providers import (
    AlphaVantageFundamentalProvider,
    FundamentalScoreSnapshot,
)
from src.models.fundamental_model import FundamentalModelEnsemble


def test_alpha_vantage_provider_maps_overview_metrics_to_scores() -> None:
    provider = AlphaVantageFundamentalProvider(api_key="demo")

    def fake_get_json(url: str, headers: dict[str, str]) -> dict[str, str]:
        assert "alphavantage.co/query" in url
        assert "symbol=AAPL" in url
        assert headers["User-Agent"] == "institutional-ai-investing-engine/0.1"
        return {
            "Symbol": "AAPL",
            "ReturnOnEquityTTM": "1.58",
            "PERatio": "28.4",
            "PriceToBookRatio": "39.7",
            "RevenueGrowthYOY": "0.061",
            "OperatingMarginTTM": "0.318",
            "DebtToEquityRatio": "1.73",
        }

    provider._get_json = fake_get_json  # type: ignore[method-assign]
    snapshot = provider.load("AAPL")

    assert snapshot.source == "alpha_vantage"
    assert snapshot.is_synthetic is False
    assert snapshot.return_on_equity == 1.58
    assert 0.0 <= snapshot.fundamental_score <= 1.0
    assert 0.0 <= snapshot.valuation_score <= 1.0
    assert 0.0 <= snapshot.financial_health <= 1.0


def test_provider_uses_yahoo_finance_for_indian_symbols() -> None:
    provider = AlphaVantageFundamentalProvider(api_key=None)

    def fake_get_json(url: str, headers: dict[str, str]) -> dict[str, object]:
        assert "query1.finance.yahoo.com" in url
        assert "INFY.NS" in url
        assert headers["User-Agent"] == "Mozilla/5.0"
        return {
            "quoteSummary": {
                "result": [
                    {
                        "financialData": {
                            "returnOnEquity": {"raw": 0.318},
                            "revenueGrowth": {"raw": 0.074},
                            "operatingMargins": {"raw": 0.219},
                            "debtToEquity": {"raw": 36.5},
                        },
                        "defaultKeyStatistics": {
                            "trailingPE": {"raw": 27.2},
                            "priceToBook": {"raw": 8.3},
                        },
                    }
                ]
            }
        }

    provider._get_json = fake_get_json  # type: ignore[method-assign]
    snapshot = provider.load("INFY.NS")

    assert snapshot.source == "yahoo_finance"
    assert snapshot.is_synthetic is False
    assert snapshot.debt_to_equity_ratio == 0.365
    assert 0.0 <= snapshot.fundamental_score <= 1.0
    assert 0.0 <= snapshot.valuation_score <= 1.0
    assert 0.0 <= snapshot.financial_health <= 1.0


def test_fundamental_ensemble_uses_provider_scores_when_available() -> None:
    class StubProvider:
        def load(self, symbol: str) -> FundamentalScoreSnapshot:
            return FundamentalScoreSnapshot(
                symbol=symbol,
                source="alpha_vantage",
                fundamental_score=0.91,
                valuation_score=0.22,
                financial_health=0.74,
                is_synthetic=False,
            )

    features = {
        "symbol": "AAPL",
        "piotroski_f_score": 7.0,
        "roe_3y_average": 0.18,
        "free_cash_flow_yield": 0.06,
        "debt_to_equity_delta": -0.05,
        "sector_cagr_3y": 0.10,
        "sector_cagr_5y": 0.12,
        "cyclicality_flag": 0.2,
        "tam_growth_estimate": 0.15,
        "patent_filings_trend": 0.08,
        "rate_regime": 1.0,
        "currency_momentum_usdinr": 0.02,
    }

    signal = FundamentalModelEnsemble(fundamental_provider=StubProvider()).predict(
        features,
        industry_context="Strong platform ecosystem with moderate disruption risk.",
    )

    assert signal.fundamental_score == 0.91
    assert signal.valuation_score == 0.22
    assert signal.financial_health == 0.74
