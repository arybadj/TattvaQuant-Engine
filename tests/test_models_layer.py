from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from src.data.timegate import PointInTimeRecord, TimeGate
from src.models.event_model import ChunkAggregator, EventExtraction, EventModelPipeline, FinBERTScorer, GPT4oExtractor, TextChunk, TextChunker
from src.models.fundamental_model import FundamentalModelEnsemble
from src.models.market_model import MarketSignal
from src.models.parallel import ParallelIntelligenceLayer


def test_text_chunker_respects_window_size() -> None:
    chunker = TextChunker(max_tokens=5)
    chunks = chunker.split("one two three four five six seven eight nine", published_at="2026-03-30T09:00:00")
    assert len(chunks) == 2
    assert all(chunk.token_count <= 5 for chunk in chunks)


def test_event_aggregator_combines_scores_and_flags() -> None:
    chunk = TextChunk(text="beat guidance", token_count=2, published_at="2026-03-30T09:00:00", recency_weight=1.0)
    scored = [{"chunk": chunk, "score": 0.8, "confidence": 0.9}]
    extracted = [EventExtraction(risk_flags=["regulatory"], guidance_tone="positive", earnings_beat=True, forward_guidance="raised")]
    signal = ChunkAggregator().aggregate(scored, extracted)
    assert signal.sentiment_score > 0
    assert signal.event_impact == "positive"
    assert signal.risk_flags == ["regulatory"]


def test_event_pipeline_uses_timegate_data() -> None:
    gate = TimeGate(
        records=[
            PointInTimeRecord(
                symbol="INFY",
                data_as_of=date(2026, 3, 30),
                available_at=datetime(2026, 3, 30, 9, 0, 0),
                data_type="news",
                payload={"headline": "INFY beat estimates and raised guidance"},
            ),
            PointInTimeRecord(
                symbol="INFY",
                data_as_of=date(2026, 3, 30),
                available_at=datetime(2026, 3, 30, 9, 5, 0),
                data_type="filing",
                payload={"summary": "No legal or regulatory issues disclosed"},
            ),
        ]
    )
    chunker = TextChunker(max_tokens=20)
    chunks = []
    for record in gate.get(symbol="INFY", as_of_date=date(2026, 3, 30), data_type="news"):
        chunks.extend(chunker.split(record.payload["headline"], published_at=record.available_at.isoformat()))
    score = FinBERTScorer().score_chunks(chunks)[0]
    extraction = GPT4oExtractor(api_key=None).extract(chunks[0].text)
    signal = ChunkAggregator().aggregate([score], [extraction])
    assert "sentiment_score" in signal.to_json()


def test_fundamental_ensemble_returns_typed_signal() -> None:
    features = {
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
    signal = FundamentalModelEnsemble().predict(features, industry_context="Moderate disruption risk but strong digital demand.")
    payload = signal.to_json()
    assert "long_term_strength" in payload
    assert "growth_potential" in payload
    assert "risk_score" in payload


def test_event_label_thresholds_write_frame(tmp_path) -> None:
    pipeline = EventModelPipeline(labels_path=tmp_path / "event_labels.parquet")
    events = pd.DataFrame([{"symbol": "INFY", "event_date": "2026-03-30"}])
    future_returns = pd.DataFrame(
        [{"symbol": "INFY", "event_date": "2026-03-30", "return_5d": 0.07, "return_10d": -0.01, "return_30d": -0.08}]
    )
    output = pipeline.label_events(events, future_returns)
    assert output.exists()


def test_parallel_layer_runs_three_brains() -> None:
    gate = TimeGate(
        records=[
            PointInTimeRecord(
                symbol="INFY",
                data_as_of=date(2026, 3, 30),
                available_at=datetime(2026, 3, 30, 9, 0, 0),
                data_type="news",
                payload={"headline": "INFY beat estimates"},
            )
        ]
    )

    class FakeMarketModel:
        def __call__(self, market_inputs):
            return MarketSignal(trend_signal=0.2, momentum_score=0.3, volatility_risk=0.1, predicted_return_5d=0.04)

    layer = ParallelIntelligenceLayer(
        market_model=FakeMarketModel(),
        event_model=EventModelPipeline(),
        fundamental_model=FundamentalModelEnsemble(),
    )
    result = layer.run(
        market_inputs={"dummy": True},
        symbol="INFY",
        as_of_date=date(2026, 3, 30),
        gate=gate,
        fundamental_features={
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
        },
        industry_context="digital demand",
    )
    assert result.market.momentum_score == 0.3
    assert "growth_potential" in result.to_json()["fundamental"]
