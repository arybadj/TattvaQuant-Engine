from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from src.data.timegate import LookaheadError, PointInTimeRecord, TimeGate
from src.models.event_model import ChunkAggregator, ChunkSentiment, EventLabeler, EventModelPipeline, EventSignal, TextChunk, TextChunker


def test_text_chunker_splits_long_text_into_multiple_chunks_of_max_512_tokens() -> None:
    document = " ".join(f"token{i}" for i in range(1200))
    chunks = TextChunker().split(document, wire_timestamp=datetime(2026, 3, 30, 9, 0, 0), symbol="AAPL")
    assert len(chunks) >= 3
    assert all(chunk.token_count <= 512 for chunk in chunks)
    assert all(chunk.symbol == "AAPL" for chunk in chunks)


def test_chunk_aggregator_returns_event_signal_with_all_fields_valid() -> None:
    chunk = TextChunk(text="strong demand beat estimates", symbol="AAPL", wire_timestamp=datetime(2026, 3, 30, 9, 0, 0), chunk_index=0)
    sentiment = ChunkSentiment(chunk=chunk, positive=0.8, negative=0.1, neutral=0.1)
    signal = ChunkAggregator().aggregate([sentiment], [])
    assert isinstance(signal, EventSignal)
    assert 0.0 <= signal.sentiment_score <= 1.0
    assert signal.event_impact in {"positive", "negative", "neutral"}
    assert 0.0 <= signal.confidence <= 1.0


def test_risk_flags_detect_regulatory_and_lawsuit_keywords() -> None:
    chunk = TextChunk(
        text="The company disclosed a regulatory review and a shareholder lawsuit.",
        symbol="AAPL",
        wire_timestamp=datetime(2026, 3, 30, 9, 0, 0),
        chunk_index=0,
    )
    sentiment = ChunkSentiment(chunk=chunk, positive=0.1, negative=0.7, neutral=0.2)
    signal = ChunkAggregator().aggregate([sentiment], [])
    assert "regulatory" in signal.risk_flags
    assert "lawsuit" in signal.risk_flags


def test_event_labeler_writes_parquet_file_with_correct_schema(tmp_path) -> None:
    labeler = EventLabeler(labels_path=tmp_path / "event_labels.parquet")
    events = pd.DataFrame([{"symbol": "AAPL", "event_date": "2026-03-30"}])
    future_returns = pd.DataFrame([{"symbol": "AAPL", "event_date": "2026-03-30", "return_5d": 0.06, "return_10d": 0.01, "return_30d": -0.07}])
    output = labeler.label_events(events, future_returns)
    written = pd.read_parquet(output)
    assert list(written.columns) == ["symbol", "event_date", "return_5d", "return_10d", "return_30d", "label_5d", "label_10d", "label_30d"]
    assert written.loc[0, "label_5d"] == "positive"
    assert written.loc[0, "label_30d"] == "negative"


def test_wire_timestamp_is_always_respected_no_future_documents_processed() -> None:
    gate = TimeGate(
        records=[
            PointInTimeRecord(
                symbol="AAPL",
                data_as_of=date(2026, 3, 30),
                available_at=datetime(2026, 3, 31, 9, 0, 0),
                data_type="news",
                payload={"headline": "AAPL receives strong regulatory approval."},
            )
        ]
    )
    with pytest.raises(LookaheadError):
        EventModelPipeline().run(symbol="AAPL", as_of_date=date(2026, 3, 30), gate=gate)
