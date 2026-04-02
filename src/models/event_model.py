"""Event-model pipeline for chunking, sentiment scoring, aggregation, and weak labeling."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.timegate import PointInTimeRecord, TimeGate

POSITIVE_LEXICON = {
    "beat",
    "growth",
    "upgrade",
    "strong",
    "expansion",
    "record",
    "raise",
    "raised",
    "improved",
    "profit",
}
NEGATIVE_LEXICON = {
    "miss",
    "downgrade",
    "lawsuit",
    "weak",
    "probe",
    "decline",
    "regulatory",
    "investigation",
    "cut",
    "pressure",
}
RISK_FLAG_PATTERNS = {
    "regulatory": re.compile(r"\bregulatory\b", re.IGNORECASE),
    "lawsuit": re.compile(r"\blawsuit\b", re.IGNORECASE),
    "investigation": re.compile(r"\binvestigation\b", re.IGNORECASE),
    "guidance cut": re.compile(r"\bguidance\s+cut\b|\bcut\s+guidance\b", re.IGNORECASE),
    "margin pressure": re.compile(r"\bmargin\s+pressure\b", re.IGNORECASE),
}


@dataclass
class EventExtraction:
    risk_flags: list[str]
    guidance_tone: str
    earnings_beat: bool
    forward_guidance: str


@dataclass
class EventSignal:
    sentiment_score: float
    event_impact: str
    risk_flags: list[str]
    confidence: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TextChunk:
    text: str
    token_count: int = 0
    published_at: str = ""
    recency_weight: float = 1.0
    symbol: str = ""
    wire_timestamp: datetime | None = None
    chunk_index: int = 0

    def __post_init__(self) -> None:
        if self.token_count == 0 and self.text:
            self.token_count = len(self.text.split())
        if self.wire_timestamp is None and self.published_at:
            self.wire_timestamp = pd.Timestamp(self.published_at).to_pydatetime()
        if not self.published_at and self.wire_timestamp is not None:
            self.published_at = self.wire_timestamp.isoformat()


@dataclass
class ChunkSentiment:
    chunk: TextChunk
    positive: float
    negative: float
    neutral: float

    @property
    def score(self) -> float:
        return float(self.positive - self.negative)

    @property
    def confidence(self) -> float:
        return float(max(self.positive, self.negative, self.neutral))

    @property
    def label(self) -> str:
        if self.positive >= self.negative and self.positive >= self.neutral:
            return "positive"
        if self.negative >= self.positive and self.negative >= self.neutral:
            return "negative"
        return "neutral"

    def to_json(self) -> dict[str, Any]:
        return {
            "chunk": {
                "text": self.chunk.text,
                "symbol": self.chunk.symbol,
                "wire_timestamp": (
                    self.chunk.wire_timestamp.isoformat() if self.chunk.wire_timestamp else ""
                ),
                "chunk_index": self.chunk.chunk_index,
            },
            "positive": self.positive,
            "negative": self.negative,
            "neutral": self.neutral,
            "label": self.label,
            "score": self.score,
            "confidence": self.confidence,
        }


class TextChunker:
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def split(
        self,
        document: str,
        wire_timestamp: datetime | None = None,
        symbol: str = "",
        published_at: str | None = None,
    ) -> list[TextChunk]:
        timestamp = wire_timestamp or (
            pd.Timestamp(published_at).to_pydatetime() if published_at else None
        )
        tokens = document.split()
        if not tokens:
            return [
                TextChunk(
                    text="",
                    token_count=0,
                    published_at=timestamp.isoformat() if timestamp else (published_at or ""),
                    recency_weight=1.0,
                    symbol=symbol,
                    wire_timestamp=timestamp,
                    chunk_index=0,
                )
            ]
        chunks: list[TextChunk] = []
        effective_overlap = self.overlap_tokens if self.max_tokens > self.overlap_tokens else 0
        step = max(self.max_tokens - effective_overlap, 1)
        for chunk_index, start in enumerate(range(0, len(tokens), step)):
            chunk_tokens = tokens[start : start + self.max_tokens]
            if not chunk_tokens:
                continue
            chunks.append(
                TextChunk(
                    text=" ".join(chunk_tokens),
                    token_count=len(chunk_tokens),
                    published_at=timestamp.isoformat() if timestamp else (published_at or ""),
                    recency_weight=1.0 + (chunk_index * 0.05),
                    symbol=symbol,
                    wire_timestamp=timestamp,
                    chunk_index=chunk_index,
                )
            )
            if start + self.max_tokens >= len(tokens):
                break
        return chunks


class FinBERTScorer:
    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        self.model_name = model_name
        self._pipeline = None

    def score_chunks(self, chunks: Iterable[TextChunk]) -> list[ChunkSentiment]:
        return [self.score_chunk(chunk) for chunk in chunks]

    def score_chunk(self, chunk: TextChunk) -> ChunkSentiment:
        pipeline_fn = self._get_pipeline()
        if pipeline_fn is None:
            return self._heuristic_score(chunk)
        try:
            result = pipeline_fn(chunk.text[:4000], truncation=True)[0]
            label = str(result["label"]).lower()
            score = float(result["score"])
            if "pos" in label:
                return ChunkSentiment(
                    chunk=chunk, positive=score, negative=max(0.0, 1.0 - score), neutral=0.0
                )
            if "neg" in label:
                return ChunkSentiment(
                    chunk=chunk, positive=max(0.0, 1.0 - score), negative=score, neutral=0.0
                )
            neutral = score
            residual = max(0.0, 1.0 - neutral)
            return ChunkSentiment(
                chunk=chunk, positive=residual / 2.0, negative=residual / 2.0, neutral=neutral
            )
        except Exception:
            return self._heuristic_score(chunk)

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        except Exception:
            self._pipeline = None
        return self._pipeline

    def _heuristic_score(self, chunk: TextChunk) -> ChunkSentiment:
        tokens = [token.strip(".,!?;:").lower() for token in chunk.text.split()]
        if not tokens:
            return ChunkSentiment(chunk=chunk, positive=0.0, negative=0.0, neutral=1.0)
        positive_hits = sum(token in POSITIVE_LEXICON for token in tokens)
        negative_hits = sum(token in NEGATIVE_LEXICON for token in tokens)
        normalized = max(
            min((positive_hits - negative_hits) / max(len(tokens) / 20.0, 1.0), 1.0), -1.0
        )
        positive = max(normalized, 0.0)
        negative = max(-normalized, 0.0)
        neutral = max(0.0, 1.0 - max(positive, negative))
        total = positive + negative + neutral
        return ChunkSentiment(
            chunk=chunk,
            positive=float(positive / total),
            negative=float(negative / total),
            neutral=float(neutral / total),
        )


class GPT4oExtractor:
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1/responses",
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base

    def extract(self, text: str) -> EventExtraction:
        if not self.api_key:
            return self._heuristic_extract(text)
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Extract structured financial event JSON. "
                                "Return keys: risk_flags (list[str]), guidance_tone (str), "
                                "earnings_beat (bool), forward_guidance (str)."
                            ),
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "input_text", "text": text}]},
            ],
        }
        request = urllib.request.Request(
            self.api_base,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = json.loads(response.read().decode("utf-8"))
            content = raw.get("output", [{}])[0].get("content", [{}])[0].get("text", "{}")
            parsed = json.loads(content)
            return EventExtraction(
                risk_flags=list(parsed.get("risk_flags", [])),
                guidance_tone=str(parsed.get("guidance_tone", "neutral")),
                earnings_beat=bool(parsed.get("earnings_beat", False)),
                forward_guidance=str(parsed.get("forward_guidance", "unchanged")),
            )
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
            return self._heuristic_extract(text)

    def _heuristic_extract(self, text: str) -> EventExtraction:
        risk_flags = sorted(_scan_risk_flags(text))
        lowered = text.lower()
        guidance_tone = (
            "positive" if "raise guidance" in lowered or "strong demand" in lowered else "neutral"
        )
        if "cut guidance" in lowered or "guidance cut" in lowered or "soft demand" in lowered:
            guidance_tone = "negative"
        earnings_beat = any(
            term in lowered for term in ("beat", "above consensus", "tops estimates")
        )
        forward_guidance = (
            "raised"
            if "raise guidance" in lowered
            else ("cut" if "cut guidance" in lowered or "guidance cut" in lowered else "unchanged")
        )
        return EventExtraction(
            risk_flags=risk_flags,
            guidance_tone=guidance_tone,
            earnings_beat=earnings_beat,
            forward_guidance=forward_guidance,
        )


def _scan_risk_flags(text: str) -> set[str]:
    lowered = text.lower()
    detected: set[str] = set()
    for flag, pattern in RISK_FLAG_PATTERNS.items():
        if pattern.search(lowered):
            detected.add(flag)
    return detected


class ChunkAggregator:
    def aggregate(
        self,
        scored_chunks: Sequence[ChunkSentiment | dict[str, Any]],
        extractions: Sequence[EventExtraction],
    ) -> EventSignal:
        if not scored_chunks:
            return EventSignal(
                sentiment_score=0.5, event_impact="neutral", risk_flags=[], confidence=0.0
            )
        normalized = [self._coerce_scored_chunk(item) for item in scored_chunks]
        timestamps = [
            chunk.chunk.wire_timestamp
            for chunk in normalized
            if chunk.chunk.wire_timestamp is not None
        ]
        latest_timestamp = max(timestamps) if timestamps else None
        weighted_sentiment = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        risk_flags: set[str] = set()
        extraction_list = list(extractions)
        for index, chunk_score in enumerate(normalized):
            chunk = chunk_score.chunk
            recency_boost = 1.0
            if latest_timestamp and chunk.wire_timestamp:
                age_hours = max(
                    (latest_timestamp - chunk.wire_timestamp).total_seconds() / 3600.0, 0.0
                )
                recency_boost = 1.0 / (1.0 + (age_hours / 24.0))
            weight = max(chunk.recency_weight, 0.1) * recency_boost
            weighted_sentiment += (chunk_score.positive - chunk_score.negative) * weight
            confidence_sum += chunk_score.confidence * weight
            total_weight += weight
            risk_flags.update(_scan_risk_flags(chunk.text))
            if index < len(extraction_list):
                risk_flags.update(extraction_list[index].risk_flags)
        signed_score = weighted_sentiment / max(total_weight, 1e-6)
        sentiment_score = max(min((signed_score + 1.0) / 2.0, 1.0), 0.0)
        confidence = max(min(confidence_sum / max(total_weight, 1e-6), 1.0), 0.0)
        if sentiment_score > 0.6:
            impact = "positive"
        elif sentiment_score < 0.4:
            impact = "negative"
        else:
            impact = "neutral"
        return EventSignal(
            sentiment_score=float(sentiment_score),
            event_impact=impact,
            risk_flags=sorted(risk_flags),
            confidence=float(confidence),
        )

    def _coerce_scored_chunk(self, scored: ChunkSentiment | dict[str, Any]) -> ChunkSentiment:
        if isinstance(scored, ChunkSentiment):
            return scored
        chunk = scored["chunk"]
        if not isinstance(chunk, TextChunk):
            raise TypeError(
                "ChunkAggregator expects scored chunks to reference TextChunk instances."
            )
        if "positive" in scored and "negative" in scored and "neutral" in scored:
            return ChunkSentiment(
                chunk=chunk,
                positive=float(scored["positive"]),
                negative=float(scored["negative"]),
                neutral=float(scored["neutral"]),
            )
        score = float(scored.get("score", 0.0))
        confidence = max(float(scored.get("confidence", abs(score))), 0.0)
        positive = max(score, 0.0)
        negative = max(-score, 0.0)
        neutral = max(0.0, 1.0 - min(positive + negative, 1.0))
        total = positive + negative + neutral
        if total <= 0:
            return ChunkSentiment(chunk=chunk, positive=0.0, negative=0.0, neutral=1.0)
        scale = min(confidence, 1.0)
        return ChunkSentiment(
            chunk=chunk,
            positive=(positive / total) * scale,
            negative=(negative / total) * scale,
            neutral=max(0.0, 1.0 - scale),
        )


@dataclass
class EventLabeler:
    labels_path: Path = Path("data/labels/event_labels.parquet")

    def label_events(
        self,
        events: pd.DataFrame,
        future_returns: pd.DataFrame,
        symbol_column: str = "symbol",
        event_date_column: str = "event_date",
    ) -> Path:
        merged = events.copy().merge(
            future_returns, on=[symbol_column, event_date_column], how="left"
        )
        merged["label_5d"] = merged["return_5d"].apply(self._threshold_label)
        merged["label_10d"] = merged["return_10d"].apply(self._threshold_label)
        merged["label_30d"] = merged["return_30d"].apply(self._threshold_label)
        ordered_columns = [
            symbol_column,
            event_date_column,
            "return_5d",
            "return_10d",
            "return_30d",
            "label_5d",
            "label_10d",
            "label_30d",
        ]
        merged = merged.loc[:, [column for column in ordered_columns if column in merged.columns]]
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.labels_path, index=False)
        return self.labels_path

    def _threshold_label(self, value: float | None) -> str:
        if value is None or pd.isna(value):
            return "neutral"
        if float(value) >= 0.05:
            return "positive"
        if float(value) <= -0.05:
            return "negative"
        return "neutral"


@dataclass
class EventModelPipeline:
    chunker: TextChunker = field(default_factory=TextChunker)
    finbert: FinBERTScorer = field(default_factory=FinBERTScorer)
    extractor: GPT4oExtractor = field(default_factory=GPT4oExtractor)
    aggregator: ChunkAggregator = field(default_factory=ChunkAggregator)
    labels_path: Path = Path("data/labels/event_labels.parquet")

    def run(self, symbol: str, as_of_date: date, gate: TimeGate) -> EventSignal:
        records: list[PointInTimeRecord] = []
        for data_type in ("news", "filing"):
            records.extend(gate.get(symbol=symbol, as_of_date=as_of_date, data_type=data_type))
        documents = [
            self._record_to_document(record)
            for record in sorted(records, key=lambda item: item.available_at)
        ]
        chunks: list[TextChunk] = []
        for document in documents:
            chunks.extend(
                self.chunker.split(
                    document["text"],
                    wire_timestamp=document["wire_timestamp"],
                    symbol=document["symbol"],
                )
            )
        scored = self.finbert.score_chunks(chunks)
        extracted = [self.extractor.extract(chunk.text) for chunk in chunks]
        return self.aggregator.aggregate(scored, extracted)

    def label_events(
        self,
        events: pd.DataFrame,
        future_returns: pd.DataFrame,
        symbol_column: str = "symbol",
        event_date_column: str = "event_date",
    ) -> Path:
        return EventLabeler(labels_path=self.labels_path).label_events(
            events=events,
            future_returns=future_returns,
            symbol_column=symbol_column,
            event_date_column=event_date_column,
        )

    def _record_to_document(self, record: PointInTimeRecord) -> dict[str, Any]:
        text_fields = []
        for key in ("headline", "body", "content", "summary"):
            if key in record.payload and record.payload[key]:
                text_fields.append(str(record.payload[key]))
        if not text_fields:
            text_fields.append(json.dumps(record.payload))
        return {
            "text": "\n".join(text_fields),
            "symbol": record.symbol,
            "wire_timestamp": record.available_at,
        }
