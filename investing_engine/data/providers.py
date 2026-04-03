from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from investing_engine.models import MarketBar, NewsItem


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _stable_hash(text: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(text))


def _safe_float(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("raw", value.get("fmt", value.get("longFmt")))
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned or cleaned.upper() in {"N/A", "NONE", "NULL", "-"}:
            return None
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        value = cleaned
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fraction_value(value: Any, *, large_percent_threshold: float = 5.0) -> float | None:
    raw = _safe_float(value)
    if raw is None:
        return None
    if isinstance(value, str) and "%" in value:
        return raw / 100.0
    if abs(raw) > large_percent_threshold:
        return raw / 100.0
    return raw


def _mean_or_default(values: list[float], default: float = 0.5) -> float:
    if not values:
        return float(default)
    return float(sum(values) / len(values))


def _bounded_score(value: float | None, *, low: float, high: float) -> float | None:
    if value is None:
        return None
    if high <= low:
        raise ValueError("high must be greater than low")
    return _clip01((float(value) - low) / (high - low))


def _inverse_score(
    value: float | None,
    *,
    best: float,
    worst: float,
    allow_negative: bool = False,
) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if numeric <= 0.0 and not allow_negative:
        return 0.0
    if numeric <= best:
        return 1.0
    if numeric >= worst:
        return 0.0
    return _clip01(1.0 - ((numeric - best) / (worst - best)))


def _read_env(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw = stripped.split("=", 1)
        if key.strip() != name:
            continue
        return raw.strip().strip('"').strip("'")
    return None


@dataclass(frozen=True)
class FundamentalScoreSnapshot:
    symbol: str
    source: str
    return_on_equity: float | None = None
    pe_ratio: float | None = None
    price_to_book_ratio: float | None = None
    revenue_growth_yoy: float | None = None
    operating_margin: float | None = None
    debt_to_equity_ratio: float | None = None
    fundamental_score: float = 0.5
    valuation_score: float = 0.5
    financial_health: float = 0.5
    is_synthetic: bool = False


@dataclass
class AlphaVantageFundamentalProvider:
    timeout_seconds: float = 10.0
    api_key: str | None = None
    cache: dict[str, FundamentalScoreSnapshot] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = _read_env("ALPHA_VANTAGE_API_KEY")

    def load(self, symbol: str) -> FundamentalScoreSnapshot:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            return self._synthetic_snapshot(symbol="UNKNOWN")
        if normalized_symbol in self.cache:
            return self.cache[normalized_symbol]

        snapshot = self._load_live_snapshot(normalized_symbol)
        if snapshot is None:
            snapshot = self._synthetic_snapshot(normalized_symbol)
        self.cache[normalized_symbol] = snapshot
        return snapshot

    def _load_live_snapshot(self, symbol: str) -> FundamentalScoreSnapshot | None:
        if symbol.endswith((".NS", ".BO")):
            return self._fetch_yahoo_finance(symbol)
        if not self.api_key:
            return None
        return self._fetch_alpha_vantage(symbol)

    def _fetch_alpha_vantage(self, symbol: str) -> FundamentalScoreSnapshot | None:
        query = urllib.parse.urlencode(
            {
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": self.api_key,
            }
        )
        payload = self._get_json(
            f"https://www.alphavantage.co/query?{query}",
            headers={"User-Agent": "institutional-ai-investing-engine/0.1"},
        )
        if not isinstance(payload, dict):
            return None
        if payload.get("Note") or payload.get("Information") or payload.get("Error Message"):
            return None
        metrics = {
            "return_on_equity": _fraction_value(payload.get("ReturnOnEquityTTM")),
            "pe_ratio": _safe_float(payload.get("PERatio")),
            "price_to_book_ratio": _safe_float(payload.get("PriceToBookRatio")),
            "revenue_growth_yoy": _fraction_value(payload.get("RevenueGrowthYOY")),
            "operating_margin": _fraction_value(payload.get("OperatingMarginTTM")),
            "debt_to_equity_ratio": _safe_float(payload.get("DebtToEquityRatio")),
        }
        return self._snapshot_from_metrics(symbol=symbol, source="alpha_vantage", metrics=metrics)

    def _fetch_yahoo_finance(self, symbol: str) -> FundamentalScoreSnapshot | None:
        query = urllib.parse.urlencode(
            {"modules": "financialData,defaultKeyStatistics", "corsDomain": "finance.yahoo.com"}
        )
        payload = self._get_json(
            f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?{query}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if not isinstance(payload, dict):
            return None
        summary = payload.get("quoteSummary", {})
        results = summary.get("result")
        if not isinstance(results, list) or not results:
            return None
        result = results[0]
        financial_data = result.get("financialData", {})
        default_stats = result.get("defaultKeyStatistics", {})
        metrics = {
            "return_on_equity": _fraction_value(financial_data.get("returnOnEquity")),
            "pe_ratio": _safe_float(default_stats.get("trailingPE")),
            "price_to_book_ratio": _safe_float(default_stats.get("priceToBook")),
            "revenue_growth_yoy": _fraction_value(financial_data.get("revenueGrowth")),
            "operating_margin": _fraction_value(financial_data.get("operatingMargins")),
            "debt_to_equity_ratio": _fraction_value(
                financial_data.get("debtToEquity"),
                large_percent_threshold=10.0,
            ),
        }
        return self._snapshot_from_metrics(symbol=symbol, source="yahoo_finance", metrics=metrics)

    def _snapshot_from_metrics(
        self,
        *,
        symbol: str,
        source: str,
        metrics: dict[str, float | None],
    ) -> FundamentalScoreSnapshot | None:
        usable_metric_count = sum(value is not None for value in metrics.values())
        if usable_metric_count == 0:
            return None

        roe_score = _bounded_score(metrics["return_on_equity"], low=0.0, high=0.30)
        growth_score = _bounded_score(metrics["revenue_growth_yoy"], low=-0.05, high=0.20)
        margin_score = _bounded_score(metrics["operating_margin"], low=0.0, high=0.30)
        pe_score = _inverse_score(metrics["pe_ratio"], best=12.0, worst=45.0)
        pb_score = _inverse_score(metrics["price_to_book_ratio"], best=1.5, worst=10.0)
        debt_score = _inverse_score(
            metrics["debt_to_equity_ratio"],
            best=0.0,
            worst=2.0,
            allow_negative=True,
        )

        fundamental_score = _mean_or_default(
            [score for score in [roe_score, growth_score, margin_score] if score is not None]
        )
        valuation_score = _mean_or_default(
            [score for score in [pe_score, pb_score] if score is not None]
        )
        financial_health = _mean_or_default(
            [score for score in [debt_score, margin_score, roe_score] if score is not None]
        )

        return FundamentalScoreSnapshot(
            symbol=symbol,
            source=source,
            return_on_equity=metrics["return_on_equity"],
            pe_ratio=metrics["pe_ratio"],
            price_to_book_ratio=metrics["price_to_book_ratio"],
            revenue_growth_yoy=metrics["revenue_growth_yoy"],
            operating_margin=metrics["operating_margin"],
            debt_to_equity_ratio=metrics["debt_to_equity_ratio"],
            fundamental_score=float(fundamental_score),
            valuation_score=float(valuation_score),
            financial_health=float(financial_health),
            is_synthetic=False,
        )

    def _synthetic_snapshot(self, symbol: str) -> FundamentalScoreSnapshot:
        seed = _stable_hash(symbol)
        fundamental_score = _clip01(0.56 + (((seed % 19) - 9) / 100.0))
        valuation_score = _clip01(0.52 + ((((seed // 5) % 17) - 8) / 100.0))
        financial_health = _clip01(0.63 + ((((seed // 11) % 15) - 7) / 100.0))
        return FundamentalScoreSnapshot(
            symbol=symbol,
            source="synthetic_fallback",
            fundamental_score=float(fundamental_score),
            valuation_score=float(valuation_score),
            financial_health=float(financial_health),
            is_synthetic=True,
        )

    def _get_json(self, url: str, headers: dict[str, str]) -> dict[str, Any] | None:
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (
            TimeoutError,
            urllib.error.HTTPError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ):
            return None


@dataclass
class SyntheticMarketDataSource:
    """Deterministic market generator for offline prototypes and tests."""

    def load(self, symbol: str, as_of_date: date, lookback_days: int) -> list[MarketBar]:
        rng = random.Random(f"market:{symbol}:{as_of_date.isoformat()}:{lookback_days}")
        start_price = 100.0 + (abs(hash(symbol)) % 50)
        current_price = start_price
        bars: list[MarketBar] = []
        for offset in range(lookback_days):
            day = as_of_date - timedelta(days=lookback_days - offset - 1)
            drift = 0.0008 if symbol < "N" else 0.0003
            shock = rng.uniform(-0.025, 0.025)
            current_price *= 1.0 + drift + shock
            high = current_price * (1.0 + rng.uniform(0.0, 0.01))
            low = current_price * (1.0 - rng.uniform(0.0, 0.01))
            open_price = (high + low) / 2.0
            bars.append(
                MarketBar(
                    symbol=symbol,
                    timestamp=datetime.combine(day, time(16, 0)),
                    open=open_price,
                    high=high,
                    low=low,
                    close=current_price,
                    volume=1_000_000 + rng.uniform(-100_000, 100_000),
                )
            )
        return bars


@dataclass
class YFinanceMarketDataSource:
    """Prototype online loader that falls back to synthetic data on failure."""

    fallback: SyntheticMarketDataSource

    def load(self, symbol: str, as_of_date: date, lookback_days: int) -> list[MarketBar]:
        try:
            import pandas as pd
            import yfinance as yf
        except ImportError:
            return self.fallback.load(symbol, as_of_date, lookback_days)

        start = as_of_date - timedelta(days=lookback_days * 2)
        end = as_of_date + timedelta(days=1)
        try:
            frame = yf.download(
                symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
                auto_adjust=False,
            )
        except Exception:
            return self.fallback.load(symbol, as_of_date, lookback_days)
        if frame.empty:
            return self.fallback.load(symbol, as_of_date, lookback_days)

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        bars: list[MarketBar] = []
        trimmed = frame.tail(lookback_days).reset_index()
        for row in trimmed.to_dict(orient="records"):
            timestamp = row["Date"]
            if hasattr(timestamp, "to_pydatetime"):
                timestamp = timestamp.to_pydatetime()
            bars.append(
                MarketBar(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0.0)),
                )
            )
        return bars or self.fallback.load(symbol, as_of_date, lookback_days)


@dataclass
class SyntheticNewsSource:
    def load(self, symbol: str, as_of_date: date, lookback_days: int) -> list[NewsItem]:
        rng = random.Random(f"news:{symbol}:{as_of_date.isoformat()}:{lookback_days}")
        news: list[NewsItem] = []
        template = [
            "beats consensus expectations",
            "announces supply-chain expansion",
            "faces margin pressure concerns",
            "launches new enterprise product",
            "draws analyst downgrade debate",
        ]
        for offset in range(0, lookback_days, 7):
            day = as_of_date - timedelta(days=offset)
            phrase = template[offset % len(template)]
            sentiment = rng.uniform(-0.6, 0.8)
            news.append(
                NewsItem(
                    symbol=symbol,
                    timestamp=datetime.combine(day, time(9, 0)),
                    headline=f"{symbol} {phrase}",
                    sentiment_hint=sentiment,
                )
            )
        return news
