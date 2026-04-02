from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from investing_engine.models import MarketBar, NewsItem


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
            frame = yf.download(symbol, start=start.isoformat(), end=end.isoformat(), progress=False, auto_adjust=False)
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
