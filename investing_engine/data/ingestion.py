from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from investing_engine.data.providers import SyntheticMarketDataSource, SyntheticNewsSource, YFinanceMarketDataSource
from investing_engine.data.timegate import TimeGate
from investing_engine.models import MarketBar, NewsItem


@dataclass
class PointInTimeBundle:
    market_bars: dict[str, list[MarketBar]]
    news_items: dict[str, list[NewsItem]]


@dataclass
class PointInTimeDataPipeline:
    market_source_name: str = "synthetic"
    news_source_name: str = "synthetic"

    def __post_init__(self) -> None:
        synthetic_market = SyntheticMarketDataSource()
        if self.market_source_name == "yfinance":
            self.market_source = YFinanceMarketDataSource(fallback=synthetic_market)
        else:
            self.market_source = synthetic_market
        self.news_source = SyntheticNewsSource()

    def load(self, symbols: tuple[str, ...], as_of_date: date, lookback_days: int) -> PointInTimeBundle:
        gate = TimeGate(as_of_date=as_of_date)
        market_bars: dict[str, list[MarketBar]] = {}
        news_items: dict[str, list[NewsItem]] = {}
        for symbol in symbols:
            raw_bars = self.market_source.load(symbol=symbol, as_of_date=as_of_date, lookback_days=lookback_days)
            raw_news = self.news_source.load(symbol=symbol, as_of_date=as_of_date, lookback_days=lookback_days)
            market_bars[symbol] = gate.filter_records(raw_bars, lambda record: record.timestamp)
            news_items[symbol] = gate.filter_records(raw_news, lambda record: record.timestamp)
        return PointInTimeBundle(market_bars=market_bars, news_items=news_items)
