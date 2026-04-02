"""Parallel execution for the three-brain intelligence layer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from src.data.timegate import TimeGate
from src.models.event_model import EventModelPipeline, EventSignal
from src.models.fundamental_model import FundamentalModelEnsemble, FundamentalSignal
from src.models.market_model import MarketSignal, MambaMarketModel


@dataclass
class ParallelSignals:
    market: MarketSignal
    event: EventSignal
    fundamental: FundamentalSignal

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParallelIntelligenceLayer:
    market_model: Any
    event_model: EventModelPipeline
    fundamental_model: FundamentalModelEnsemble

    def run(
        self,
        market_inputs: Any,
        symbol: str,
        as_of_date: date,
        gate: TimeGate,
        fundamental_features: dict[str, float],
        industry_context: str = "",
    ) -> ParallelSignals:
        with ThreadPoolExecutor(max_workers=3) as executor:
            market_future = executor.submit(self._run_market, market_inputs)
            event_future = executor.submit(self.event_model.run, symbol, as_of_date, gate)
            fundamental_future = executor.submit(self.fundamental_model.predict, fundamental_features, industry_context)
        return ParallelSignals(
            market=market_future.result(),
            event=event_future.result(),
            fundamental=fundamental_future.result(),
        )

    def _run_market(self, market_inputs: Any) -> MarketSignal:
        output = self.market_model(market_inputs)
        if isinstance(output, MarketSignal):
            return output
        raise TypeError("Market model must return a MarketSignal instance.")


def build_parallel_layer(
    market_model: MambaMarketModel,
    event_model: EventModelPipeline | None = None,
    fundamental_model: FundamentalModelEnsemble | None = None,
) -> ParallelIntelligenceLayer:
    return ParallelIntelligenceLayer(
        market_model=market_model,
        event_model=event_model or EventModelPipeline(),
        fundamental_model=fundamental_model or FundamentalModelEnsemble(),
    )
