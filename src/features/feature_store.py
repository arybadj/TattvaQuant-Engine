"""Feature store, validation, and pipeline orchestration."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.data.timegate import PointInTimeRecord, TimeGate


class FeatureError(ValueError):
    """Raised when a feature pipeline emits invalid output."""


def _validate_numeric(name: str, value: float) -> float:
    if value is None or math.isnan(float(value)) or math.isinf(float(value)):
        raise FeatureError(f"Critical feature '{name}' cannot be NaN or infinite.")
    return float(value)


class StrictFeatureModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._ensure_no_nan()

    def _ensure_no_nan(self) -> None:
        for key, value in self.model_dump(mode="python").items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                _validate_numeric(key, float(value))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    _validate_numeric(f"{key}[{index}]", item)


class TimeSeriesFeatures(StrictFeatureModel):
    symbol: str
    as_of_date: date
    log_return_1d: float
    log_return_5d: float
    log_return_20d: float
    rolling_mean_10d: float
    rolling_mean_20d: float
    rolling_mean_50d: float
    rolling_std_10d: float
    rolling_std_20d: float
    rolling_std_50d: float
    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    bollinger_band_width: float
    realized_volatility_20d: float
    volume_zscore_30d: float
    bid_ask_spread: float
    adv_30d: float
    unusual_options_activity: float


class FundamentalFeatures(StrictFeatureModel):
    symbol: str
    as_of_date: date
    revenue_growth_yoy: float
    revenue_growth_qoq: float
    ebitda_margin_trend: float
    free_cash_flow_yield: float
    roe_3y_average: float
    debt_to_equity_delta: float
    earnings_surprise_pct: float
    piotroski_f_score: float


class TextFeatures(StrictFeatureModel):
    symbol: str
    as_of_date: date
    finbert_sentiment_score: float
    topic_vector: list[float] = Field(min_length=1)
    news_volume_zscore: float
    earnings_call_tone_delta: float
    risk_flag_count: float


class MacroFeatures(StrictFeatureModel):
    symbol: str
    as_of_date: date
    rate_regime: float
    sector_relative_strength: float
    ff5_market_beta: float
    ff5_size_beta: float
    ff5_value_beta: float
    ff5_profitability_beta: float
    ff5_investment_beta: float
    currency_momentum_usdinr: float
    vix_regime: float


class FeatureSnapshot(StrictFeatureModel):
    symbol: str
    as_of_date: date
    time_series: TimeSeriesFeatures
    fundamental: FundamentalFeatures
    text: TextFeatures
    macro: MacroFeatures

    def flatten(self) -> dict[str, Any]:
        row = {
            "symbol": self.symbol,
            "as_of_date": self.as_of_date.isoformat(),
        }
        for group_name in ("time_series", "fundamental", "text", "macro"):
            group = getattr(self, group_name).model_dump(mode="python")
            for key, value in group.items():
                if key in {"symbol", "as_of_date"}:
                    continue
                row[key] = value
        return row


def _payload_frame(records: Iterable[PointInTimeRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "symbol": record.symbol,
            "data_as_of": record.data_as_of,
            "available_at": record.available_at,
        }
        row.update(record.payload)
        rows.append(row)
    return pd.DataFrame(rows)


def _ema(values: Sequence[float], span: int) -> list[float]:
    alpha = 2.0 / (span + 1.0)
    result: list[float] = []
    for value in values:
        if not result:
            result.append(value)
        else:
            result.append((alpha * value) + ((1.0 - alpha) * result[-1]))
    return result


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise FeatureError("Division by zero encountered in feature computation.")
    return float(numerator) / float(denominator)


def _rolling_zscore(series: pd.Series, window: int) -> float:
    window_series = series.tail(window)
    mean = float(window_series.mean())
    std = float(window_series.std(ddof=0))
    if std == 0:
        return 0.0
    return float((window_series.iloc[-1] - mean) / std)


@dataclass
class TimeSeriesFeaturePipeline:
    def compute(self, symbol: str, as_of_date: date, gate: TimeGate) -> TimeSeriesFeatures:
        price_records = gate.get(symbol=symbol, as_of_date=as_of_date, data_type="price")
        order_book_records = gate.get(symbol=symbol, as_of_date=as_of_date, data_type="order_book")
        options_records = gate.get(symbol=symbol, as_of_date=as_of_date, data_type="options")

        price_frame = _payload_frame(price_records).sort_values("data_as_of")
        if len(price_frame) < 50:
            raise FeatureError(f"Need at least 50 price observations for {symbol}.")

        closes = price_frame["close"].astype(float).reset_index(drop=True)
        volumes = price_frame["volume"].astype(float).reset_index(drop=True)
        log_prices = closes.apply(math.log)
        log_returns = log_prices.diff().fillna(0.0)

        delta = closes.diff().fillna(0.0)
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        avg_gain = gains.rolling(14).mean().iloc[-1]
        avg_loss = losses.rolling(14).mean().iloc[-1]
        rs = float("inf") if avg_loss == 0 else float(avg_gain / avg_loss)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        ema_12 = _ema(closes.tolist(), 12)
        ema_26 = _ema(closes.tolist(), 26)
        macd_line_series = [fast - slow for fast, slow in zip(ema_12, ema_26)]
        macd_signal_series = _ema(macd_line_series, 9)

        rolling_mean_20 = float(closes.tail(20).mean())
        rolling_std_20 = float(closes.tail(20).std(ddof=0))
        bollinger_width = 0.0 if rolling_mean_20 == 0 else (4.0 * rolling_std_20) / rolling_mean_20

        latest_spread = 0.0
        if order_book_records:
            latest_order_book = sorted(order_book_records, key=lambda item: item.available_at)[-1]
            bid = float(latest_order_book.payload.get("best_bid", 0.0))
            ask = float(latest_order_book.payload.get("best_ask", 0.0))
            latest_spread = max(0.0, ask - bid)

        unusual_options_activity = 0.0
        if options_records:
            unusual_options_activity = float(
                sorted(options_records, key=lambda item: item.available_at)[-1].payload.get("unusual_activity_flag", 0.0)
            )

        return TimeSeriesFeatures(
            symbol=symbol,
            as_of_date=as_of_date,
            log_return_1d=float(log_returns.iloc[-1]),
            log_return_5d=float(math.log(closes.iloc[-1] / closes.iloc[-6])),
            log_return_20d=float(math.log(closes.iloc[-1] / closes.iloc[-21])),
            rolling_mean_10d=float(closes.tail(10).mean()),
            rolling_mean_20d=rolling_mean_20,
            rolling_mean_50d=float(closes.tail(50).mean()),
            rolling_std_10d=float(closes.tail(10).std(ddof=0)),
            rolling_std_20d=rolling_std_20,
            rolling_std_50d=float(closes.tail(50).std(ddof=0)),
            rsi_14=float(rsi),
            macd_line=float(macd_line_series[-1]),
            macd_signal=float(macd_signal_series[-1]),
            macd_histogram=float(macd_line_series[-1] - macd_signal_series[-1]),
            bollinger_band_width=float(bollinger_width),
            realized_volatility_20d=float(log_returns.tail(20).std(ddof=0) * math.sqrt(252.0)),
            volume_zscore_30d=float(_rolling_zscore(volumes, 30)),
            bid_ask_spread=latest_spread,
            adv_30d=float(volumes.tail(30).mean()),
            unusual_options_activity=unusual_options_activity,
        )


@dataclass
class FundamentalFeaturePipeline:
    def compute(self, symbol: str, as_of_date: date, gate: TimeGate) -> FundamentalFeatures:
        records = gate.get(symbol=symbol, as_of_date=as_of_date, data_type="fundamental")
        frame = _payload_frame(records).sort_values("data_as_of")
        if len(frame) < 5:
            raise FeatureError(f"Need at least 5 fundamental observations for {symbol}.")

        revenue = frame["revenue"].astype(float).reset_index(drop=True)
        ebitda = frame["ebitda"].astype(float).reset_index(drop=True)
        market_cap = frame["market_cap"].astype(float).reset_index(drop=True)
        free_cash_flow = frame["free_cash_flow"].astype(float).reset_index(drop=True)
        roe = frame["roe"].astype(float).reset_index(drop=True)
        debt_to_equity = frame["debt_to_equity"].astype(float).reset_index(drop=True)
        eps_actual = frame["eps_actual"].astype(float).reset_index(drop=True)
        eps_estimate = frame["eps_estimate"].astype(float).reset_index(drop=True)

        revenue_growth_yoy = _safe_divide(revenue.iloc[-1] - revenue.iloc[-5], revenue.iloc[-5])
        revenue_growth_qoq = _safe_divide(revenue.iloc[-1] - revenue.iloc[-2], revenue.iloc[-2])
        ebitda_margin_latest = _safe_divide(ebitda.iloc[-1], revenue.iloc[-1])
        ebitda_margin_prior = _safe_divide(ebitda.iloc[-2], revenue.iloc[-2])
        roe_3y_average = float(roe.tail(min(len(roe), 12)).mean())
        earnings_surprise_pct = _safe_divide(eps_actual.iloc[-1] - eps_estimate.iloc[-1], abs(eps_estimate.iloc[-1]))

        return FundamentalFeatures(
            symbol=symbol,
            as_of_date=as_of_date,
            revenue_growth_yoy=float(revenue_growth_yoy),
            revenue_growth_qoq=float(revenue_growth_qoq),
            ebitda_margin_trend=float(ebitda_margin_latest - ebitda_margin_prior),
            free_cash_flow_yield=float(_safe_divide(free_cash_flow.iloc[-1], market_cap.iloc[-1])),
            roe_3y_average=roe_3y_average,
            debt_to_equity_delta=float(debt_to_equity.iloc[-1] - debt_to_equity.iloc[-2]),
            earnings_surprise_pct=float(earnings_surprise_pct),
            piotroski_f_score=float(self._piotroski_score(frame.tail(2))),
        )

    def _piotroski_score(self, frame: pd.DataFrame) -> int:
        latest = frame.iloc[-1]
        prior = frame.iloc[0]
        score = 0
        score += int(float(latest.get("net_income", 0.0)) > 0)
        score += int(float(latest.get("operating_cash_flow", 0.0)) > 0)
        score += int(float(latest.get("operating_cash_flow", 0.0)) > float(latest.get("net_income", 0.0)))
        score += int(float(latest.get("roa", 0.0)) > float(prior.get("roa", 0.0)))
        score += int(float(latest.get("long_term_debt_ratio", 1.0)) < float(prior.get("long_term_debt_ratio", 1.0)))
        score += int(float(latest.get("current_ratio", 0.0)) > float(prior.get("current_ratio", 0.0)))
        score += int(float(latest.get("shares_outstanding", 0.0)) <= float(prior.get("shares_outstanding", 0.0)))
        score += int(float(latest.get("gross_margin", 0.0)) > float(prior.get("gross_margin", 0.0)))
        score += int(float(latest.get("asset_turnover", 0.0)) > float(prior.get("asset_turnover", 0.0)))
        return score


@dataclass
class TextFeaturePipeline:
    topic_vector_size: int = 8

    def compute(self, symbol: str, as_of_date: date, gate: TimeGate) -> TextFeatures:
        news_records = gate.get(symbol=symbol, as_of_date=as_of_date, data_type="news")
        transcript_records = gate.get(symbol=symbol, as_of_date=as_of_date, data_type="transcript")
        if not news_records:
            raise FeatureError(f"Need at least one news record for {symbol}.")

        news_frame = _payload_frame(news_records).sort_values("available_at")
        transcript_frame = _payload_frame(transcript_records).sort_values("available_at") if transcript_records else pd.DataFrame()

        sentiment_scores = news_frame.get("finbert_sentiment_score", pd.Series([0.0] * len(news_frame))).astype(float)
        risk_flag_count = 0.0
        if "risk_flags" in news_frame.columns:
            risk_flag_count = float(sum(len(flags) if isinstance(flags, list) else 0 for flags in news_frame["risk_flags"]))

        daily_counts = (
            news_frame.assign(news_day=pd.to_datetime(news_frame["available_at"]).dt.date)
            .groupby("news_day")
            .size()
            .astype(float)
        )
        news_volume_zscore = 0.0 if len(daily_counts) < 2 else float(_rolling_zscore(daily_counts, min(len(daily_counts), 30)))

        latest_topic_vector = None
        for candidate in reversed(news_frame.get("topic_vector", pd.Series([], dtype=object)).tolist()):
            if isinstance(candidate, list) and candidate:
                latest_topic_vector = [float(value) for value in candidate[: self.topic_vector_size]]
                break
        if latest_topic_vector is None:
            latest_topic_vector = [0.0] * self.topic_vector_size
        if len(latest_topic_vector) < self.topic_vector_size:
            latest_topic_vector.extend([0.0] * (self.topic_vector_size - len(latest_topic_vector)))

        tone_delta = 0.0
        if len(transcript_frame) >= 2 and "tone_score" in transcript_frame.columns:
            tone_scores = transcript_frame["tone_score"].astype(float).reset_index(drop=True)
            tone_delta = float(tone_scores.iloc[-1] - tone_scores.iloc[-2])

        return TextFeatures(
            symbol=symbol,
            as_of_date=as_of_date,
            finbert_sentiment_score=float(sentiment_scores.iloc[-1]),
            topic_vector=latest_topic_vector,
            news_volume_zscore=float(news_volume_zscore),
            earnings_call_tone_delta=tone_delta,
            risk_flag_count=risk_flag_count,
        )


@dataclass
class MacroFeaturePipeline:
    def compute(self, symbol: str, as_of_date: date, gate: TimeGate) -> MacroFeatures:
        macro_records = gate.get(symbol=symbol, as_of_date=as_of_date, data_type="macro")
        if not macro_records:
            macro_records = gate.get(symbol="GLOBAL", as_of_date=as_of_date, data_type="macro")
        frame = _payload_frame(macro_records).sort_values("data_as_of")
        if frame.empty:
            raise FeatureError(f"Need macro records for {symbol}.")

        current_rate = float(frame["policy_rate"].iloc[-1])
        prior_rate = float(frame["policy_rate"].iloc[-2]) if len(frame) >= 2 else current_rate
        rate_regime = 1.0 if current_rate > prior_rate else (-1.0 if current_rate < prior_rate else 0.0)

        current_vix = float(frame["vix"].iloc[-1])
        if current_vix < 15.0:
            vix_regime = 0.0
        elif current_vix < 25.0:
            vix_regime = 1.0
        else:
            vix_regime = 2.0

        return MacroFeatures(
            symbol=symbol,
            as_of_date=as_of_date,
            rate_regime=rate_regime,
            sector_relative_strength=float(frame["sector_relative_strength"].iloc[-1]),
            ff5_market_beta=float(frame["ff5_market_beta"].iloc[-1]),
            ff5_size_beta=float(frame["ff5_size_beta"].iloc[-1]),
            ff5_value_beta=float(frame["ff5_value_beta"].iloc[-1]),
            ff5_profitability_beta=float(frame["ff5_profitability_beta"].iloc[-1]),
            ff5_investment_beta=float(frame["ff5_investment_beta"].iloc[-1]),
            currency_momentum_usdinr=float(frame["currency_momentum_usdinr"].iloc[-1]),
            vix_regime=vix_regime,
        )


@dataclass
class NormalizationManager:
    scaler_root: Path = Path("data/scalers")

    def fit_on_training_window(
        self,
        training_frame: pd.DataFrame,
        feature_columns: Sequence[str],
        version: str,
    ) -> Path:
        if training_frame.empty:
            raise FeatureError("Training frame cannot be empty.")
        TimeGate().validate_no_lookahead(training_frame)
        try:
            from sklearn.preprocessing import RobustScaler
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise FeatureError("scikit-learn is required for normalization.") from exc

        scaler = RobustScaler()
        scaler.fit(training_frame.loc[:, list(feature_columns)])

        self.scaler_root.mkdir(parents=True, exist_ok=True)
        target = self.scaler_root / f"robust_scaler_{version}.pkl"
        target.write_bytes(pickle.dumps(scaler))
        self._log_mlflow(target=target, version=version, feature_columns=feature_columns)
        return target

    def _log_mlflow(self, target: Path, version: str, feature_columns: Sequence[str]) -> None:
        try:
            import mlflow
        except ImportError:
            return

        try:
            with mlflow.start_run(run_name=f"feature-scaler-{version}", nested=True):
                mlflow.log_param("scaler_version", version)
                mlflow.log_param("feature_columns", json.dumps(list(feature_columns)))
                mlflow.log_artifact(str(target))
        except Exception:
            return


@dataclass
class FeatureStore:
    offline_root: Path = Path("data/features")
    online_ttl_seconds: int = 3600
    redis_url: str | None = None
    registry_path: Path = Path("config/feature_registry.yaml")
    normalization: NormalizationManager = field(default_factory=NormalizationManager)
    time_series_pipeline: TimeSeriesFeaturePipeline = field(default_factory=TimeSeriesFeaturePipeline)
    fundamental_pipeline: FundamentalFeaturePipeline = field(default_factory=FundamentalFeaturePipeline)
    text_pipeline: TextFeaturePipeline = field(default_factory=TextFeaturePipeline)
    macro_pipeline: MacroFeaturePipeline = field(default_factory=MacroFeaturePipeline)

    def compute_features(self, symbol: str, as_of_date: date, gate: TimeGate) -> FeatureSnapshot:
        snapshot = FeatureSnapshot(
            symbol=symbol,
            as_of_date=as_of_date,
            time_series=self.time_series_pipeline.compute(symbol=symbol, as_of_date=as_of_date, gate=gate),
            fundamental=self.fundamental_pipeline.compute(symbol=symbol, as_of_date=as_of_date, gate=gate),
            text=self.text_pipeline.compute(symbol=symbol, as_of_date=as_of_date, gate=gate),
            macro=self.macro_pipeline.compute(symbol=symbol, as_of_date=as_of_date, gate=gate),
        )
        return snapshot

    def write_offline(self, snapshot: FeatureSnapshot) -> Path:
        target_dir = self.offline_root / snapshot.symbol
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{snapshot.as_of_date.isoformat()}.parquet"
        frame = pd.DataFrame([snapshot.flatten()])
        try:
            frame.to_parquet(target, index=False)
        except Exception:
            target = target_dir / f"{snapshot.as_of_date.isoformat()}.json"
            target.write_text(json.dumps([snapshot.flatten()], indent=2), encoding="utf-8")
        return target

    def write_online(self, snapshot: FeatureSnapshot) -> None:
        if not self.redis_url:
            return
        try:
            import redis
        except ImportError:
            return

        payload = snapshot.flatten()
        key = f"feat:{snapshot.symbol}:{snapshot.as_of_date.isoformat()}"
        try:
            client = redis.Redis.from_url(self.redis_url)
            client.hset(key, mapping={field: json.dumps(value) for field, value in payload.items()})
            client.expire(key, self.online_ttl_seconds)
        except Exception:
            return

    def write_registry(self) -> Path:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(self._registry_yaml(), encoding="utf-8")
        return self.registry_path

    def fit_normalizer(self, training_frame: pd.DataFrame, feature_columns: Sequence[str], version: str) -> Path:
        return self.normalization.fit_on_training_window(
            training_frame=training_frame,
            feature_columns=feature_columns,
            version=version,
        )

    def materialize(self, symbol: str, as_of_date: date, gate: TimeGate) -> FeatureSnapshot:
        snapshot = self.compute_features(symbol=symbol, as_of_date=as_of_date, gate=gate)
        self.write_registry()
        self.write_offline(snapshot)
        self.write_online(snapshot)
        return snapshot

    def _registry_yaml(self) -> str:
        registry: dict[str, dict[str, str]] = {
            "log_return_1d": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "log_return_5d": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "log_return_20d": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "rolling_mean_10d": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "rolling_mean_20d": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "rolling_mean_50d": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "rolling_std_10d": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "rolling_std_20d": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "rolling_std_50d": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "rsi_14": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "macd_line": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "macd_signal": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "macd_histogram": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "bollinger_band_width": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "realized_volatility_20d": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "volume_zscore_30d": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "bid_ask_spread": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "adv_30d": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "unusual_options_activity": {"type": "time_series", "pipeline": "TimeSeriesFeaturePipeline"},
            "revenue_growth_yoy": {"type": "fundamental", "pipeline": "FundamentalFeaturePipeline"},
            "revenue_growth_qoq": {"type": "fundamental", "pipeline": "FundamentalFeaturePipeline"},
            "ebitda_margin_trend": {"type": "fundamental", "pipeline": "FundamentalFeaturePipeline"},
            "free_cash_flow_yield": {"type": "fundamental", "pipeline": "FundamentalFeaturePipeline"},
            "roe_3y_average": {"type": "fundamental", "pipeline": "FundamentalFeaturePipeline"},
            "debt_to_equity_delta": {"type": "fundamental", "pipeline": "FundamentalFeaturePipeline"},
            "earnings_surprise_pct": {"type": "fundamental", "pipeline": "FundamentalFeaturePipeline"},
            "piotroski_f_score": {"type": "fundamental", "pipeline": "FundamentalFeaturePipeline"},
            "finbert_sentiment_score": {"type": "text", "pipeline": "TextFeaturePipeline"},
            "topic_vector": {"type": "text", "pipeline": "TextFeaturePipeline"},
            "news_volume_zscore": {"type": "text", "pipeline": "TextFeaturePipeline"},
            "earnings_call_tone_delta": {"type": "text", "pipeline": "TextFeaturePipeline"},
            "risk_flag_count": {"type": "text", "pipeline": "TextFeaturePipeline"},
            "rate_regime": {"type": "macro", "pipeline": "MacroFeaturePipeline"},
            "sector_relative_strength": {"type": "macro", "pipeline": "MacroFeaturePipeline"},
            "ff5_market_beta": {"type": "macro", "pipeline": "MacroFeaturePipeline"},
            "ff5_size_beta": {"type": "macro", "pipeline": "MacroFeaturePipeline"},
            "ff5_value_beta": {"type": "macro", "pipeline": "MacroFeaturePipeline"},
            "ff5_profitability_beta": {"type": "macro", "pipeline": "MacroFeaturePipeline"},
            "ff5_investment_beta": {"type": "macro", "pipeline": "MacroFeaturePipeline"},
            "currency_momentum_usdinr": {"type": "macro", "pipeline": "MacroFeaturePipeline"},
            "vix_regime": {"type": "macro", "pipeline": "MacroFeaturePipeline"},
        }
        lines = ["features:"]
        for name, metadata in registry.items():
            lines.append(f"  {name}:")
            lines.append(f"    type: {metadata['type']}")
            lines.append(f"    pipeline: {metadata['pipeline']}")
        return "\n".join(lines) + "\n"
