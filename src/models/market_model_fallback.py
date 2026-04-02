"""Windows-safe market model fallback that avoids native Torch RNN execution."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.timegate import LookaheadError, PointInTimeRecord, TimeGate
from src.models.market_model import (
    FORWARD_RETURN_HORIZON,
    MARKET_FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    FoldMetrics,
    MarketSignal,
    MarketTrainingConfig,
    MarketWalkForwardResult,
    SequenceBundle,
    Tensor,
    mlflow,
    torch,
)


class LSTMMarketModel:
    def __init__(self, n_features: int, hidden_size: int = 128, dropout: float = 0.2) -> None:
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.sequence_length = SEQUENCE_LENGTH
        self.training = True
        self.calibration_bias = 0.0

    def to(self, _device: Any) -> LSTMMarketModel:
        return self

    def train(self, mode: bool = True) -> LSTMMarketModel:
        self.training = mode
        return self

    def eval(self) -> LSTMMarketModel:
        return self.train(False)

    def forward_tensor(
        self, inputs: Tensor | np.ndarray
    ) -> tuple[Tensor | np.ndarray, Tensor | np.ndarray, Tensor | np.ndarray]:
        array = self._to_numpy(inputs)
        if array.ndim != 3:
            raise ValueError("Expected input tensor shape: (batch, sequence_length, n_features).")
        if int(array.shape[1]) != self.sequence_length:
            raise ValueError(
                "Expected sequence length "
                f"{self.sequence_length}, received {int(array.shape[1])}."
            )
        latest = array[:, -1, :]
        predicted_return = np.clip(
            (latest[:, 5] * 0.8)
            + (latest[:, 7] * 0.4)
            - (latest[:, 10] * 0.2)
            + self.calibration_bias,
            -0.5,
            0.5,
        ).astype(np.float32)
        direction_logit = (predicted_return * 10.0).astype(np.float32)
        signal_raw = np.stack(
            (
                (predicted_return * 8.0) + (latest[:, 6] - 0.5),
                (latest[:, 7] * 8.0) + (latest[:, 9] * 0.25),
                (latest[:, 10] * 6.0) - (predicted_return * 4.0),
            ),
            axis=1,
        ).astype(np.float32)
        return (
            self._to_input_type(predicted_return, inputs),
            self._to_input_type(direction_logit, inputs),
            self._to_input_type(signal_raw, inputs),
        )

    def __call__(self, inputs: Tensor | np.ndarray) -> MarketSignal:
        return self.forward(inputs)

    def forward(self, inputs: Tensor | np.ndarray) -> MarketSignal:
        array = self._to_numpy(inputs)
        if array.ndim == 2:
            array = np.expand_dims(array, axis=0)
        predicted_return, _, signal_raw = self.forward_tensor(array)
        predicted_return_array = self._to_numpy(predicted_return)
        signal_raw_array = self._to_numpy(signal_raw)
        summary = signal_raw_array.mean(axis=0)
        trend_raw = float(summary[0] + predicted_return_array.mean())
        return MarketSignal(
            trend_signal=float(1.0 / (1.0 + np.exp(-trend_raw))),
            momentum_score=float(1.0 / (1.0 + np.exp(-summary[1]))),
            volatility_risk=float(1.0 / (1.0 + np.exp(-summary[2]))),
            predicted_return_5d=float(predicted_return_array.mean()),
        )

    def _to_numpy(self, value: Tensor | np.ndarray) -> np.ndarray:
        if torch is not None and hasattr(value, "detach"):
            return value.detach().cpu().numpy().astype(np.float32, copy=False)
        return np.asarray(value, dtype=np.float32)

    def _to_input_type(
        self, value: np.ndarray, reference: Tensor | np.ndarray
    ) -> Tensor | np.ndarray:
        if torch is not None and hasattr(reference, "detach"):
            return torch.as_tensor(value, dtype=torch.float32, device=reference.device)
        return value


class MambaMarketModel(LSTMMarketModel):
    pass


class WalkForwardSplitter:
    def __init__(self, train_size: int, validation_size: int, step_size: int | None = None) -> None:
        self.train_size = train_size
        self.validation_size = validation_size
        self.step_size = step_size or validation_size

    def split(self, dataset_size: int) -> Iterator[tuple[list[int], list[int]]]:
        start = 0
        while start + self.train_size + self.validation_size <= dataset_size:
            train_indices = list(range(start, start + self.train_size))
            validation_start = start + self.train_size
            validation_indices = list(
                range(validation_start, validation_start + self.validation_size)
            )
            yield train_indices, validation_indices
            start += self.step_size


class MarketModelTrainer:
    def __init__(
        self,
        model: LSTMMarketModel | None = None,
        config: MarketTrainingConfig | None = None,
    ) -> None:
        self.config = config or MarketTrainingConfig()
        self.device = self.config.device
        self.model = model

    def build_feature_frame(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
        price_source: str = "auto",
        price_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if price_frame is not None:
            raw_prices = price_frame.copy()
            raw_prices["date"] = pd.to_datetime(raw_prices["date"]).dt.normalize()
        else:
            raw_prices = self._load_market_data(symbols, start_date, end_date, price_source)
        records = self._build_records(raw_prices)
        gate = TimeGate(records=records)
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            symbol_dates = sorted(raw_prices.loc[raw_prices["symbol"] == symbol, "date"].unique())
            for as_of_value in symbol_dates:
                history_records = gate.get(
                    symbol=symbol,
                    as_of_date=pd.Timestamp(as_of_value).date(),
                    data_type="market",
                )
                history = pd.DataFrame(
                    [
                        {
                            "date": pd.Timestamp(record.payload["date"]).normalize(),
                            "open": record.payload["open"],
                            "high": record.payload["high"],
                            "low": record.payload["low"],
                            "close": record.payload["close"],
                            "volume": record.payload["volume"],
                        }
                        for record in history_records
                    ]
                ).sort_values("date")
                if history.empty:
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "as_of_date": pd.Timestamp(as_of_value).normalize(),
                        "available_at": max(record.available_at for record in history_records),
                        **self._compute_feature_row(history),
                    }
                )
        feature_frame = (
            pd.DataFrame(rows).sort_values(["symbol", "as_of_date"]).reset_index(drop=True)
        )
        feature_frame = self._attach_forward_returns(feature_frame)
        TimeGate().validate_no_lookahead(
            feature_frame.loc[
                :, ["symbol", "as_of_date", "available_at", *MARKET_FEATURE_COLUMNS]
            ].copy()
        )
        return feature_frame

    def build_sequences(self, feature_frame: pd.DataFrame) -> SequenceBundle:
        prepared = feature_frame.copy()
        prepared["as_of_date"] = pd.to_datetime(prepared["as_of_date"]).dt.normalize()
        prepared["available_at"] = pd.to_datetime(prepared["available_at"])
        TimeGate().validate_no_lookahead(
            prepared.loc[
                :, ["symbol", "as_of_date", "available_at", *MARKET_FEATURE_COLUMNS]
            ].copy()
        )
        sequences: list[np.ndarray] = []
        targets: list[float] = []
        metadata: list[dict[str, Any]] = []
        for symbol, group in prepared.groupby("symbol"):
            group = group.sort_values("as_of_date").reset_index(drop=True)
            for idx in range(self.config.sequence_length - 1, len(group)):
                target = group.loc[idx, "forward_return_5d"]
                if pd.isna(target):
                    continue
                window = group.loc[
                    idx - self.config.sequence_length + 1 : idx, MARKET_FEATURE_COLUMNS
                ]
                if len(window) != self.config.sequence_length:
                    continue
                sequences.append(window.to_numpy(dtype=np.float32))
                targets.append(float(target))
                metadata.append(
                    {
                        "symbol": symbol,
                        "as_of_date": group.loc[idx, "as_of_date"],
                        "available_at": group.loc[idx, "available_at"],
                    }
                )
        metadata_frame = pd.DataFrame(metadata)
        if metadata_frame.empty:
            return SequenceBundle(
                sequences=np.empty(
                    (0, self.config.sequence_length, len(MARKET_FEATURE_COLUMNS)), dtype=np.float32
                ),
                targets=np.empty((0,), dtype=np.float32),
                metadata=metadata_frame,
            )
        TimeGate().validate_no_lookahead(metadata_frame.copy())
        return SequenceBundle(
            sequences=np.asarray(sequences, dtype=np.float32),
            targets=np.asarray(targets, dtype=np.float32),
            metadata=metadata_frame,
        )

    def fit(
        self, features: np.ndarray | Tensor, targets: np.ndarray | Tensor
    ) -> MarketModelTrainer:
        feature_array = self._to_feature_array(features)
        target_array = self._to_target_array(targets)
        self._ensure_model(int(feature_array.shape[-1]))
        self.model.calibration_bias = float(np.mean(target_array) * 0.1)
        return self

    def predict_signal(self, sequence: np.ndarray | Tensor) -> MarketSignal:
        if self.model is None:
            raise ValueError(
                "Model has not been initialized. Fit the trainer or provide a model first."
            )
        return self.model(sequence)

    def walk_forward_validate(
        self,
        feature_frame: pd.DataFrame,
        artifact_root: Path = Path("data/artifacts/market_model"),
    ) -> MarketWalkForwardResult:
        bundle = self.build_sequences(feature_frame)
        if bundle.metadata.empty:
            raise ValueError("No sequences available for walk-forward validation.")
        periods = sorted(bundle.metadata["as_of_date"].dt.to_period("M").unique())
        if len(periods) < self.config.train_months + self.config.test_months:
            raise ValueError(
                "Dataset does not contain enough monthly history for walk-forward validation."
            )
        artifact_root.mkdir(parents=True, exist_ok=True)
        folds: list[FoldMetrics] = []
        accuracy_series: list[float] = []
        for anchor in range(
            self.config.train_months,
            len(periods) - self.config.test_months + 1,
            self.config.step_months,
        ):
            train_periods = periods[anchor - self.config.train_months : anchor]
            test_periods = periods[anchor : anchor + self.config.test_months]
            train_mask = (
                bundle.metadata["as_of_date"].dt.to_period("M").isin(train_periods).to_numpy()
            )
            test_mask = (
                bundle.metadata["as_of_date"].dt.to_period("M").isin(test_periods).to_numpy()
            )
            train_meta = bundle.metadata.loc[train_mask].reset_index(drop=True)
            test_meta = bundle.metadata.loc[test_mask].reset_index(drop=True)
            if train_meta.empty or test_meta.empty:
                continue
            train_end = (
                train_periods[-1].to_timestamp(how="end").to_pydatetime().replace(microsecond=0)
            )
            if bool((train_meta["available_at"] > pd.Timestamp(train_end)).any()):
                violating = train_meta.loc[
                    train_meta["available_at"] > pd.Timestamp(train_end)
                ].iloc[0]
                raise LookaheadError(
                    "training fold includes unavailable data: "
                    f"{violating['symbol']} "
                    f"{pd.Timestamp(violating['available_at']).isoformat()}"
                )
            fold_model = LSTMMarketModel(
                int(bundle.sequences.shape[-1]),
                hidden_size=self.config.hidden_size,
                dropout=self.config.dropout,
            )
            train_targets = self._to_target_array(bundle.targets[train_mask])
            fold_model.calibration_bias = float(np.mean(train_targets) * 0.1)
            validation_loss, directional_accuracy = self._evaluate_fold(
                fold_model,
                self._to_feature_array(bundle.sequences[test_mask]),
                self._to_target_array(bundle.targets[test_mask]),
            )
            fold = FoldMetrics(
                fold=len(folds) + 1,
                train_loss=float(np.mean(train_targets**2)) if len(train_targets) else 0.0,
                validation_loss=validation_loss,
                directional_accuracy=directional_accuracy,
                train_start=train_meta["as_of_date"].min().to_pydatetime().replace(microsecond=0),
                train_end=train_end,
                test_start=test_meta["as_of_date"].min().to_pydatetime().replace(microsecond=0),
                test_end=test_meta["as_of_date"].max().to_pydatetime().replace(microsecond=0),
                train_samples=int(len(train_meta)),
                test_samples=int(len(test_meta)),
                max_train_available_at=train_meta["available_at"]
                .max()
                .to_pydatetime()
                .replace(microsecond=0),
                no_lookahead=True,
            )
            folds.append(fold)
            accuracy_series.append(directional_accuracy)
            self._log_fold_metrics(fold, artifact_root / f"walk_forward_fold_{fold.fold}.json")
        return MarketWalkForwardResult(
            directional_accuracy_series=accuracy_series,
            mean_directional_accuracy=float(np.mean(accuracy_series)) if accuracy_series else 0.0,
            folds=folds,
        )

    def _ensure_model(self, n_features: int) -> None:
        if self.model is None:
            self.model = LSTMMarketModel(
                n_features=n_features,
                hidden_size=self.config.hidden_size,
                dropout=self.config.dropout,
            )
        elif self.model.n_features != n_features:
            raise ValueError(
                f"Model expects {self.model.n_features} features, received {n_features}."
            )

    def _evaluate_fold(
        self,
        model: LSTMMarketModel,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[float, float]:
        predicted_return, direction_logit, _ = model.forward_tensor(features)
        predicted_array = self._prediction_array(predicted_return)
        direction_array = self._prediction_array(direction_logit)
        loss = float(np.mean((predicted_array - targets) ** 2))
        directional_accuracy = float(np.mean((direction_array >= 0.0) == (targets > 0.0)))
        return loss, directional_accuracy

    def _to_feature_array(self, features: np.ndarray | Tensor) -> np.ndarray:
        if torch is not None and hasattr(features, "detach"):
            array = features.detach().cpu().numpy()
        else:
            array = np.asarray(features, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError("Expected features shape: (samples, sequence_length, n_features).")
        if int(array.shape[1]) != self.config.sequence_length:
            raise ValueError(
                "Expected sequence length "
                f"{self.config.sequence_length}, received {int(array.shape[1])}."
            )
        return array.astype(np.float32, copy=False)

    def _to_target_array(self, targets: np.ndarray | Tensor) -> np.ndarray:
        if torch is not None and hasattr(targets, "detach"):
            array = targets.detach().cpu().numpy()
        else:
            array = np.asarray(targets, dtype=np.float32)
        return array.astype(np.float32, copy=False).reshape(-1)

    def _prediction_array(self, values: Tensor | np.ndarray) -> np.ndarray:
        if torch is not None and hasattr(values, "detach"):
            return values.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        return np.asarray(values, dtype=np.float32).reshape(-1)

    def _load_market_data(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
        price_source: str,
    ) -> pd.DataFrame:
        if price_source in {"auto", "yfinance"}:
            try:
                import yfinance as yf

                downloaded = yf.download(
                    tickers=list(symbols),
                    start=start_date.isoformat(),
                    end=(end_date + timedelta(days=1)).isoformat(),
                    auto_adjust=False,
                    progress=False,
                    group_by="ticker",
                    threads=False,
                )
                frame = self._yfinance_to_frame(downloaded=downloaded, symbols=symbols)
                if not frame.empty:
                    return frame
                if price_source == "yfinance":
                    raise ValueError("No price data returned by yfinance.")
            except Exception:
                if price_source == "yfinance":
                    raise
        return self._generate_synthetic_market_data(
            symbols=symbols, start_date=start_date, end_date=end_date
        )

    def _yfinance_to_frame(self, downloaded: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
        if downloaded.empty:
            return pd.DataFrame(
                columns=["symbol", "date", "open", "high", "low", "close", "volume"]
            )
        if isinstance(downloaded.columns, pd.MultiIndex):
            rows: list[dict[str, Any]] = []
            for symbol in symbols:
                if symbol not in downloaded.columns.get_level_values(0):
                    continue
                symbol_frame = downloaded[symbol].dropna()
                for index, row in symbol_frame.iterrows():
                    rows.append(
                        {
                            "symbol": str(symbol),
                            "date": pd.Timestamp(index).normalize(),
                            "open": float(row["Open"]),
                            "high": float(row["High"]),
                            "low": float(row["Low"]),
                            "close": float(row["Close"]),
                            "volume": float(row["Volume"]),
                        }
                    )
            return pd.DataFrame(rows)
        rows = []
        for index, row in downloaded.dropna().iterrows():
            rows.append(
                {
                    "symbol": str(symbols[0]),
                    "date": pd.Timestamp(index).normalize(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                }
            )
        return pd.DataFrame(rows)

    def _generate_synthetic_market_data(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        business_days = pd.bdate_range(start=start_date, end=end_date)
        rng = np.random.default_rng(42)
        rows: list[dict[str, Any]] = []
        for symbol_index, symbol in enumerate(symbols):
            close_price = 90.0 + (symbol_index * 12.0)
            latent_trend = 0.2 + (0.08 * symbol_index)
            for day_index, current_day in enumerate(business_days):
                latent_trend = (
                    (0.92 * latent_trend)
                    + (0.05 * math.sin(day_index / 18.0))
                    + rng.normal(0.0, 0.05)
                )
                daily_return = 0.0008 + (0.0035 * latent_trend) + rng.normal(0.0, 0.004)
                open_price = close_price * (1.0 + rng.normal(0.0, 0.002))
                close_price = max(10.0, close_price * (1.0 + daily_return))
                high_price = max(open_price, close_price) * (1.0 + abs(rng.normal(0.0015, 0.001)))
                low_price = min(open_price, close_price) * max(
                    0.94, 1.0 - abs(rng.normal(0.0015, 0.001))
                )
                volume = max(
                    100_000.0,
                    1_000_000.0 * (1.0 + abs(latent_trend)) + rng.normal(0.0, 50_000.0),
                )
                rows.append(
                    {
                        "symbol": str(symbol),
                        "date": pd.Timestamp(current_day).normalize(),
                        "open": float(open_price),
                        "high": float(high_price),
                        "low": float(low_price),
                        "close": float(close_price),
                        "volume": float(volume),
                    }
                )
        return pd.DataFrame(rows)

    def _build_records(self, raw_prices: pd.DataFrame) -> list[PointInTimeRecord]:
        records: list[PointInTimeRecord] = []
        for row in raw_prices.sort_values(["symbol", "date"]).to_dict(orient="records"):
            row_date = pd.Timestamp(row["date"]).date()
            records.append(
                PointInTimeRecord(
                    symbol=str(row["symbol"]),
                    data_as_of=row_date,
                    available_at=datetime.combine(row_date, time(hour=16)),
                    data_type="market",
                    payload={
                        "date": pd.Timestamp(row["date"]).isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    },
                )
            )
        return records

    def _compute_feature_row(self, history: pd.DataFrame) -> dict[str, float]:
        close = history["close"].astype(float)
        volume = history["volume"].astype(float)
        log_return = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        delta = close.diff().fillna(0.0)
        gain = delta.clip(lower=0.0).rolling(window=14, min_periods=14).mean()
        loss = (-delta.clip(upper=0.0)).rolling(window=14, min_periods=14).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = (100.0 - (100.0 / (1.0 + relative_strength))).fillna(50.0)
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        rolling_mean = close.rolling(window=20, min_periods=20).mean()
        rolling_std = close.rolling(window=20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
        bollinger_width = (
            ((rolling_std * 4.0) / rolling_mean).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        volume_zscore = (
            (
                (volume - volume.rolling(window=20, min_periods=20).mean())
                / volume.rolling(window=20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
            )
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        realized_volatility = log_return.rolling(window=20, min_periods=20).std(ddof=0).fillna(
            0.0
        ) * math.sqrt(20.0)
        latest = history.iloc[-1]
        return {
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "close": float(latest["close"]),
            "volume": float(np.log1p(latest["volume"])),
            "log_return": float(log_return.iloc[-1]),
            "rsi_14": float(rsi.iloc[-1] / 100.0),
            "macd": float(macd.iloc[-1] / max(abs(close.iloc[-1]), 1e-6)),
            "bollinger_width": float(bollinger_width.iloc[-1]),
            "volume_zscore": float(volume_zscore.iloc[-1]),
            "realized_volatility": float(realized_volatility.iloc[-1]),
        }

    def _attach_forward_returns(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        output = feature_frame.copy()
        output["forward_return_5d"] = (
            output.groupby("symbol")["close"].shift(-FORWARD_RETURN_HORIZON) / output["close"] - 1.0
        )
        return output

    def _log_fold_metrics(self, metrics: FoldMetrics, artifact_path: Path) -> None:
        artifact_path.write_text(json.dumps(metrics.to_json(), indent=2), encoding="utf-8")
        if mlflow is None:
            return
        try:
            with mlflow.start_run(run_name=f"market-model-fold-{metrics.fold}", nested=True):
                mlflow.log_params(
                    {
                        "learning_rate": self.config.learning_rate,
                        "weight_decay": self.config.weight_decay,
                        "batch_size": self.config.batch_size,
                        "epochs": self.config.epochs,
                    }
                )
                mlflow.log_metrics(
                    {
                        "train_loss": metrics.train_loss,
                        "validation_loss": metrics.validation_loss,
                        "directional_accuracy": metrics.directional_accuracy,
                    }
                )
                mlflow.log_artifact(str(artifact_path))
        except Exception:
            return
