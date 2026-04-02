"""LSTM market-model signal engine and point-in-time walk-forward utilities."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.timegate import LookaheadError, PointInTimeRecord, TimeGate

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None

try:
    import torch
    from torch import Tensor, nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object
    nn = object
    AdamW = None
    CosineAnnealingLR = None
    DataLoader = Dataset = object

SEQUENCE_LENGTH = 60
FORWARD_RETURN_HORIZON = 5
USE_TORCH_MARKET_MODEL = torch is not None and os.name != "nt"
MARKET_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "log_return",
    "rsi_14",
    "macd",
    "bollinger_width",
    "volume_zscore",
    "realized_volatility",
]


@dataclass
class MarketSignal:
    trend_signal: float
    momentum_score: float
    volatility_risk: float
    predicted_return_5d: float

    def to_json(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class MarketTrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 4
    batch_size: int = 32
    directional_weight: float = 0.3
    device: str = "cpu"
    hidden_size: int = 128
    dropout: float = 0.2
    train_months: int = 24
    test_months: int = 1
    step_months: int = 1
    sequence_length: int = SEQUENCE_LENGTH


@dataclass
class SequenceBundle:
    sequences: np.ndarray
    targets: np.ndarray
    metadata: pd.DataFrame


@dataclass
class FoldMetrics:
    fold: int
    train_loss: float
    validation_loss: float
    directional_accuracy: float
    train_start: datetime | None = None
    train_end: datetime | None = None
    test_start: datetime | None = None
    test_end: datetime | None = None
    train_samples: int = 0
    test_samples: int = 0
    max_train_available_at: datetime | None = None
    no_lookahead: bool = True

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("train_start", "train_end", "test_start", "test_end", "max_train_available_at"):
            if payload[key] is not None:
                payload[key] = payload[key].isoformat()
        return payload


@dataclass
class MarketWalkForwardResult:
    directional_accuracy_series: list[float]
    mean_directional_accuracy: float
    folds: list[FoldMetrics]

    def to_json(self) -> dict[str, Any]:
        return {
            "directional_accuracy_series": self.directional_accuracy_series,
            "mean_directional_accuracy": self.mean_directional_accuracy,
            "folds": [fold.to_json() for fold in self.folds],
        }


if USE_TORCH_MARKET_MODEL:

    class MarketSequenceDataset(Dataset):
        def __init__(self, features: Tensor, targets: Tensor) -> None:
            self.features = features
            self.targets = targets

        def __len__(self) -> int:
            return int(self.features.shape[0])

        def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
            return self.features[index], self.targets[index]

    class LSTMMarketModel(nn.Module):
        def __init__(self, n_features: int, hidden_size: int = 128, dropout: float = 0.2) -> None:
            super().__init__()
            self.n_features = n_features
            self.sequence_length = SEQUENCE_LENGTH
            self.input_norm = nn.LayerNorm(n_features)
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=dropout,
                batch_first=True,
            )
            self.residual_proj = nn.Linear(n_features, hidden_size)
            self.output_norm = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.return_head = nn.Linear(hidden_size, 1)
            self.direction_head = nn.Linear(hidden_size, 1)
            self.signal_head = nn.Linear(hidden_size, 3)

        def forward_tensor(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            if inputs.ndim != 3:
                raise ValueError(
                    "Expected input tensor shape: (batch, sequence_length, n_features)."
                )
            if int(inputs.shape[1]) != self.sequence_length:
                raise ValueError(
                    "Expected sequence length "
                    f"{self.sequence_length}, received {int(inputs.shape[1])}."
                )
            normalized = self.input_norm(inputs.float())
            lstm_out, _ = self.lstm(normalized)
            encoded = self.output_norm(lstm_out + self.residual_proj(normalized))
            pooled = self.dropout(encoded[:, -1, :])
            predicted_return = self.return_head(pooled).squeeze(-1)
            direction_logit = self.direction_head(pooled).squeeze(-1)
            signal_raw = self.signal_head(pooled)
            return predicted_return, direction_logit, signal_raw

        def forward(self, inputs: Tensor) -> MarketSignal:
            tensor = inputs.unsqueeze(0) if inputs.ndim == 2 else inputs
            predicted_return, _, signal_raw = self.forward_tensor(tensor)
            summary = signal_raw.mean(dim=0)
            trend_raw = summary[0] + predicted_return.mean()
            return MarketSignal(
                trend_signal=float(torch.sigmoid(trend_raw).item()),
                momentum_score=float(torch.sigmoid(summary[1]).item()),
                volatility_risk=float(torch.sigmoid(summary[2]).item()),
                predicted_return_5d=float(predicted_return.mean().item()),
            )

    class WalkForwardSplitter:
        def __init__(
            self, train_size: int, validation_size: int, step_size: int | None = None
        ) -> None:
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
            self, model: LSTMMarketModel | None = None, config: MarketTrainingConfig | None = None
        ) -> None:
            self.config = config or MarketTrainingConfig()
            self.device = torch.device(self.config.device)
            self.model = model.to(self.device) if model is not None else None
            self.optimizer = None
            self.scheduler = None
            self.regression_loss = nn.MSELoss()
            self.directional_loss = nn.BCEWithLogitsLoss()

        def build_feature_frame(
            self,
            symbols: Sequence[str],
            start_date: date,
            end_date: date,
            price_source: str = "auto",
            price_frame: pd.DataFrame | None = None,
        ) -> pd.DataFrame:
            raw_prices = self._load_market_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                price_source=price_source,
                price_frame=price_frame,
            )
            records = self._build_records(raw_prices)
            gate = TimeGate(records=records)
            rows: list[dict[str, Any]] = []
            for symbol in symbols:
                symbol_dates = sorted(
                    raw_prices.loc[raw_prices["symbol"] == symbol, "date"].unique()
                )
                for as_of_date in symbol_dates:
                    history_records = gate.get(
                        symbol=symbol,
                        as_of_date=pd.Timestamp(as_of_date).date(),
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
                            "as_of_date": pd.Timestamp(as_of_date).normalize(),
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
                        (0, self.config.sequence_length, len(MARKET_FEATURE_COLUMNS)),
                        dtype=np.float32,
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
            feature_tensor = self._to_feature_tensor(features)
            target_tensor = self._to_target_tensor(targets)
            self._ensure_model(int(feature_tensor.shape[-1]))
            self._reset_optimizers()
            self._train_model(
                self.model, self._make_loader(feature_tensor, target_tensor, shuffle=False)
            )
            return self

        def predict_signal(self, sequence: np.ndarray | Tensor) -> MarketSignal:
            if self.model is None:
                raise ValueError(
                    "Model has not been initialized. Fit the trainer or provide a model first."
                )
            tensor = torch.as_tensor(sequence, dtype=torch.float32, device=self.device)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                return self.model(tensor)

        def evaluate(
            self, features: np.ndarray | Tensor, targets: np.ndarray | Tensor
        ) -> tuple[float, float]:
            if self.model is None:
                raise ValueError(
                    "Model has not been initialized. Fit the trainer or provide a model first."
                )
            return self._evaluate_model(
                self.model,
                self._make_loader(
                    self._to_feature_tensor(features),
                    self._to_target_tensor(targets),
                    shuffle=False,
                ),
            )

        def fit_walk_forward(
            self,
            features: Tensor | np.ndarray,
            target_forward_returns: Tensor | np.ndarray,
            splitter: WalkForwardSplitter,
            artifact_root: Path = Path("data/artifacts/market_model"),
        ) -> list[FoldMetrics]:
            feature_tensor = self._to_feature_tensor(features)
            target_tensor = self._to_target_tensor(target_forward_returns)
            metrics: list[FoldMetrics] = []
            artifact_root.mkdir(parents=True, exist_ok=True)
            for fold_index, (train_indices, validation_indices) in enumerate(
                splitter.split(len(feature_tensor)), start=1
            ):
                fold_model = LSTMMarketModel(
                    int(feature_tensor.shape[-1]),
                    hidden_size=self.config.hidden_size,
                    dropout=self.config.dropout,
                ).to(self.device)
                self._reset_optimizers(fold_model)
                train_loss = self._train_model(
                    fold_model,
                    self._make_loader(
                        feature_tensor[train_indices], target_tensor[train_indices], shuffle=False
                    ),
                )
                validation_loss, directional_accuracy = self._evaluate_model(
                    fold_model,
                    self._make_loader(
                        feature_tensor[validation_indices],
                        target_tensor[validation_indices],
                        shuffle=False,
                    ),
                )
                fold_metrics = FoldMetrics(
                    fold=fold_index,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    directional_accuracy=directional_accuracy,
                    train_samples=len(train_indices),
                    test_samples=len(validation_indices),
                )
                metrics.append(fold_metrics)
                self._log_fold_metrics(fold_metrics, artifact_root / f"fold_{fold_index}.json")
            return metrics

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
                ).to(self.device)
                self._reset_optimizers(fold_model)
                train_loss = self._train_model(
                    fold_model,
                    self._make_loader(
                        self._to_feature_tensor(bundle.sequences[train_mask]),
                        self._to_target_tensor(bundle.targets[train_mask]),
                        shuffle=False,
                    ),
                )
                validation_loss, directional_accuracy = self._evaluate_model(
                    fold_model,
                    self._make_loader(
                        self._to_feature_tensor(bundle.sequences[test_mask]),
                        self._to_target_tensor(bundle.targets[test_mask]),
                        shuffle=False,
                    ),
                )
                fold = FoldMetrics(
                    fold=len(folds) + 1,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    directional_accuracy=directional_accuracy,
                    train_start=train_meta["as_of_date"]
                    .min()
                    .to_pydatetime()
                    .replace(microsecond=0),
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
                mean_directional_accuracy=(
                    float(np.mean(accuracy_series)) if accuracy_series else 0.0
                ),
                folds=folds,
            )

        def _ensure_model(self, n_features: int) -> None:
            if self.model is None:
                self.model = LSTMMarketModel(
                    n_features=n_features,
                    hidden_size=self.config.hidden_size,
                    dropout=self.config.dropout,
                ).to(self.device)
            elif self.model.n_features != n_features:
                raise ValueError(
                    f"Model expects {self.model.n_features} features, received {n_features}."
                )

        def _reset_optimizers(self, model: LSTMMarketModel | None = None) -> None:
            target_model = model or self.model
            self.optimizer = AdamW(
                target_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(self.config.epochs, 1))

        def _make_loader(self, features: Tensor, targets: Tensor, shuffle: bool) -> DataLoader:
            return DataLoader(
                MarketSequenceDataset(features, targets),
                batch_size=self.config.batch_size,
                shuffle=shuffle,
            )

        def _train_model(self, model: LSTMMarketModel, loader: DataLoader) -> float:
            train_loss_value = 0.0
            for _ in range(self.config.epochs):
                model.train()
                total_loss = 0.0
                batches = 0
                for batch_features, batch_targets in loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    self.optimizer.zero_grad()
                    predicted_return, direction_logit, _ = model.forward_tensor(batch_features)
                    loss = self._compute_loss(predicted_return, direction_logit, batch_targets)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += float(loss.item())
                    batches += 1
                self.scheduler.step()
                train_loss_value = total_loss / max(batches, 1)
            return train_loss_value

        def _evaluate_model(
            self, model: LSTMMarketModel, loader: DataLoader
        ) -> tuple[float, float]:
            model.eval()
            total_loss = 0.0
            total_batches = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_features, batch_targets in loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    predicted_return, direction_logit, _ = model.forward_tensor(batch_features)
                    loss = self._compute_loss(predicted_return, direction_logit, batch_targets)
                    total_loss += float(loss.item())
                    total_batches += 1
                    predicted_direction = (torch.sigmoid(direction_logit) >= 0.5).long()
                    actual_direction = (batch_targets > 0).long()
                    correct += int((predicted_direction == actual_direction).sum().item())
                    total += int(batch_targets.shape[0])
            return total_loss / max(total_batches, 1), float(correct / max(total, 1))

        def _compute_loss(
            self, predicted_return: Tensor, direction_logit: Tensor, targets: Tensor
        ) -> Tensor:
            regression = self.regression_loss(predicted_return, targets)
            directional = self.directional_loss(direction_logit, (targets > 0).float())
            return regression + (self.config.directional_weight * directional)

        def _to_feature_tensor(self, features: np.ndarray | Tensor) -> Tensor:
            tensor = (
                features
                if isinstance(features, Tensor)
                else torch.as_tensor(features, dtype=torch.float32)
            )
            if tensor.ndim != 3:
                raise ValueError("Expected features shape: (samples, sequence_length, n_features).")
            if int(tensor.shape[1]) != self.config.sequence_length:
                raise ValueError(
                    "Expected sequence length "
                    f"{self.config.sequence_length}, received {int(tensor.shape[1])}."
                )
            return tensor.float()

        def _to_target_tensor(self, targets: np.ndarray | Tensor) -> Tensor:
            tensor = (
                targets
                if isinstance(targets, Tensor)
                else torch.as_tensor(targets, dtype=torch.float32)
            )
            return tensor.float().view(-1)

        def _load_market_data(
            self,
            symbols: Sequence[str],
            start_date: date,
            end_date: date,
            price_source: str,
            price_frame: pd.DataFrame | None,
        ) -> pd.DataFrame:
            if price_frame is not None:
                frame = price_frame.copy()
                frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
                return frame.sort_values(["symbol", "date"]).reset_index(drop=True)
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

        def _yfinance_to_frame(
            self, downloaded: pd.DataFrame, symbols: Sequence[str]
        ) -> pd.DataFrame:
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
            self, symbols: Sequence[str], start_date: date, end_date: date
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
                    high_price = max(open_price, close_price) * (
                        1.0 + abs(rng.normal(0.0015, 0.001))
                    )
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
                output.groupby("symbol")["close"].shift(-FORWARD_RETURN_HORIZON) / output["close"]
                - 1.0
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

    MambaMarketModel = LSTMMarketModel


else:
    from src.models import market_model_fallback as _market_model_fallback

    LSTMMarketModel = _market_model_fallback.LSTMMarketModel
    MambaMarketModel = _market_model_fallback.MambaMarketModel
    MarketModelTrainer = _market_model_fallback.MarketModelTrainer
    WalkForwardSplitter = _market_model_fallback.WalkForwardSplitter
