"""Regime classification and regime-aware signal fusion."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.event_model import EventSignal
from src.models.fundamental_model import FundamentalSignal
from src.models.market_model import MarketSignal

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover - optional runtime dependency
    GaussianHMM = None

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    Tensor = object
    nn = object


REGIME_LABELS: dict[int, str] = {
    0: "bull",
    1: "bear",
    2: "sideways",
    3: "crisis",
}


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _normalize_probabilities(probabilities: list[float]) -> list[float]:
    total = sum(float(value) for value in probabilities)
    if total <= 0.0:
        return [0.25, 0.25, 0.25, 0.25]
    return [float(value) / total for value in probabilities]


def _market_components(signal: MarketSignal) -> list[float]:
    return [
        _clip01(signal.trend_signal),
        _clip01(signal.momentum_score),
        _clip01(1.0 - signal.volatility_risk),
    ]


def _event_components(signal: EventSignal) -> list[float]:
    raw_sentiment = float(signal.sentiment_score)
    sentiment_component = (
        raw_sentiment if 0.0 <= raw_sentiment <= 1.0 else _clip01((raw_sentiment + 1.0) / 2.0)
    )
    impact_component = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}.get(
        signal.event_impact, 0.5
    )
    risk_component = _clip01(1.0 - (len(signal.risk_flags) / 5.0))
    return [sentiment_component, impact_component, risk_component]


def _fundamental_components(signal: FundamentalSignal) -> list[float]:
    return [
        _clip01(getattr(signal, "fundamental_score", signal.long_term_strength)),
        _clip01(getattr(signal, "valuation_score", signal.growth_potential)),
        _clip01(getattr(signal, "financial_health", 1.0 - signal.risk_score)),
    ]


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _bias_label(score: float, *, upper: float = 0.6, lower: float = 0.4) -> str:
    if score >= upper:
        return "positive"
    if score <= lower:
        return "negative"
    return "neutral"


class FusionEngine:
    """Simple alpha registry for downstream execution checks."""

    _expected_alpha: dict[str, float] = {}

    @classmethod
    def set_expected_alpha(cls, symbol: str, alpha: float) -> None:
        cls._expected_alpha[symbol] = float(alpha)

    @classmethod
    def expected_alpha(cls, symbol: str) -> float:
        return float(cls._expected_alpha.get(symbol, 0.0))


@dataclass
class RegimeClassification:
    regime_id: int
    regime_label: str
    regime_proba: list[float]
    as_of_date: date | None = None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["regime_name"] = self.regime_name
        payload["label"] = self.regime_label
        return payload

    @property
    def regime_name(self) -> str:
        return self.regime_label

    @property
    def label(self) -> str:
        return self.regime_label


@dataclass
class LambdaWeights:
    fundamental: float
    market: float
    event: float
    lambda_cvar: float
    lambda_dd: float
    lambda_hhi: float

    @property
    def cvar(self) -> float:
        return self.lambda_cvar

    @property
    def dd(self) -> float:
        return self.lambda_dd

    @property
    def hhi(self) -> float:
        return self.lambda_hhi

    def to_json(self) -> dict[str, float]:
        return asdict(self)

    @property
    def fusion_weights(self) -> tuple[float, float, float]:
        return (self.fundamental, self.market, self.event)


@dataclass
class FusionOutput:
    combined_signal: float
    short_term_bias: str
    long_term_bias: str
    feature_importance: dict[str, float]
    lambda_weights: LambdaWeights
    regime: RegimeClassification

    def to_json(self) -> dict[str, Any]:
        return {
            "combined_signal": self.combined_signal,
            "short_term_bias": self.short_term_bias,
            "long_term_bias": self.long_term_bias,
            "feature_importance": self.feature_importance,
            "lambda_weights": self.lambda_weights.to_json(),
            "regime": self.regime.to_json(),
        }


class RegimeClassifier:
    """
    Hidden Markov Model (hmmlearn) with 4 hidden states:
    0=bull, 1=bear, 2=sideways, 3=crisis
    Features: VIX level, 20d return, volume Z-score,
              yield curve slope, credit spread
    Output: regime_id + regime_proba (4-dim vector)
    Retrain weekly on rolling 2-year window.
    """

    feature_columns = [
        "vix_level",
        "return_20d",
        "volume_zscore",
        "yield_curve_slope",
        "credit_spread",
    ]

    def __init__(
        self,
        artifact_path: Path = Path("data/models/regime_classifier.json"),
        rolling_window_days: int = 730,
        retrain_interval_days: int = 30,
    ) -> None:
        self.artifact_path = artifact_path
        self.rolling_window_days = rolling_window_days
        self.retrain_interval_days = retrain_interval_days
        self.model: Any | None = None
        self.state_map: dict[int, int] = {state: state for state in REGIME_LABELS}
        self.last_trained_on: date | None = None

    def maybe_retrain(self, history: pd.DataFrame, as_of_date: date) -> None:
        if self.last_trained_on is None or (as_of_date - self.last_trained_on) >= timedelta(
            days=self.retrain_interval_days
        ):
            self.fit(history=history, as_of_date=as_of_date)

    def fit(self, history: pd.DataFrame, as_of_date: date) -> None:
        frame = history.copy()
        frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.date
        cutoff = as_of_date - timedelta(days=self.rolling_window_days)
        frame = frame[(frame["as_of_date"] <= as_of_date) & (frame["as_of_date"] >= cutoff)]
        if frame.empty:
            raise ValueError("RegimeClassifier requires non-empty history.")

        features = frame.reindex(columns=self.feature_columns).fillna(0.0).astype(float)
        if GaussianHMM is not None and len(features) >= 20:
            model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=200, random_state=42)
            model.fit(features.to_numpy())
            states = model.predict(features.to_numpy())
            self.state_map = self._infer_state_map(frame.reset_index(drop=True), states)
            self.model = model
        else:
            self.model = None
            self.state_map = {state: state for state in REGIME_LABELS}
        self.last_trained_on = as_of_date
        self._persist_metadata()

    def predict(
        self, feature_row: dict[str, float], as_of_date: date | None = None
    ) -> RegimeClassification:
        cleaned_row = {
            column: float(feature_row.get(column, 0.0)) for column in self.feature_columns
        }
        values = pd.DataFrame([cleaned_row], columns=self.feature_columns).fillna(0.0)
        if self.model is not None:
            probabilities = self.model.predict_proba(values.to_numpy())[0].tolist()
            raw_state = int(self.model.predict(values.to_numpy())[0])
            regime_id = int(self.state_map.get(raw_state, raw_state))
            mapped_probabilities = [0.0] * 4
            for raw_index, probability in enumerate(probabilities):
                mapped_probabilities[self.state_map.get(raw_index, raw_index)] += float(probability)
            probabilities = _normalize_probabilities(mapped_probabilities)
        else:
            regime_id, probabilities = self._heuristic_predict(cleaned_row)
        return RegimeClassification(
            regime_id=regime_id,
            regime_label=REGIME_LABELS[regime_id],
            regime_proba=_normalize_probabilities([float(value) for value in probabilities]),
            as_of_date=as_of_date,
        )

    def predict_from_history(self, history: pd.DataFrame, as_of_date: date) -> RegimeClassification:
        self.maybe_retrain(history=history, as_of_date=as_of_date)
        frame = history.copy()
        frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.date
        eligible = frame[frame["as_of_date"] <= as_of_date].sort_values("as_of_date")
        if eligible.empty:
            raise ValueError("No point-in-time regime features available for the requested date.")
        latest_row = eligible.iloc[-1].to_dict()
        return self.predict(latest_row, as_of_date=as_of_date)

    def _infer_state_map(self, frame: pd.DataFrame, states: Any) -> dict[int, int]:
        summary_rows: list[tuple[int, float, float, float]] = []
        frame = frame.copy()
        frame["state"] = states
        for state in sorted(frame["state"].unique()):
            subset = frame[frame["state"] == state]
            avg_return = float(subset["return_20d"].mean())
            avg_vix = float(subset["vix_level"].mean())
            avg_credit = float(subset["credit_spread"].mean())
            summary_rows.append((int(state), avg_return, avg_vix, avg_credit))

        crisis_state = max(summary_rows, key=lambda row: (row[2], row[3]))[0]
        bull_state = max(summary_rows, key=lambda row: row[1])[0]
        bear_candidates = [row for row in summary_rows if row[0] not in {crisis_state, bull_state}]
        bear_state = (
            min(bear_candidates, key=lambda row: row[1])[0] if bear_candidates else crisis_state
        )
        assigned = {crisis_state: 3, bull_state: 0, bear_state: 1}
        for state, *_ in summary_rows:
            if state not in assigned:
                assigned[state] = 2
        return assigned

    def _heuristic_predict(self, feature_row: dict[str, float]) -> tuple[int, list[float]]:
        vix = float(feature_row.get("vix_level", 0.0))
        ret20 = float(feature_row.get("return_20d", 0.0))
        volume = float(feature_row.get("volume_zscore", 0.0))
        slope = float(feature_row.get("yield_curve_slope", 0.0))
        credit = float(feature_row.get("credit_spread", 0.0))

        if vix >= 30.0 or credit >= 2.5 or (ret20 <= -0.12 and volume >= 1.5):
            return 3, [0.05, 0.10, 0.10, 0.75]
        if ret20 <= -0.05 or slope < 0.0:
            return 1, [0.10, 0.65, 0.15, 0.10]
        if abs(ret20) < 0.03 and abs(volume) < 1.0:
            return 2, [0.15, 0.10, 0.65, 0.10]
        return 0, [0.70, 0.10, 0.15, 0.05]

    def _persist_metadata(self) -> None:
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_trained_on": self.last_trained_on.isoformat() if self.last_trained_on else None,
            "state_map": self.state_map,
        }
        self.artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class LambdaController:
    """
    Maps regime_id -> weight vector for fusion + reward fn.
    """

    _weights: dict[int, LambdaWeights] = {
        0: LambdaWeights(
            fundamental=0.50,
            market=0.30,
            event=0.20,
            lambda_cvar=0.5,
            lambda_dd=0.3,
            lambda_hhi=0.10,
        ),
        1: LambdaWeights(
            fundamental=0.30,
            market=0.45,
            event=0.25,
            lambda_cvar=1.0,
            lambda_dd=0.8,
            lambda_hhi=0.20,
        ),
        2: LambdaWeights(
            fundamental=0.45,
            market=0.35,
            event=0.20,
            lambda_cvar=0.7,
            lambda_dd=0.5,
            lambda_hhi=0.15,
        ),
        3: LambdaWeights(
            fundamental=0.20,
            market=0.50,
            event=0.30,
            lambda_cvar=2.0,
            lambda_dd=1.5,
            lambda_hhi=0.35,
        ),
    }

    def get(self, regime_id: int) -> LambdaWeights:
        if regime_id not in self._weights:
            raise ValueError(f"Unsupported regime_id: {regime_id}")
        return self._weights[regime_id]


if torch is not None:

    class AttentionFusion(nn.Module):
        """Dynamic attention-style fusion with a small learned MLP re-weighting layer."""

        def __init__(self, hidden_size: int = 32) -> None:
            super().__init__()
            self.lambda_controller = LambdaController()
            self.hidden_size = hidden_size
            self.attention_mlp = nn.Sequential(
                nn.Linear(10, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 3),
            )
            self.output_head = nn.Sequential(
                nn.Linear(13, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(
            self,
            market_signal: MarketSignal,
            event_signal: EventSignal,
            fund_signal: FundamentalSignal,
            regime: RegimeClassification,
            lambda_weights: LambdaWeights | None = None,
        ) -> FusionOutput:
            active_lambdas = lambda_weights or self.lambda_controller.get(regime.regime_id)
            device = self.output_head[0].weight.device
            market_components = self._market_tensor(market_signal).to(device)
            event_components = self._event_tensor(event_signal).to(device)
            fund_components = self._fund_tensor(fund_signal).to(device)

            market_score = market_components.mean()
            event_score = event_components.mean()
            fundamental_score = fund_components.mean()
            signal_scores = torch.stack([market_score, event_score, fundamental_score], dim=0)

            prior_weights = torch.tensor(
                [active_lambdas.market, active_lambdas.event, active_lambdas.fundamental],
                dtype=torch.float32,
                device=device,
            )
            regime_tensor = torch.tensor(
                _normalize_probabilities(list(regime.regime_proba)),
                dtype=torch.float32,
                device=device,
            )
            attention_features = torch.cat([signal_scores, prior_weights, regime_tensor], dim=0)
            attention_delta = self.attention_mlp(attention_features)
            attention_logits = torch.log(prior_weights + 1e-6) + attention_delta
            attention_weights = torch.softmax(attention_logits, dim=0)

            combined_base = torch.sum(attention_weights * signal_scores)
            output_features = torch.cat(
                [signal_scores, attention_weights, prior_weights, regime_tensor], dim=0
            )
            residual_score = torch.sigmoid(self.output_head(output_features).squeeze(-1))
            combined_signal = float(
                _clip01((0.7 * combined_base.item()) + (0.3 * residual_score.item()))
            )
            weight_vector = attention_weights.detach().cpu().tolist()
            feature_importance = {
                "market": float(weight_vector[0]),
                "event": float(weight_vector[1]),
                "fundamental": float(weight_vector[2]),
            }
            return FusionOutput(
                combined_signal=combined_signal,
                short_term_bias=self._short_term_bias(
                    market_score=float(market_score.item()), event_score=float(event_score.item())
                ),
                long_term_bias=self._long_term_bias(
                    fundamental_score=float(fundamental_score.item()),
                    combined_signal=combined_signal,
                ),
                feature_importance=feature_importance,
                lambda_weights=active_lambdas,
                regime=regime,
            )

        def _market_tensor(self, signal: MarketSignal) -> Tensor:
            return torch.tensor(_market_components(signal), dtype=torch.float32)

        def _event_tensor(self, signal: EventSignal) -> Tensor:
            return torch.tensor(_event_components(signal), dtype=torch.float32)

        def _fund_tensor(self, signal: FundamentalSignal) -> Tensor:
            return torch.tensor(_fundamental_components(signal), dtype=torch.float32)

        def _short_term_bias(self, market_score: float, event_score: float) -> str:
            return _bias_label((market_score * 0.6) + (event_score * 0.4))

        def _long_term_bias(self, fundamental_score: float, combined_signal: float) -> str:
            return _bias_label((fundamental_score * 0.7) + (combined_signal * 0.3))

        def fuse(
            self,
            market_signal: MarketSignal,
            event_signal: EventSignal,
            fund_signal: FundamentalSignal,
            regime: RegimeClassification,
            lambda_weights: LambdaWeights | None = None,
        ) -> FusionOutput:
            return self.forward(market_signal, event_signal, fund_signal, regime, lambda_weights)

else:

    class AttentionFusion:  # pragma: no cover - import guard only
        def __init__(self, *args, **kwargs) -> None:
            self.lambda_controller = LambdaController()

        def forward(
            self,
            market_signal: MarketSignal,
            event_signal: EventSignal,
            fund_signal: FundamentalSignal,
            regime: RegimeClassification,
            lambda_weights: LambdaWeights | None = None,
        ) -> FusionOutput:
            active_lambdas = lambda_weights or self.lambda_controller.get(regime.regime_id)
            market_score = _mean(_market_components(market_signal))
            event_score = _mean(_event_components(event_signal))
            fundamental_score = _mean(_fundamental_components(fund_signal))
            signal_scores = [market_score, event_score, fundamental_score]
            prior_weights = [
                active_lambdas.market,
                active_lambdas.event,
                active_lambdas.fundamental,
            ]
            regime_probabilities = _normalize_probabilities(list(regime.regime_proba))
            attention_delta = self._attention_delta(
                signal_scores, prior_weights, regime_probabilities
            )
            attention_logits = [
                math.log(weight + 1e-6) + delta
                for weight, delta in zip(prior_weights, attention_delta, strict=False)
            ]
            max_logit = max(attention_logits)
            exp_logits = [math.exp(value - max_logit) for value in attention_logits]
            total_exp = sum(exp_logits)
            normalized_attention = [value / total_exp for value in exp_logits]
            combined_base = sum(
                weight * score
                for weight, score in zip(normalized_attention, signal_scores, strict=False)
            )
            residual = self._residual_score(
                signal_scores, normalized_attention, prior_weights, regime_probabilities
            )
            combined = _clip01((0.7 * combined_base) + (0.3 * residual))
            feature_importance = {
                "market": float(normalized_attention[0]),
                "event": float(normalized_attention[1]),
                "fundamental": float(normalized_attention[2]),
            }
            return FusionOutput(
                combined_signal=float(combined),
                short_term_bias=_bias_label((market_score * 0.6) + (event_score * 0.4)),
                long_term_bias=_bias_label((fundamental_score * 0.7) + (combined * 0.3)),
                feature_importance=feature_importance,
                lambda_weights=active_lambdas,
                regime=regime,
            )

        def _attention_delta(
            self,
            signal_scores: list[float],
            prior_weights: list[float],
            regime_probabilities: list[float],
        ) -> list[float]:
            base_features = signal_scores + prior_weights + regime_probabilities
            projection = [
                0.35 * base_features[0]
                + 0.20 * base_features[3]
                + 0.15 * base_features[6]
                - 0.10 * base_features[8],
                0.30 * base_features[1]
                + 0.25 * base_features[4]
                + 0.10 * base_features[7]
                + 0.05 * base_features[9],
                0.25 * base_features[2]
                + 0.30 * base_features[5]
                + 0.10 * base_features[6]
                + 0.10 * base_features[8],
            ]
            hidden = [max(0.0, value) for value in projection]
            return [
                hidden[0] * 0.8 + hidden[1] * 0.1,
                hidden[1] * 0.8 + hidden[2] * 0.1,
                hidden[2] * 0.8 + hidden[0] * 0.1,
            ]

        def _residual_score(
            self,
            signal_scores: list[float],
            attention_weights: list[float],
            prior_weights: list[float],
            regime_probabilities: list[float],
        ) -> float:
            features = signal_scores + attention_weights + prior_weights + regime_probabilities
            score = sum(
                weight * feature
                for weight, feature in zip(
                    [0.11, 0.12, 0.15, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.06, 0.04, 0.02, 0.02],
                    features,
                    strict=False,
                )
            )
            return _clip01(score)

        def fuse(
            self,
            market_signal: MarketSignal,
            event_signal: EventSignal,
            fund_signal: FundamentalSignal,
            regime: RegimeClassification,
            lambda_weights: LambdaWeights | None = None,
        ) -> FusionOutput:
            return self.forward(market_signal, event_signal, fund_signal, regime, lambda_weights)


RegimeState = RegimeClassification
LambdaSet = LambdaWeights
FusedSignal = FusionOutput
